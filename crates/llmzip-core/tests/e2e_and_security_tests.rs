use llmzip_core::error::LlmZipError;
use llmzip_core::hasher::{sha256_hex, StreamingSha256};
use llmzip_core::layout::{
    build_deterministic_layout, QuantFormat, QuantPreset, GGUF_MAGIC_UNVERIFIED, GGUF_MAGIC_VERIFIED,
};
use llmzip_core::preflight::{
    build_virtual_tensor_space, parse_and_validate_header_len, validate_shard_filename, ShardDescriptor,
};
use llmzip_core::quant::gguf_simd::{
    decode_and_validate_upstream_to_f32, dequantize_to_f32, f32_to_f16_clamped, quantize_f32_slice,
};
use llmzip_core::streamer::{
     verify_http_range_response, EngineConfig, HttpResponseMeta, StreamingQuantizeEngine,
};
use llmzip_core::wal::MicroWal;
use std::fs;
use std::path::PathBuf;

/// Helper to build a valid synthetic `.safetensors` binary image in memory.
fn build_synthetic_safetensors(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> (Vec<u8>, u64) {
    let mut header_map = serde_json::Map::new();
    let mut data_payload = Vec::new();
    let mut offset_cursor = 0u64;

    for (name, dtype, shape, raw_bytes) in tensors {
        let start = offset_cursor;
        let end = start + (raw_bytes.len() as u64);
        offset_cursor = end;
        data_payload.extend_from_slice(raw_bytes);

        let mut t_obj = serde_json::Map::new();
        t_obj.insert("dtype".to_string(), serde_json::Value::String((*dtype).to_string()));
        t_obj.insert(
            "shape".to_string(),
            serde_json::Value::Array(shape.iter().map(|&d| serde_json::Value::from(d as u64)).collect()),
        );
        t_obj.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::from(start), serde_json::Value::from(end)]),
        );
        header_map.insert((*name).to_string(), serde_json::Value::Object(t_obj));
    }

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header_map)).unwrap();
    let header_len = header_bytes.len() as u64;

    let mut full_file = Vec::with_capacity(8 + header_bytes.len() + data_payload.len());
    full_file.extend_from_slice(&header_len.to_le_bytes());
    full_file.extend_from_slice(&header_bytes);
    full_file.extend_from_slice(&data_payload);

    (full_file, header_len)
}

fn encode_f32_slice_as_f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_f16_clamped(v).to_le_bytes());
    }
    out
}

fn temp_test_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("llmzip_test_{}_{}", std::process::id(), name));
    p
}

#[test]
fn test_e01_header_bomb_guardrail() {
    // 1. Header length = u64::MAX (18 Exabytes) -> must reject immediately with HeaderTooLarge
    let malicious_len = u64::MAX.to_le_bytes();
    let res = parse_and_validate_header_len("model.safetensors", malicious_len, 1024 * 1024);
    assert!(matches!(res, Err(LlmZipError::HeaderTooLarge { .. })));

    // 2. Header length = 101 MiB (> 100 MiB limit) -> must reject
    let over_100mb = (101u64 * 1024 * 1024).to_le_bytes();
    let res2 = parse_and_validate_header_len("model.safetensors", over_100mb, 500 * 1024 * 1024);
    assert!(matches!(res2, Err(LlmZipError::HeaderTooLarge { .. })));

    // 3. Header length = 0 -> must reject
    let zero_len = 0u64.to_le_bytes();
    let res3 = parse_and_validate_header_len("model.safetensors", zero_len, 1024);
    assert!(matches!(res3, Err(LlmZipError::HeaderTooLarge { .. })));
}

#[test]
fn test_e02_offset_overlap_and_gap_guardrail() {
    let bad_json = br#"{
        "t1": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]},
        "t2": {"dtype": "F16", "shape": [4, 4], "data_offsets": [16, 48]}
    }"#;
    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: 8 + (bad_json.len() as u64) + 48,
        header_len: bad_json.len() as u64,
        expected_sha256_hex: "00".repeat(32),
    };
    let res = build_virtual_tensor_space(vec![(desc, bad_json.to_vec())], false, 8);
    assert!(matches!(res, Err(LlmZipError::InvalidTensorBounds { .. })));
}

#[test]
fn test_e12_path_traversal_and_pickle_rejection() {
    assert!(matches!(
        validate_shard_filename("../../etc/passwd.safetensors"),
        Err(LlmZipError::SecurityPathViolation { .. })
    ));
    assert!(matches!(
        validate_shard_filename("pytorch_model.bin"),
        Err(LlmZipError::SecurityPathViolation { .. })
    ));
    assert!(matches!(
        validate_shard_filename("model.pt"),
        Err(LlmZipError::SecurityPathViolation { .. })
    ));
    assert!(validate_shard_filename("model-00001-of-00004.safetensors").is_ok());
}

#[test]
fn test_e05_non_aligned_dimension_fallback_and_moe_strided_layout() {
    let w_256 = vec![0u8; 2 * 256 * 2]; // shape [2, 256] F16 -> divisible by 256 -> Q4_K / Q6_K
    let w_96 = vec![0u8; 2 * 96 * 2]; // shape [2, 96] F16 -> not div by 256, div by 32 -> fallback Q8_0
    let w_33 = vec![0u8; 2 * 33 * 2]; // shape [2, 33] F16 -> not div by 32 -> fallback F16
    let exp0 = vec![0u8; 2 * 256 * 2]; // MoE expert 0
    let exp1 = vec![0u8; 2 * 256 * 2]; // MoE expert 1

    let (raw_file, header_len) = build_synthetic_safetensors(&[
        ("model.embed_tokens.weight", "F16", vec![2, 96], w_96),
        ("model.layers.0.self_attn.v_proj.weight", "F16", vec![2, 256], w_256),
        ("model.layers.0.self_attn.q_proj.weight", "F16", vec![2, 33], w_33),
        (
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "F16",
            vec![2, 256],
            exp0,
        ),
        (
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            "F16",
            vec![2, 256],
            exp1,
        ),
    ]);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: raw_file.len() as u64,
        header_len,
        expected_sha256_hex: sha256_hex(&raw_file),
    };
    let header_slice = raw_file[8..8 + (header_len as usize)].to_vec();
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], true, 2).unwrap();
    let plan = build_deterministic_layout(&vtas, QuantPreset::GgufQ4KM, "mixtral").unwrap();

    // Check [2, 96] fallback to Q8_0
    assert_eq!(plan.write_targets[0].assigned_quant, QuantFormat::GgufQ8_0);
    assert!(plan.write_targets[0].fallback_applied);
    // Check Weight Tying [E07] assigned secondary offset for `output.weight`
    assert!(plan.write_targets[0].tied_secondary_offset.is_some());

    // Check `v_proj` [2, 256] got high-precision Q6_K in Q4_K_M preset
    assert_eq!(plan.write_targets[1].assigned_quant, QuantFormat::GgufQ6K);
    assert!(!plan.write_targets[1].fallback_applied);

    // Check `q_proj` [2, 33] fallback to F16
    assert_eq!(plan.write_targets[2].assigned_quant, QuantFormat::GgufF16);
    assert!(plan.write_targets[2].fallback_applied);

    // Check MoE experts 0 and 1 share `blk.0.ffn_gate_exps.weight` with exact strided sub-offset!
    assert_eq!(plan.write_targets[3].output_tensor_name, "blk.0.ffn_gate_exps.weight");
    assert_eq!(plan.write_targets[4].output_tensor_name, "blk.0.ffn_gate_exps.weight");
    assert_eq!(
        plan.write_targets[4].file_write_offset,
        plan.write_targets[3].file_write_offset + plan.write_targets[3].quantized_byte_len
    );
}

#[test]
fn test_sha256_112b_midstate_and_micro_wal_roundtrip() {
    let payload: Vec<u8> = (0..10_000).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    let full_expected = sha256_hex(&payload);

    // Feed first 3,333 bytes (unaligned to 64-byte block boundary!)
    let split_point = 3_333;
    let mut h1 = StreamingSha256::new();
    h1.update(&payload[..split_point]);

    // Export 112-Byte Mid-State & save into Micro-WAL
    let mid = h1.export_midstate();
    assert_eq!(mid.to_bytes().len(), 112);

    let mut wal = MicroWal::new(128, mid);
    wal.current_shard_idx = 0;
    wal.shard_confirmed_offset = split_point as u64;
    wal.mark_tensor_completed(0);
    wal.mark_tensor_completed(42);

    let wal_bytes = wal.to_bytes();
    assert!(wal_bytes.len() < 4096, "Micro-WAL must be < 4KB");

    // Verify CRC32 detects 1-bit corruption
    let mut corrupted = wal_bytes.clone();
    corrupted[50] ^= 0x01;
    assert!(matches!(
        MicroWal::from_bytes(&corrupted),
        Err(LlmZipError::WalCorrupted { .. })
    ));

    // Restore from valid WAL and feed remaining bytes
    let restored_wal = MicroWal::from_bytes(&wal_bytes).unwrap();
    assert!(restored_wal.is_tensor_completed(0));
    assert!(restored_wal.is_tensor_completed(42));
    assert!(!restored_wal.is_tensor_completed(1));

    let mut h2 = StreamingSha256::from_midstate(restored_wal.sha256_midstate);
    h2.update(&payload[split_point..]);
    assert_eq!(h2.finalize_hex(), full_expected);
}

#[test]
fn test_e04_and_e11_nan_inf_poisoning_and_bf16_dynamic_range() {
    // 1. F16 NaN (0x7E00) must fail when sanitize=false, succeed when sanitize=true [E04]
    let nan_bytes = 0x7E00u16.to_le_bytes();
    let err = decode_and_validate_upstream_to_f32(
        "bad_tensor",
        &nan_bytes,
        llmzip_core::preflight::UpstreamDType::F16,
        false,
    );
    assert!(matches!(err, Err(LlmZipError::NumericalPoisoning { .. })));

    let sanitized = decode_and_validate_upstream_to_f32(
        "bad_tensor",
        &nan_bytes,
        llmzip_core::preflight::UpstreamDType::F16,
        true,
    )
    .unwrap();
    assert_eq!(sanitized[0], 0.0);

    // 2. BF16 value = 100,000.0 (> FP16 max 65504.0) must widen to FP32 and clamp cleanly to 65504.0 [E11]
    let large_f32 = 100_000.0f32;
    let bf16_bits = (large_f32.to_bits() >> 16) as u16;
    let decoded = decode_and_validate_upstream_to_f32(
        "bf16_tensor",
        &bf16_bits.to_le_bytes(),
        llmzip_core::preflight::UpstreamDType::BF16,
        false,
    )
    .unwrap();
    assert_eq!(decoded[0], 65504.0);
}

#[test]
fn test_gguf_q4k_q6k_q8_0_and_awq_quant_dequant_fidelity() {
    let data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.025).collect();

    for fmt in [
        QuantFormat::GgufQ8_0,
        QuantFormat::GgufQ6K,
        QuantFormat::GgufQ4K,
        QuantFormat::AwqInt4G128,
    ] {
        let q = quantize_f32_slice(&data, fmt);
        let deq = dequantize_to_f32(&q, fmt);
        assert_eq!(deq.len(), data.len());

        let mse: f32 = data
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / (data.len() as f32);
        assert!(mse < 0.05, "Quantization format {:?} MSE too high: {}", fmt, mse);
    }
}

#[test]
fn test_e2e_streaming_gguf_chaos_disconnect_zero_redownload_resume_and_e03_commit_lock() {
    let n_elem = 2 * 256;
    let embd_f32: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.01).sin()).collect();
    let q_f32: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.02).cos()).collect();
    let v_f32: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.03).sin()).collect();
    let norm_f32: Vec<f32> = vec![1.0f32; 256];

    let (raw_shard, header_len) = build_synthetic_safetensors(&[
        (
            "model.embed_tokens.weight",
            "F16",
            vec![2, 256],
            encode_f32_slice_as_f16_bytes(&embd_f32),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "F16",
            vec![2, 256],
            encode_f32_slice_as_f16_bytes(&q_f32),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "F16",
            vec![2, 256],
            encode_f32_slice_as_f16_bytes(&v_f32),
        ),
        (
            "model.norm.weight",
            "F16",
            vec![256],
            encode_f32_slice_as_f16_bytes(&norm_f32),
        ),
    ]);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: raw_shard.len() as u64,
        header_len,
        expected_sha256_hex: sha256_hex(&raw_shard),
    };
    let header_slice = raw_shard[8..8 + (header_len as usize)].to_vec();
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], true, 8).unwrap();
    let plan = build_deterministic_layout(&vtas, QuantPreset::GgufQ4KM, "llama").unwrap();

    let out_path = temp_test_path("chaos_resume.gguf");
    let mut wal_os = out_path.as_os_str().to_os_string();
    wal_os.push(".wal");
    let wal_path = PathBuf::from(wal_os);
    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_file(&wal_path);

    // Phase A: Start streaming with prime chunk size (137 bytes) and interrupt after 2 tensors!
    {
        let mut engine = StreamingQuantizeEngine::new_or_resume(
            vtas.clone(),
            plan.clone(),
            &out_path,
            EngineConfig::default(),
        )
        .unwrap();

        // Verify [E10] Concurrent access lock prevents a second process from opening the same target file!
        let concurrent_attempt = StreamingQuantizeEngine::new_or_resume(
            vtas.clone(),
            plan.clone(),
            &out_path,
            EngineConfig::default(),
        );
        assert!(matches!(concurrent_attempt, Err(LlmZipError::FileLocked { .. })));

        let committed = engine
            .stream_shard_slice(
                0,
                HttpResponseMeta {
                    status_code: 200,
                    content_range_start: None,
                },
                &raw_shard,
                137,
                Some(2), // Disconnect right after tensor #1 (`q_proj`) finishes!
            )
            .unwrap();
        assert_eq!(committed, 2);
        assert!(!engine.is_committed());
    }

    // Verify on-disk file still has "UNVF" unverified magic [E03] and `.wal` exists
    let disk_bytes_mid = fs::read(&out_path).unwrap();
    assert_eq!(&disk_bytes_mid[0..4], &GGUF_MAGIC_UNVERIFIED);
    assert_eq!(disk_bytes_mid.len() as u64, plan.total_output_file_size);
    assert!(wal_path.exists());

    // Phase B: Resume from `.wal` using HTTP 206 Partial Content from exact `resume_cursor` (0 bytes re-downloaded!)
    {
        let mut engine = StreamingQuantizeEngine::new_or_resume(
            vtas.clone(),
            plan.clone(),
            &out_path,
            EngineConfig::default(),
        )
        .unwrap();

        let (shard_idx, resume_offset) = engine.resume_cursor();
        assert_eq!(shard_idx, 0);
        assert_eq!(resume_offset, vtas.tensors[1].stream_abs_end);

        let committed_rest = engine
            .stream_shard_slice(
                0,
                HttpResponseMeta {
                    status_code: 206,
                    content_range_start: Some(resume_offset),
                },
                &raw_shard[resume_offset as usize..],
                199,
                None,
            )
            .unwrap();
        assert_eq!(committed_rest, 2);
        assert!(engine.is_committed());
    }

    // Verify Magic Commit Lock flipped to "GGUF" [E03] and `.wal` was removed
    let final_bytes = fs::read(&out_path).unwrap();
    assert_eq!(&final_bytes[0..4], &GGUF_MAGIC_VERIFIED);
    assert!(!wal_path.exists());

    let _ = fs::remove_file(&out_path);
}

#[test]
fn test_e03_mitm_tampered_shard_quarantine() {
    let w = vec![0u8; 2 * 256 * 2];
    let (mut raw_shard, header_len) =
        build_synthetic_safetensors(&[("model.embed_tokens.weight", "F16", vec![2, 256], w)]);
    let original_sha = sha256_hex(&raw_shard);

    // MITM attacker flips 1 bit in the weight payload
    let last_idx = raw_shard.len() - 1;
    raw_shard[last_idx] ^= 0x01;

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: raw_shard.len() as u64,
        header_len,
        expected_sha256_hex: original_sha,
    };
    let header_slice = raw_shard[8..8 + (header_len as usize)].to_vec();
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8).unwrap();
    let plan = build_deterministic_layout(&vtas, QuantPreset::GgufQ4KM, "llama").unwrap();

    let out_path = temp_test_path("tampered_mitm.gguf");
    let _ = fs::remove_file(&out_path);

    let mut engine =
        StreamingQuantizeEngine::new_or_resume(vtas, plan, &out_path, EngineConfig::default()).unwrap();
    let res = engine.stream_shard_slice(
        0,
        HttpResponseMeta {
            status_code: 200,
            content_range_start: None,
        },
        &raw_shard,
        512,
        None,
    );

    assert!(matches!(res, Err(LlmZipError::Sha256Mismatch { .. })));
    assert!(!engine.is_committed());

    // Verify the poisoned file was quarantined ("POISONED") and truncated
    let quarantined = fs::read(&out_path).unwrap();
    assert_eq!(&quarantined, b"POISONED");

    let mut wal_os = out_path.as_os_str().to_os_string();
    wal_os.push(".wal");
    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_file(PathBuf::from(wal_os));
}

#[test]
fn test_e09_http_range_violation_and_200_ok_fallback() {
    // 1. Mismatched 206 Content-Range start -> Err(HttpRangeViolation)
    let err = verify_http_range_response(
        "shard.safetensors",
        4096,
        HttpResponseMeta {
            status_code: 206,
            content_range_start: Some(2048),
        },
    );
    assert!(matches!(err, Err(LlmZipError::HttpRangeViolation { .. })));

    // 2. Mirror ignores Range and returns 200 OK -> returns discard_bytes = 4096
    let discard = verify_http_range_response(
        "shard.safetensors",
        4096,
        HttpResponseMeta {
            status_code: 200,
            content_range_start: None,
        },
    )
    .unwrap();
    assert_eq!(discard, 4096);
}

#[test]
fn test_e2e_awq_sliding_window_calibration_and_crash_resume_reforward() {
    let hidden_dim = 128;
    let vocab_size = 16;
    let embd_f32: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| ((i as f32) * 0.05).sin() * 0.5)
        .collect();
    let q_f32: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.013).cos() * 0.25)
        .collect();
    let k_f32: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.017).sin() * 0.25)
        .collect();

    let (raw_shard, header_len) = build_synthetic_safetensors(&[
        (
            "model.embed_tokens.weight",
            "F16",
            vec![vocab_size, hidden_dim],
            encode_f32_slice_as_f16_bytes(&embd_f32),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "F16",
            vec![hidden_dim, hidden_dim],
            encode_f32_slice_as_f16_bytes(&q_f32),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "F16",
            vec![hidden_dim, hidden_dim],
            encode_f32_slice_as_f16_bytes(&k_f32),
        ),
    ]);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: raw_shard.len() as u64,
        header_len,
        expected_sha256_hex: sha256_hex(&raw_shard),
    };
    let header_slice = raw_shard[8..8 + (header_len as usize)].to_vec();
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8).unwrap();
    let plan = build_deterministic_layout(&vtas, QuantPreset::AwqW4G128, "llama").unwrap();

    let out_path = temp_test_path("awq_sliding_resume.gguf");
    let mut wal_os = out_path.as_os_str().to_os_string();
    wal_os.push(".wal");
    let wal_path = PathBuf::from(wal_os);
    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_file(&wal_path);

    let cfg = EngineConfig {
        awq_calib_samples: 2,
        awq_calib_seq_len: 4,
        ..Default::default()
    };

    // 1. Stream `embed_tokens` and `q_proj` with AWQ Sliding-Window Calibration, then crash!
    {
        let mut engine =
            StreamingQuantizeEngine::new_or_resume(vtas.clone(), plan.clone(), &out_path, cfg.clone())
                .unwrap();
        let count = engine
            .stream_shard_slice(
                0,
                HttpResponseMeta {
                    status_code: 200,
                    content_range_start: None,
                },
                &raw_shard,
                256,
                Some(2),
            )
            .unwrap();
        assert_eq!(count, 2);
    }

    // 2. Resume from `.wal` -> reconstructs AWQ calibration hidden state from on-disk quantized layers and calibrates `k_proj`!
    {
        let mut engine =
            StreamingQuantizeEngine::new_or_resume(vtas.clone(), plan.clone(), &out_path, cfg).unwrap();
        let (_, offset) = engine.resume_cursor();
        let count = engine
            .stream_shard_slice(
                0,
                HttpResponseMeta {
                    status_code: 206,
                    content_range_start: Some(offset),
                },
                &raw_shard[offset as usize..],
                256,
                None,
            )
            .unwrap();
        assert_eq!(count, 1);
        assert!(engine.is_committed());
    }

    let _ = fs::remove_file(&out_path);
}

#[test]
fn test_e06_multi_shard_streaming_and_out_of_order_topology_detection() {
    let w0 = encode_f32_slice_as_f16_bytes(&vec![0.1f32; 2 * 256]);
    let w1 = encode_f32_slice_as_f16_bytes(&vec![0.2f32; 2 * 256]);

    // Put `layers.0` in Shard 0 and `embed_tokens` in Shard 1 to trigger out-of-order topology detection (Question 6)
    // and multi-shard sequential streaming [E06]!
    let (raw_s0, hlen0) = build_synthetic_safetensors(&[(
        "model.layers.0.self_attn.q_proj.weight",
        "F16",
        vec![2, 256],
        w0,
    )]);
    let (raw_s1, hlen1) = build_synthetic_safetensors(&[(
        "model.embed_tokens.weight",
        "F16",
        vec![2, 256],
        w1,
    )]);

    let d0 = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model-00001-of-00002.safetensors".to_string(),
        total_file_size: raw_s0.len() as u64,
        header_len: hlen0,
        expected_sha256_hex: sha256_hex(&raw_s0),
    };
    let d1 = ShardDescriptor {
        shard_idx: 1,
        shard_name: "model-00002-of-00002.safetensors".to_string(),
        total_file_size: raw_s1.len() as u64,
        header_len: hlen1,
        expected_sha256_hex: sha256_hex(&raw_s1),
    };

    let vtas = build_virtual_tensor_space(
        vec![
            (d0, raw_s0[8..8 + (hlen0 as usize)].to_vec()),
            (d1, raw_s1[8..8 + (hlen1 as usize)].to_vec()),
        ],
        false,
        8,
    )
    .unwrap();

    // Verify Question 6 Preflight Topology Check detected that `embed_tokens` is out of topological order
    assert!(!vtas.topology.is_strictly_topological);
    assert_eq!(vtas.topology.out_of_order_prefetch_ids, vec![1]);

    let plan = build_deterministic_layout(&vtas, QuantPreset::GgufQ8_0, "llama").unwrap();
    let out_path = temp_test_path("multishard_e06.gguf");
    let _ = fs::remove_file(&out_path);

    let mut engine =
        StreamingQuantizeEngine::new_or_resume(vtas, plan, &out_path, EngineConfig::default()).unwrap();

    // Stream Shard 0 -> should NOT be committed yet because Shard 1 remains
    engine
        .stream_shard_slice(
            0,
            HttpResponseMeta {
                status_code: 200,
                content_range_start: None,
            },
            &raw_s0,
            128,
            None,
        )
        .unwrap();
    assert!(!engine.is_committed());

    // Stream Shard 1 -> now both shards pass SHA-256 and Magic Commit Lock flips to "GGUF"!
    engine
        .stream_shard_slice(
            1,
            HttpResponseMeta {
                status_code: 200,
                content_range_start: None,
            },
            &raw_s1,
            128,
            None,
        )
        .unwrap();
    assert!(engine.is_committed());

    let _ = fs::remove_file(&out_path);
}

#[test]
fn test_rope_permutation_and_qkv_fusion_in_ringbuf() {
    use llmzip_core::ringbuf::{fuse_qkv_weights_f32, permute_rope_weights_f32, LayerRingBuffer};

    // 1 head, head_dim = 4, hidden_dim = 2:
    // Row 0: [1, 2], Row 1: [3, 4], Row 2: [5, 6], Row 3: [7, 8]
    // Split-half -> Interleaved pairs: Row 0, Row 2, Row 1, Row 3
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let permuted = permute_rope_weights_f32(&src, 1, 4, 2);
    assert_eq!(permuted, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);

    let fused = fuse_qkv_weights_f32(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);
    assert_eq!(fused, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Verify ring buffer budget enforcement
    let mut rb = LayerRingBuffer::new(16);
    assert!(rb.store_layer_tensor("small".to_string(), vec![1.0, 2.0]).is_ok());
    assert!(rb
        .store_layer_tensor("overflow".to_string(), vec![1.0, 2.0, 3.0])
        .is_err());
}

#[test]
fn test_e2m1_and_fp8_e4m3fn_bit_exact_encoding() {
    use llmzip_core::quant::nvfp4_microscale::{
        e2m1_to_f32, f32_to_e2m1, f32_to_fp8_e4m3fn, fp8_e4m3fn_to_f32, E2M1_VALUES,
    };

    // 1. Verify exact 16 round-trip values of E2M1
    for &expected in &E2M1_VALUES {
        let code_pos = f32_to_e2m1(expected);
        assert_eq!(e2m1_to_f32(code_pos), expected);

        if expected > 0.0 {
            let code_neg = f32_to_e2m1(-expected);
            assert_eq!(e2m1_to_f32(code_neg), -expected);
        }
    }

    // 2. Verify FP8 E4M3FN subnormals and maximum finite clamp (448.0)
    let max_fp8 = f32_to_fp8_e4m3fn(500.0);
    assert_eq!(fp8_e4m3fn_to_f32(max_fp8), 448.0);

    let subnormal_f32 = 0.00390625; // 2^(-8)
    let subnormal_code = f32_to_fp8_e4m3fn(subnormal_f32);
    let recovered_subnormal = fp8_e4m3fn_to_f32(subnormal_code);
    assert!((recovered_subnormal - subnormal_f32).abs() < 1e-4);
}

#[test]
fn test_mxfp4_e8m0_scaling() {
    use llmzip_core::quant::nvfp4_microscale::{
        dequantize_mxfp4_b32, e8m0_to_f32, f32_to_e8m0_shared_exp, quantize_mxfp4_b32,
    };

    // 1. Verify E8M0 power-of-two scaling without mantissa
    assert_eq!(e8m0_to_f32(0), 0.0);
    assert!(e8m0_to_f32(255).is_nan());
    assert_eq!(e8m0_to_f32(127), 1.0); // 2^(127 - 127) = 1.0
    assert_eq!(e8m0_to_f32(128), 2.0); // 2^(128 - 127) = 2.0
    assert_eq!(e8m0_to_f32(126), 0.5); // 2^(126 - 127) = 0.5

    // 2. Verify OCP v1.0 formula: E = clamp(floor(log2(amax)) - emax + bias) with emax=2, bias=127
    // For amax in [4.0, 8.0): floor(log2(amax)) = 2. E = 2 - 2 + 127 = 127 -> scale = 1.0
    let (exp_6, s_6) = f32_to_e8m0_shared_exp(6.0);
    assert_eq!(exp_6, 127);
    assert_eq!(s_6, 1.0);

    // For amax in [8.0, 16.0): floor(log2(amax)) = 3. E = 3 - 2 + 127 = 128 -> scale = 2.0
    let (exp_8, s_8) = f32_to_e8m0_shared_exp(8.0);
    assert_eq!(exp_8, 128);
    assert_eq!(s_8, 2.0);

    let (exp_15, s_15) = f32_to_e8m0_shared_exp(15.9);
    assert_eq!(exp_15, 128);
    assert_eq!(s_15, 2.0);

    // Zero / subnormal underflow threshold
    let (exp_0, s_0) = f32_to_e8m0_shared_exp(0.0);
    assert_eq!(exp_0, 0);
    assert_eq!(s_0, 0.0);

    // 3. Block-32 roundtrip fidelity
    let block: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let q = quantize_mxfp4_b32(&block);
    assert_eq!(q.len(), 17); // 1B E8M0 + 16B nibbles
    let deq = dequantize_mxfp4_b32(&q);
    assert_eq!(deq.len(), 32);
    for (orig, rec) in block.iter().zip(deq.iter()) {
        let diff = (orig - rec).abs();
        // With scale = 0.5, maximum E2M1 discretization step is (6.0 - 4.0) * 0.5 = 1.0.
        // Round to nearest guarantees max absolute error <= half step (0.5).
        assert!(
            diff <= 0.5 + 1e-5,
            "orig: {}, rec: {}, diff: {}",
            orig,
            rec,
            diff
        );
    }
}

#[test]
fn test_nvfp4_sb256_b16_mxfp4_and_fp8_reconstruction_snr() {
    let data: Vec<f32> = (0..512)
        .map(|i| {
            let base = ((i as f32) * 0.04).sin() * 3.5;
            // Inject realistic outliers at every 64th element
            if i % 64 == 0 {
                base * 8.0
            } else {
                base
            }
        })
        .collect();

    for fmt in [
        QuantFormat::Nvfp4Sb256,
        QuantFormat::Nvfp4B16,
        QuantFormat::Mxfp4B32,
        QuantFormat::Fp8E4M3B32,
    ] {
        let q = quantize_f32_slice(&data, fmt);
        let deq = dequantize_to_f32(&q, fmt);
        assert_eq!(deq.len(), data.len());

        let mse: f32 = data
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / (data.len() as f32);
        assert!(
            mse < 0.25,
            "Format {:?} MSE too high on outlier distribution: {}",
            fmt,
            mse
        );
    }
}

#[test]
fn test_nvfp4_dimension_fallback_256_to_16_to_f16() {
    let w_256 = vec![0u8; 2 * 256 * 2]; // shape [2, 256] F16 -> div by 256 -> Nvfp4Sb256
    let w_48 = vec![0u8; 2 * 48 * 2];   // shape [2, 48] F16 -> not div by 256, div by 16 -> fallback Nvfp4B16
    let w_33 = vec![0u8; 2 * 33 * 2];   // shape [2, 33] F16 -> not div by 16 -> fallback GgufF16

    let (raw_file, header_len) = build_synthetic_safetensors(&[
        ("model.layers.0.self_attn.q_proj.weight", "F16", vec![2, 256], w_256),
        ("model.layers.0.self_attn.k_proj.weight", "F16", vec![2, 48], w_48),
        ("model.layers.0.self_attn.v_proj.weight", "F16", vec![2, 33], w_33),
    ]);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: raw_file.len() as u64,
        header_len,
        expected_sha256_hex: sha256_hex(&raw_file),
    };
    let header_slice = raw_file[8..8 + (header_len as usize)].to_vec();
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8).unwrap();
    let plan = build_deterministic_layout(&vtas, QuantPreset::Nvfp4, "llama").unwrap();

    assert_eq!(plan.write_targets[0].assigned_quant, QuantFormat::Nvfp4Sb256);
    assert!(!plan.write_targets[0].fallback_applied);

    assert_eq!(plan.write_targets[1].assigned_quant, QuantFormat::Nvfp4B16);
    assert!(plan.write_targets[1].fallback_applied);

    assert_eq!(plan.write_targets[2].assigned_quant, QuantFormat::GgufF16);
    assert!(plan.write_targets[2].fallback_applied);
}

#[test]
fn test_e2e_streaming_nvfp4_and_mxfp4_chaos_resume() {
    let n_elem = 2 * 256;
    let w0: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.03).cos() * 2.0).collect();
    let w1: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.05).sin() * 2.0).collect();

    let (raw_shard, header_len) = build_synthetic_safetensors(&[
        (
            "model.layers.0.self_attn.q_proj.weight",
            "F16",
            vec![2, 256],
            encode_f32_slice_as_f16_bytes(&w0),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "F16",
            vec![2, 256],
            encode_f32_slice_as_f16_bytes(&w1),
        ),
    ]);

    let desc = ShardDescriptor {
        shard_idx: 0,
        shard_name: "model.safetensors".to_string(),
        total_file_size: raw_shard.len() as u64,
        header_len,
        expected_sha256_hex: sha256_hex(&raw_shard),
    };
    let header_slice = raw_shard[8..8 + (header_len as usize)].to_vec();
    let vtas = build_virtual_tensor_space(vec![(desc, header_slice)], false, 8).unwrap();
    let plan = build_deterministic_layout(&vtas, QuantPreset::Nvfp4, "llama").unwrap();

    let out_path = temp_test_path("nvfp4_chaos.gguf");
    let mut wal_os = out_path.as_os_str().to_os_string();
    wal_os.push(".wal");
    let wal_path = PathBuf::from(wal_os);
    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_file(&wal_path);

    // 1. Interrupt after tensor 0
    {
        let mut engine = StreamingQuantizeEngine::new_or_resume(
            vtas.clone(),
            plan.clone(),
            &out_path,
            EngineConfig::default(),
        )
        .unwrap();
        let committed = engine
            .stream_shard_slice(
                0,
                HttpResponseMeta {
                    status_code: 200,
                    content_range_start: None,
                },
                &raw_shard,
                113,
                Some(1),
            )
            .unwrap();
        assert_eq!(committed, 1);
        assert!(!engine.is_committed());
    }

    // 2. Resume from WAL and finish
    {
        let mut engine = StreamingQuantizeEngine::new_or_resume(
            vtas.clone(),
            plan.clone(),
            &out_path,
            EngineConfig::default(),
        )
        .unwrap();
        let (_, offset) = engine.resume_cursor();
        let committed = engine
            .stream_shard_slice(
                0,
                HttpResponseMeta {
                    status_code: 206,
                    content_range_start: Some(offset),
                },
                &raw_shard[offset as usize..],
                113,
                None,
            )
            .unwrap();
        assert_eq!(committed, 1);
        assert!(engine.is_committed());
    }

    let _ = fs::remove_file(&out_path);
}

#[test]
fn test_nvfp4_split_tensors_for_vllm_sglang() {
    use llmzip_core::quant::nvfp4_microscale::quantize_nvfp4_split_tensors;

    let weights: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.02).sin() * 4.0).collect();
    let split = quantize_nvfp4_split_tensors(&weights);

    // 512 elements -> 256 bytes of packed E2M1 u8
    assert_eq!(split.weight_u8.len(), 256);
    // 512 elements / 16 = 32 micro-scales in FP8 E4M3FN
    assert_eq!(split.weight_scale_e4m3.len(), 32);
    // Positive finite global scale
    assert!(split.weight_scale_2_f32 > 0.0 && split.weight_scale_2_f32.is_finite());
}

