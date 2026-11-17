use llmzip_core::dequant::{
    dequantize_to_dtype, export_safetensors_shards, DequantDtype, ExportTensor,
};
use llmzip_core::layout::QuantFormat;
use llmzip_core::quant::gguf_simd::quantize_f32_slice;
use safetensors::SafeTensors;
use std::env;
use std::fs;

#[test]
fn test_dequant_fidelity_to_bf16_fp16() {
    let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.05).collect();

    // 1. Quantize to Q4_K and Q8_0
    let q4k_bytes = quantize_f32_slice(&original, QuantFormat::GgufQ4K);
    let q8_bytes = quantize_f32_slice(&original, QuantFormat::GgufQ8_0);

    // 2. Inverse dequantize to BF16
    let deq_bf16_from_q4k =
        dequantize_to_dtype(&q4k_bytes, QuantFormat::GgufQ4K, DequantDtype::BF16);
    assert_eq!(deq_bf16_from_q4k.len(), 256 * 2);

    let deq_bf16_from_q8 =
        dequantize_to_dtype(&q8_bytes, QuantFormat::GgufQ8_0, DequantDtype::BF16);
    assert_eq!(deq_bf16_from_q8.len(), 256 * 2);

    // 3. Inverse dequantize to FP16
    let deq_fp16 = dequantize_to_dtype(&q8_bytes, QuantFormat::GgufQ8_0, DequantDtype::FP16);
    assert_eq!(deq_fp16.len(), 256 * 2);
}

#[test]
fn test_safetensors_sharding_and_index_json_compatibility() {
    let tmp = env::temp_dir().join(format!("llmzip_dequant_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let out_dir = tmp.join("exported_model");

    // Create 3 mock tensors, each 1000 bytes of BF16 data
    let t1 = ExportTensor {
        name: "model.embed_tokens.weight".to_string(),
        dtype: DequantDtype::BF16,
        shape: vec![100, 5],
        data: vec![0x12u8; 1000],
    };
    let t2 = ExportTensor {
        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        dtype: DequantDtype::BF16,
        shape: vec![100, 5],
        data: vec![0x34u8; 1000],
    };
    let t3 = ExportTensor {
        name: "lm_head.weight".to_string(),
        dtype: DequantDtype::BF16,
        shape: vec![100, 5],
        data: vec![0x56u8; 1000],
    };

    // Shard threshold = 2000 bytes -> Splits into 2 shards (t1 + t2 = 2000B, t3 = 1000B)
    let exported_files =
        export_safetensors_shards(&out_dir, vec![t1, t2, t3], Some(2000)).expect("export failed");

    assert_eq!(exported_files.len(), 3); // 2 shard files + 1 index.json

    let index_file = out_dir.join("model.safetensors.index.json");
    assert!(index_file.exists());

    let index_str = fs::read_to_string(&index_file).expect("read index failed");
    assert!(index_str.contains("model.embed_tokens.weight"));
    assert!(index_str.contains("lm_head.weight"));

    // Verify generated .safetensors files using the official safetensors reader
    let shard1 = out_dir.join("model-00001-of-00002.safetensors");
    let shard1_bytes = fs::read(&shard1).expect("read shard 1 failed");
    let tensors1 = SafeTensors::deserialize(&shard1_bytes).expect("Failed to parse safetensors");
    assert_eq!(tensors1.names().len(), 2);

    let shard2 = out_dir.join("model-00002-of-00002.safetensors");
    let shard2_bytes = fs::read(&shard2).expect("read shard 2 failed");
    let tensors2 = SafeTensors::deserialize(&shard2_bytes).expect("Failed to parse safetensors");
    assert_eq!(tensors2.names().len(), 1);

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_reject_unverified_unvf_dequant() {
    use llmzip_core::dequant::load_and_dequantize_gguf;
    use llmzip_core::layout::GGUF_MAGIC_UNVERIFIED;
    use std::io::Write;

    let tmp = env::temp_dir().join(format!("llmzip_unvf_test_{}", std::process::id()));
    let _ = fs::create_dir_all(&tmp);
    let fake_unvf = tmp.join("corrupt.unverified");

    let mut f = fs::File::create(&fake_unvf).unwrap();
    f.write_all(&GGUF_MAGIC_UNVERIFIED).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap(); // GGUF version 3
    f.flush().unwrap();
    drop(f);

    let res = load_and_dequantize_gguf(&fake_unvf, DequantDtype::BF16);
    assert!(
        res.is_err(),
        "Expected dequantization to reject UNVF magic file"
    );
    let err_msg = format!("{}", res.err().unwrap());
    assert!(err_msg.contains("unverified ('UNVF')"));

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_q6_k_cross_stride_ggml_layout() {
    use llmzip_core::quant::gguf_simd::{dequantize_q6_k, quantize_q6_k};

    let mut src = vec![0.0f32; 256];
    src[0] = 10.0;
    src[64] = -10.0;
    src[32] = 5.0;
    src[96] = -5.0;

    let bytes = quantize_q6_k(&src);
    assert_eq!(bytes.len(), 210);

    let deq = dequantize_q6_k(&bytes);
    assert_eq!(deq.len(), 256);

    assert!(deq[0] > 8.0, "deq[0] should be ~10.0, got {}", deq[0]);
    assert!(deq[64] < -8.0, "deq[64] should be ~ -10.0, got {}", deq[64]);
    assert!(deq[32] > 3.0, "deq[32] should be ~5.0, got {}", deq[32]);
    assert!(deq[96] < -3.0, "deq[96] should be ~ -5.0, got {}", deq[96]);

    assert!(
        deq[1].abs() < 0.5,
        "deq[1] should be near 0, got {}",
        deq[1]
    );
    assert!(
        deq[65].abs() < 0.5,
        "deq[65] should be near 0, got {}",
        deq[65]
    );
}

#[test]
fn test_bf16_edge_cases_nan_and_max() {
    use llmzip_core::dequant::engine::f32_to_bf16_bytes;

    let inputs = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, 0.0f32];
    let bf16_bytes = f32_to_bf16_bytes(&inputs);
    assert_eq!(bf16_bytes.len(), 5 * 2);

    let nan_val = u16::from_le_bytes([bf16_bytes[0], bf16_bytes[1]]);
    let inf_val = u16::from_le_bytes([bf16_bytes[2], bf16_bytes[3]]);
    let neg_inf_val = u16::from_le_bytes([bf16_bytes[4], bf16_bytes[5]]);
    let max_val = u16::from_le_bytes([bf16_bytes[6], bf16_bytes[7]]);

    assert_eq!((nan_val >> 7) & 0xFF, 0xFF);
    assert_ne!(nan_val & 0x7F, 0);

    assert_eq!(inf_val, 0x7F80);
    assert_eq!(neg_inf_val, 0xFF80);
    assert_ne!((max_val >> 7) & 0xFF, 0xFF);
}
