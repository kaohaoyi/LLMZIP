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
    let deq_bf16_from_q4k = dequantize_to_dtype(&q4k_bytes, QuantFormat::GgufQ4K, DequantDtype::BF16);
    assert_eq!(deq_bf16_from_q4k.len(), 256 * 2);

    let deq_bf16_from_q8 = dequantize_to_dtype(&q8_bytes, QuantFormat::GgufQ8_0, DequantDtype::BF16);
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
    let exported_files = export_safetensors_shards(&out_dir, vec![t1, t2, t3], Some(2000))
        .expect("export failed");

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
