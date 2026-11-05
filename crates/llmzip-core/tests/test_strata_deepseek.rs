use llmzip_core::plugins::deepseek::{DeepSeekPlugin, MlaTensorKind};
use llmzip_core::plugins::strata::{StrataConfig, StrataPlugin};
use std::fs;

#[test]
fn test_deepseek_mla_classification_and_sglang_config() {
    let temp_dir = std::env::temp_dir().join("llmzip_test_deepseek");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    // Verify MLA tensor classification
    assert_eq!(
        DeepSeekPlugin::classify_mla_tensor("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"),
        Some(MlaTensorKind::KvAProjWithMqa)
    );
    assert_eq!(
        DeepSeekPlugin::classify_mla_tensor("model.layers.0.self_attn.kv_b_proj.weight"),
        Some(MlaTensorKind::KvBProj)
    );
    assert_eq!(
        DeepSeekPlugin::classify_mla_tensor("model.layers.0.self_attn.q_a_proj.weight"),
        Some(MlaTensorKind::QAProj)
    );
    assert_eq!(
        DeepSeekPlugin::classify_mla_tensor("model.layers.0.mlp.gate.weight"),
        None
    );

    // Generate SGLang config
    let plugin = DeepSeekPlugin::new(temp_dir.clone());
    let sglang_cfg = plugin.generate_sglang_config("deepseek-ai/DeepSeek-V3", 128, 512).unwrap();
    assert!(sglang_cfg.exists());

    let content = fs::read_to_string(&sglang_cfg).unwrap();
    let json_val: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(json_val["model_type"], "deepseek_v3");
    assert_eq!(json_val["mla_latent_dim"], 512);

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_strata_expert_partition_and_config() {
    let temp_dir = std::env::temp_dir().join("llmzip_test_strata");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let plugin = StrataPlugin::new(temp_dir.clone());

    // Create synthetic expert weights (e.g. 4 experts of 256 bytes each)
    let num_experts = 4;
    let expert_size = 256;
    let mut all_expert_data = Vec::new();
    for i in 0..num_experts {
        all_expert_data.extend(vec![i as u8; expert_size]);
    }

    let dummy_shared = temp_dir.join("temp_shared.bin");
    fs::write(&dummy_shared, b"SHARED_WEIGHTS_DATA").unwrap();

    let (experts_dir, config_path) = plugin
        .partition_experts(&all_expert_data, num_experts, &dummy_shared, None)
        .unwrap();

    // Verify individual expert_{id}.bin files
    assert!(experts_dir.exists());
    for i in 0..num_experts {
        let expert_bin = experts_dir.join(format!("expert_{}.bin", i));
        assert!(expert_bin.exists());
        let bytes = fs::read(&expert_bin).unwrap();
        assert_eq!(bytes.len(), expert_size);
        assert_eq!(bytes[0], i as u8);
    }

    // Verify strata_config.json
    assert!(config_path.exists());
    let cfg_str = fs::read_to_string(&config_path).unwrap();
    let cfg: StrataConfig = serde_json::from_str(&cfg_str).unwrap();
    assert_eq!(cfg.num_routed_experts, num_experts);
    assert_eq!(cfg.expert_file_pattern, "experts/expert_{id}.bin");

    let _ = fs::remove_dir_all(&temp_dir);
}
