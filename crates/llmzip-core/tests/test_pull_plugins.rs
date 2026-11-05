use llmzip_core::plugins::desktop::{JanPlugin, LmStudioPlugin};
use llmzip_core::plugins::ollama::{OllamaManifest, OllamaPlugin};
use std::fs;

#[test]
fn test_ollama_storage_and_manifest() {
    let temp_dir = std::env::temp_dir().join("llmzip_test_ollama_root");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    // Create a dummy GGUF file
    let gguf_path = temp_dir.join("test_model.gguf");
    fs::write(&gguf_path, b"GGUF_TEST_BLOB_PAYLOAD_12345").unwrap();

    let plugin = OllamaPlugin::with_root(temp_dir.clone());
    let (blob_path, manifest_path) = plugin
        .inject_model("qwen2.5-7b", "latest", &gguf_path)
        .unwrap();

    // 1. Verify blob file exists in blobs/
    assert!(blob_path.exists());
    assert!(blob_path.to_str().unwrap().contains("sha256-"));
    let blob_bytes = fs::read(&blob_path).unwrap();
    assert_eq!(blob_bytes, b"GGUF_TEST_BLOB_PAYLOAD_12345");

    // 2. Verify manifest exists in manifests/registry.ollama.ai/library/qwen2.5-7b/latest
    assert!(manifest_path.exists());
    let manifest_content = fs::read_to_string(&manifest_path).unwrap();
    let manifest: OllamaManifest = serde_json::from_str(&manifest_content).unwrap();

    assert_eq!(manifest.schema_version, 2);
    assert_eq!(manifest.layers.len(), 1);
    assert!(manifest.layers[0].digest.starts_with("sha256:"));
    assert_eq!(manifest.layers[0].size, blob_bytes.len() as u64);

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_desktop_lmstudio_and_jan() {
    let temp_root = std::env::temp_dir().join("llmzip_test_desktop_root");
    let _ = fs::remove_dir_all(&temp_root);
    fs::create_dir_all(&temp_root).unwrap();

    // 1. Test LM Studio layout
    let lm_plugin = LmStudioPlugin::with_root(temp_root.join("lm-studio"));
    let lm_target = lm_plugin
        .resolve_model_path("meta-llama", "Llama-3-8B", "q4_k_m")
        .unwrap();
    assert!(lm_target.to_str().unwrap().contains("meta-llama"));
    assert!(lm_target.to_str().unwrap().contains("Llama-3-8B-q4_k_m.gguf"));

    // 2. Test Jan layout and model.json generation
    let dummy_gguf = temp_root.join("sample.gguf");
    fs::write(&dummy_gguf, b"GGUF_SAMPLE").unwrap();

    let jan_plugin = JanPlugin::with_root(temp_root.join("jan"));
    let (jan_model_file, jan_json_file) = jan_plugin
        .setup_model_entry("deepseek-r1-7b", &dummy_gguf)
        .unwrap();

    assert!(jan_model_file.exists());
    assert!(jan_json_file.exists());

    let json_content = fs::read_to_string(&jan_json_file).unwrap();
    let json_val: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    assert_eq!(json_val["id"], "deepseek-r1-7b");
    assert_eq!(json_val["format"], "gguf");

    let _ = fs::remove_dir_all(&temp_root);
}
