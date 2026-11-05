use crate::error::{LlmZipError, Result};
use crate::hasher::sha256_hex;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaLayer {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub config: OllamaConfig,
    pub layers: Vec<OllamaLayer>,
}

pub struct OllamaPlugin {
    pub root_dir: PathBuf,
}

impl OllamaPlugin {
    /// Detect default Ollama models directory.
    pub fn default_root() -> PathBuf {
        if let Ok(val) = env::var("OLLAMA_MODELS") {
            if !val.trim().is_empty() {
                return PathBuf::from(val);
            }
        }
        let home = env::var("USERPROFILE")
            .or_else(|_| env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".ollama").join("models")
    }

    pub fn new() -> Self {
        Self {
            root_dir: Self::default_root(),
        }
    }

    pub fn with_root(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root_dir.join("blobs")
    }

    pub fn manifests_dir(
        &self,
        registry: &str,
        namespace: &str,
        model: &str,
    ) -> PathBuf {
        self.root_dir
            .join("manifests")
            .join(registry)
            .join(namespace)
            .join(model)
    }

    /// Injects a completed GGUF file directly into Ollama's local storage and registers its manifest.
    pub fn inject_model(
        &self,
        model_name: &str,
        tag: &str,
        gguf_path: &Path,
    ) -> Result<(PathBuf, PathBuf)> {
        let blobs_dir = self.blobs_dir();
        fs::create_dir_all(&blobs_dir).map_err(LlmZipError::Io)?;

        let gguf_bytes = fs::read(gguf_path).map_err(LlmZipError::Io)?;
        let gguf_hash = sha256_hex(&gguf_bytes);
        let blob_filename = format!("sha256-{}", gguf_hash);
        let blob_path = blobs_dir.join(&blob_filename);

        // Atomic write to .unverified first, then atomic rename
        let unverified_blob = blobs_dir.join(format!("{}.unverified", blob_filename));
        fs::write(&unverified_blob, &gguf_bytes).map_err(LlmZipError::Io)?;
        fs::rename(&unverified_blob, &blob_path).map_err(LlmZipError::Io)?;

        // 2. Create config JSON blob
        let config_json = serde_json::json!({
            "model_format": "gguf",
            "model_family": "llama",
            "model_type": "quantized"
        });
        let config_bytes = serde_json::to_vec(&config_json)
            .map_err(|e| LlmZipError::PluginError {
                target: "ollama".to_string(),
                reason: format!("Failed serializing config JSON: {}", e),
            })?;
        let config_hash = sha256_hex(&config_bytes);
        let config_blob_name = format!("sha256-{}", config_hash);
        let config_blob_path = blobs_dir.join(&config_blob_name);
        fs::write(&config_blob_path, &config_bytes).map_err(LlmZipError::Io)?;

        // 3. Create and register Manifest
        let manifest = OllamaManifest {
            schema_version: 2,
            media_type: "application/vnd.docker.distribution.manifest.v2+json".to_string(),
            config: OllamaConfig {
                media_type: "application/vnd.docker.container.image.v1+json".to_string(),
                digest: format!("sha256:{}", config_hash),
                size: config_bytes.len() as u64,
            },
            layers: vec![OllamaLayer {
                media_type: "application/vnd.ollama.image.model".to_string(),
                digest: format!("sha256:{}", gguf_hash),
                size: gguf_bytes.len() as u64,
            }],
        };

        let (registry, namespace, pure_model) = Self::parse_model_name(model_name);
        let manifest_dir = self.manifests_dir(&registry, &namespace, &pure_model);
        fs::create_dir_all(&manifest_dir).map_err(LlmZipError::Io)?;

        let manifest_path = manifest_dir.join(tag);
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            LlmZipError::PluginError {
                target: "ollama".to_string(),
                reason: format!("Failed serializing manifest: {}", e),
            }
        })?;
        fs::write(&manifest_path, manifest_json).map_err(LlmZipError::Io)?;

        Ok((blob_path, manifest_path))
    }

    /// Splits a model name into `(registry, namespace, model_name)`.
    /// E.g. "qwen2.5-7b" -> ("registry.ollama.ai", "library", "qwen2.5-7b")
    /// "myuser/my-model" -> ("registry.ollama.ai", "myuser", "my-model")
    pub fn parse_model_name(name: &str) -> (String, String, String) {
        let parts: Vec<&str> = name.split('/').collect();
        match parts.len() {
            1 => (
                "registry.ollama.ai".to_string(),
                "library".to_string(),
                parts[0].to_string(),
            ),
            2 => (
                "registry.ollama.ai".to_string(),
                parts[0].to_string(),
                parts[1].to_string(),
            ),
            _ => (
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2..].join("/"),
            ),
        }
    }
}

impl Default for OllamaPlugin {
    fn default() -> Self {
        Self::new()
    }
}
