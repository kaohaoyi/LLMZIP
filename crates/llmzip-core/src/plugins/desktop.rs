use crate::error::{LlmZipError, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// LM Studio Desktop Model Resolver and Importer.
pub struct LmStudioPlugin {
    pub root_dir: PathBuf,
}

impl LmStudioPlugin {
    pub fn default_root() -> PathBuf {
        let home = env::var("USERPROFILE")
            .or_else(|_| env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".cache").join("lm-studio").join("models")
    }

    pub fn new() -> Self {
        Self {
            root_dir: Self::default_root(),
        }
    }

    pub fn with_root(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn resolve_model_path(
        &self,
        publisher: &str,
        model_name: &str,
        preset: &str,
    ) -> Result<PathBuf> {
        let dir = self.root_dir.join(publisher).join(model_name);
        fs::create_dir_all(&dir).map_err(LlmZipError::Io)?;
        let file_name = format!("{}-{}.gguf", model_name, preset);
        Ok(dir.join(file_name))
    }
}

impl Default for LmStudioPlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// Jan.ai Desktop Model Resolver and Manifest (`model.json`) Generator.
pub struct JanPlugin {
    pub root_dir: PathBuf,
}

impl JanPlugin {
    pub fn default_root() -> PathBuf {
        let home = env::var("USERPROFILE")
            .or_else(|_| env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".jan").join("models")
    }

    pub fn new() -> Self {
        Self {
            root_dir: Self::default_root(),
        }
    }

    pub fn with_root(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn setup_model_entry(
        &self,
        model_id: &str,
        gguf_source: &Path,
    ) -> Result<(PathBuf, PathBuf)> {
        let model_dir = self.root_dir.join(model_id);
        fs::create_dir_all(&model_dir).map_err(LlmZipError::Io)?;

        let dest_gguf = model_dir.join("model.gguf");
        if gguf_source != dest_gguf {
            fs::copy(gguf_source, &dest_gguf).map_err(LlmZipError::Io)?;
        }

        let model_json = serde_json::json!({
            "id": model_id,
            "format": "gguf",
            "sources": [{ "filename": "model.gguf" }],
            "settings": {
                "ctx_len": 4096,
                "temperature": 0.7
            }
        });

        let json_path = model_dir.join("model.json");
        let json_str = serde_json::to_string_pretty(&model_json).map_err(|e| {
            LlmZipError::PluginError {
                target: "jan".to_string(),
                reason: format!("Failed serializing model.json: {}", e),
            }
        })?;
        fs::write(&json_path, json_str).map_err(LlmZipError::Io)?;

        Ok((dest_gguf, json_path))
    }
}

impl Default for JanPlugin {
    fn default() -> Self {
        Self::new()
    }
}
