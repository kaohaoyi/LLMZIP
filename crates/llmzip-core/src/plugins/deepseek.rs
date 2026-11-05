use crate::error::{LlmZipError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// DeepSeek MLA (Multi-Head Latent Attention) tensor classification kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlaTensorKind {
    KvAProjWithMqa,
    KvBProj,
    QAProj,
    QBProj,
}

pub struct DeepSeekPlugin {
    pub out_dir: PathBuf,
}

impl DeepSeekPlugin {
    pub fn new(out_dir: PathBuf) -> Self {
        Self { out_dir }
    }

    /// Identifies whether a tensor belongs to DeepSeek Multi-Head Latent Attention (MLA).
    pub fn classify_mla_tensor(name: &str) -> Option<MlaTensorKind> {
        if name.contains("kv_a_proj_with_mqa") {
            Some(MlaTensorKind::KvAProjWithMqa)
        } else if name.contains("kv_b_proj") {
            Some(MlaTensorKind::KvBProj)
        } else if name.contains("q_a_proj") {
            Some(MlaTensorKind::QAProj)
        } else if name.contains("q_b_proj") {
            Some(MlaTensorKind::QBProj)
        } else {
            None
        }
    }

    /// Emits SGLang-compatible model configuration for DeepSeek MLA execution.
    pub fn generate_sglang_config(
        &self,
        model_name: &str,
        kv_lora_rank: usize,
        mla_latent_dim: usize,
    ) -> Result<PathBuf> {
        fs::create_dir_all(&self.out_dir).map_err(LlmZipError::Io)?;

        let config_json = serde_json::json!({
            "model_name": model_name,
            "model_type": "deepseek_v3",
            "kv_lora_rank": kv_lora_rank,
            "mla_latent_dim": mla_latent_dim,
            "attention_backend": "triton_mla",
            "rope_scaling": {
                "type": "yarn",
                "factor": 40.0
            }
        });

        let config_path = self.out_dir.join("sglang_config.json");
        let formatted = serde_json::to_string_pretty(&config_json).map_err(|e| {
            LlmZipError::PluginError {
                target: "deepseek/sglang".to_string(),
                reason: format!("Failed serializing SGLang config: {}", e),
            }
        })?;

        fs::write(&config_path, formatted).map_err(LlmZipError::Io)?;
        Ok(config_path)
    }
}
