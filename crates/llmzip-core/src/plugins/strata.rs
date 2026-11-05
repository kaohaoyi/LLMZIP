use crate::error::{LlmZipError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Strata configuration descriptor for 3-tier offloaded inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataConfig {
    pub model_family: String,
    pub num_routed_experts: usize,
    pub num_shared_experts: usize,
    pub expert_file_pattern: String,
    pub shared_weights: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_draft: Option<String>,
}

/// Niko1221/Strata MoE partitioner and tiered cache organizer.
pub struct StrataPlugin {
    pub out_dir: PathBuf,
}

impl StrataPlugin {
    pub fn new(out_dir: PathBuf) -> Self {
        Self { out_dir }
    }

    /// Partitions contiguous MoE weights into individual `expert_{id}.bin` files
    /// and generates `strata_config.json` for Niko1221/Strata 3-tier offloading.
    pub fn partition_experts(
        &self,
        moe_raw_bytes: &[u8],
        num_experts: usize,
        shared_weights_src: &Path,
        mtp_draft_src: Option<&Path>,
    ) -> Result<(PathBuf, PathBuf)> {
        if num_experts == 0 {
            return Err(LlmZipError::PluginError {
                target: "strata".to_string(),
                reason: "num_experts must be greater than 0".to_string(),
            });
        }
        if moe_raw_bytes.len() % num_experts != 0 {
            return Err(LlmZipError::PluginError {
                target: "strata".to_string(),
                reason: format!(
                    "Raw MoE bytes ({}) not divisible by num_experts ({})",
                    moe_raw_bytes.len(),
                    num_experts
                ),
            });
        }

        let experts_dir = self.out_dir.join("experts");
        fs::create_dir_all(&experts_dir).map_err(LlmZipError::Io)?;

        // Copy or write shared weights
        let shared_dest = self.out_dir.join("shared_weights.safetensors");
        if shared_weights_src != shared_dest && shared_weights_src.exists() {
            fs::copy(shared_weights_src, &shared_dest).map_err(LlmZipError::Io)?;
        }

        // Copy MTP draft if present
        let mut mtp_dest_name = None;
        if let Some(mtp_src) = mtp_draft_src {
            if mtp_src.exists() {
                let mtp_dest = self.out_dir.join("mtp_draft.safetensors");
                fs::copy(mtp_src, &mtp_dest).map_err(LlmZipError::Io)?;
                mtp_dest_name = Some("mtp_draft.safetensors".to_string());
            }
        }

        // Split individual expert binary slices
        let expert_stride = moe_raw_bytes.len() / num_experts;
        for i in 0..num_experts {
            let start = i * expert_stride;
            let end = start + expert_stride;
            let expert_path = experts_dir.join(format!("expert_{}.bin", i));
            fs::write(&expert_path, &moe_raw_bytes[start..end]).map_err(LlmZipError::Io)?;
        }

        // Generate strata_config.json
        let config = StrataConfig {
            model_family: "deepseek-moe".to_string(),
            num_routed_experts: num_experts,
            num_shared_experts: 1,
            expert_file_pattern: "experts/expert_{id}.bin".to_string(),
            shared_weights: "shared_weights.safetensors".to_string(),
            mtp_draft: mtp_dest_name,
        };

        let config_path = self.out_dir.join("strata_config.json");
        let formatted = serde_json::to_string_pretty(&config).map_err(|e| {
            LlmZipError::PluginError {
                target: "strata".to_string(),
                reason: format!("Failed serializing strata_config.json: {}", e),
            }
        })?;
        fs::write(&config_path, formatted).map_err(LlmZipError::Io)?;

        Ok((experts_dir, config_path))
    }
}
