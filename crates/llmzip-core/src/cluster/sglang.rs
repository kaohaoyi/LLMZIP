use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SglangClusterConfig {
    pub model_id: String,
    pub tp_size: usize,
    pub enable_mla: bool,
}

pub struct SglangClusterAdapter {
    pub model_id: String,
    pub tp_size: usize,
}

impl SglangClusterAdapter {
    pub fn new(model_id: &str, tp_size: usize) -> Self {
        Self {
            model_id: model_id.to_string(),
            tp_size: if tp_size == 0 { 1 } else { tp_size },
        }
    }

    pub fn generate_config(&self) -> SglangClusterConfig {
        let enable_mla = self.model_id.to_lowercase().contains("deepseek");
        SglangClusterConfig {
            model_id: self.model_id.clone(),
            tp_size: self.tp_size,
            enable_mla,
        }
    }
}
