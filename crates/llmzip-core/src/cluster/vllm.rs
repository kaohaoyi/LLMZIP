use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmClusterManifest {
    pub model_id: String,
    pub tp_world_size: usize,
    pub tensor_keys: Vec<String>,
}

pub struct VllmClusterAdapter {
    pub model_id: String,
    pub tp_world_size: usize,
}

impl VllmClusterAdapter {
    pub fn new(model_id: &str, tp_world_size: usize) -> Self {
        Self {
            model_id: model_id.to_string(),
            tp_world_size: if tp_world_size == 0 { 1 } else { tp_world_size },
        }
    }

    pub fn generate_manifest(&self, tensors: Vec<(String, Vec<usize>)>) -> VllmClusterManifest {
        let keys = tensors.into_iter().map(|(k, _)| k).collect();
        VllmClusterManifest {
            model_id: self.model_id.clone(),
            tp_world_size: self.tp_world_size,
            tensor_keys: keys,
        }
    }
}
