use crate::dequant::engine::DequantDtype;
use crate::error::{LlmZipError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DEFAULT_MAX_SHARD_SIZE: u64 = 5 * 1024 * 1024 * 1024; // 5 GiB

#[derive(Debug, Clone)]
pub struct ExportTensor {
    pub name: String,
    pub dtype: DequantDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SafeTensorsIndex {
    pub metadata: IndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub total_size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TensorEntryHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Streams and writes out sharded SafeTensors files directly into `out_dir` with `model.safetensors.index.json`.
pub fn export_safetensors_shards(
    out_dir: &Path,
    tensors: Vec<ExportTensor>,
    max_shard_size: Option<u64>,
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(out_dir)?;

    let max_bytes = max_shard_size.unwrap_or(DEFAULT_MAX_SHARD_SIZE);

    // Group tensors into shards based on max_bytes
    let mut shards: Vec<Vec<ExportTensor>> = Vec::new();
    let mut current_shard: Vec<ExportTensor> = Vec::new();
    let mut current_shard_bytes: u64 = 0;

    for t in tensors {
        let t_len = t.data.len() as u64;
        if !current_shard.is_empty() && current_shard_bytes + t_len > max_bytes {
            shards.push(std::mem::take(&mut current_shard));
            current_shard_bytes = 0;
        }
        current_shard_bytes += t_len;
        current_shard.push(t);
    }
    if !current_shard.is_empty() {
        shards.push(current_shard);
    }

    if shards.is_empty() {
        return Ok(Vec::new());
    }

    let total_shards = shards.len();
    let mut weight_map = BTreeMap::new();
    let mut total_model_bytes: u64 = 0;
    let mut written_paths = Vec::new();

    for (idx, shard_tensors) in shards.into_iter().enumerate() {
        let shard_idx = idx + 1;
        let shard_filename = format!("model-{:05}-of-{:05}.safetensors", shard_idx, total_shards);
        let shard_path = out_dir.join(&shard_filename);

        for t in &shard_tensors {
            weight_map.insert(t.name.clone(), shard_filename.clone());
            total_model_bytes += t.data.len() as u64;
        }

        write_single_safetensors_file(&shard_path, shard_tensors)?;
        written_paths.push(shard_path);
    }

    // Write model.safetensors.index.json
    let index = SafeTensorsIndex {
        metadata: IndexMetadata {
            total_size: total_model_bytes,
        },
        weight_map,
    };

    let index_path = out_dir.join("model.safetensors.index.json");
    let index_json = serde_json::to_string_pretty(&index).map_err(|e| {
        LlmZipError::SecurityPathViolation {
            offending_path: index_path.display().to_string(),
            reason: format!("Failed to serialize index json: {}", e),
        }
    })?;
    fs::write(&index_path, index_json)?;
    written_paths.push(index_path);

    Ok(written_paths)
}

/// Writes a single standalone `.safetensors` file according to the official spec.
pub fn write_single_safetensors_file(path: &Path, tensors: Vec<ExportTensor>) -> Result<()> {
    let mut header_map = serde_json::Map::new();

    // Add __metadata__
    let mut meta = serde_json::Map::new();
    meta.insert("format".to_string(), serde_json::Value::String("pt".to_string()));
    header_map.insert("__metadata__".to_string(), serde_json::Value::Object(meta));

    let mut current_offset: u64 = 0;
    for t in &tensors {
        let start = current_offset;
        let end = start + t.data.len() as u64;
        current_offset = end;

        let entry = TensorEntryHeader {
            dtype: t.dtype.safetensors_dtype_str().to_string(),
            shape: t.shape.clone(),
            data_offsets: [start, end],
        };
        let val = serde_json::to_value(&entry).map_err(|e| LlmZipError::SecurityPathViolation {
            offending_path: path.display().to_string(),
            reason: format!("Failed to serialize tensor header: {}", e),
        })?;
        header_map.insert(t.name.clone(), val);
    }

    let header_json = serde_json::to_string(&header_map).map_err(|e| {
        LlmZipError::SecurityPathViolation {
            offending_path: path.display().to_string(),
            reason: format!("Failed to format header JSON: {}", e),
        }
    })?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;

    for t in tensors {
        file.write_all(&t.data)?;
    }

    file.flush()?;
    Ok(())
}
