use crate::error::{LlmZipError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Hard security limit for SafeTensors JSON header: 100 MiB [E01].
pub const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100 * 1024 * 1024;
/// Maximum number of tensors permitted in a single model repository.
pub const MAX_TOTAL_TENSORS: usize = 65_536;

/// Supported upstream element data types in `.safetensors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpstreamDType {
    F16,
    BF16,
    F32,
}

impl UpstreamDType {
    pub fn from_str_dtype(s: &str) -> Option<Self> {
        match s {
            "F16" | "float16" => Some(Self::F16),
            "BF16" | "bfloat16" => Some(Self::BF16),
            "F32" | "float32" => Some(Self::F32),
            _ => None,
        }
    }

    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Topological role of a tensor for execution order & quantization assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorRole {
    Embedding,
    Norm,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    FfnGate,
    FfnUp,
    FfnDown,
    MoeRouter,
    MoeExpert { expert_idx: usize, num_experts: usize },
    LmHead,
    Other,
}

/// Descriptor for a single upstream shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardDescriptor {
    pub shard_idx: usize,
    pub shard_name: String,
    pub total_file_size: u64,
    pub header_len: u64,
    pub expected_sha256_hex: String,
}

/// Entry in the Virtual Tensor Address Space (VTAS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualTensorEntry {
    pub tensor_id: usize,
    pub hf_name: String,
    pub shard_idx: usize,
    /// Relative byte offset inside the data section of the `.safetensors` file.
    pub data_rel_start: u64,
    pub data_rel_end: u64,
    /// Absolute byte offset in the `.safetensors` HTTP stream (`8 + header_len + data_rel_start`).
    pub stream_abs_start: u64,
    pub stream_abs_end: u64,
    pub dtype: UpstreamDType,
    pub shape: Vec<usize>,
    pub layer_idx: Option<usize>,
    pub role: TensorRole,
}

impl VirtualTensorEntry {
    pub fn num_elements(&self) -> Result<u64> {
        if self.shape.is_empty() {
            return Err(LlmZipError::InvalidShape {
                tensor_name: self.hf_name.clone(),
                shape: self.shape.clone(),
            });
        }
        let mut prod: u64 = 1;
        for &dim in &self.shape {
            if dim == 0 {
                return Err(LlmZipError::InvalidShape {
                    tensor_name: self.hf_name.clone(),
                    shape: self.shape.clone(),
                });
            }
            prod = prod.checked_mul(dim as u64).ok_or_else(|| {
                LlmZipError::InvalidTensorBounds {
                    shard_name: format!("shard_{}", self.shard_idx),
                    tensor_name: self.hf_name.clone(),
                    reason: "Integer overflow in shape product".to_string(),
                }
            })?;
        }
        Ok(prod)
    }

    pub fn inner_dim(&self) -> usize {
        *self.shape.last().unwrap_or(&0)
    }
}

/// Result of the Preflight Topology Check (Question 6).
#[derive(Debug, Clone)]
pub struct TopologyCheckReport {
    /// True if physical shard/offset order matches topological execution order (Embedding -> Layer 0 -> Layer 1 ...).
    pub is_strictly_topological: bool,
    /// List of tensor IDs that must be pre-fetched via targeted Range requests before starting local AWQ calibration.
    pub out_of_order_prefetch_ids: Vec<usize>,
    /// True if `tie_word_embeddings` is active (`lm_head.weight` shares data with `model.embed_tokens.weight`).
    pub tie_word_embeddings: bool,
}

/// Complete Virtual Tensor Address Space (VTAS) built during Phase 1 Preflight.
#[derive(Debug, Clone)]
pub struct VirtualTensorSpace {
    pub shards: Vec<ShardDescriptor>,
    pub tensors: Vec<VirtualTensorEntry>,
    pub topology: TopologyCheckReport,
}

/// Validates a shard filename against path traversal and forbidden `.bin` pickle formats [E12].
pub fn validate_shard_filename(name: &str) -> Result<()> {
    if name.is_empty()
        || name.contains("..")
        || name.contains('/')
        || name.contains('\\')
        || name.contains(':')
        || name.starts_with('.')
    {
        return Err(LlmZipError::SecurityPathViolation {
            offending_path: name.to_string(),
            reason: "Path traversal characters or directory separators are prohibited".to_string(),
        });
    }
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".bin") || lower.ends_with(".pt") || lower.ends_with(".pth") || lower.ends_with(".pkl") {
        return Err(LlmZipError::SecurityPathViolation {
            offending_path: name.to_string(),
            reason: "PyTorch Pickle (.bin/.pt/.pth) files are strictly rejected to prevent RCE".to_string(),
        });
    }
    if !lower.ends_with(".safetensors") {
        return Err(LlmZipError::SecurityPathViolation {
            offending_path: name.to_string(),
            reason: "Only .safetensors shards are permitted".to_string(),
        });
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(LlmZipError::SecurityPathViolation {
            offending_path: name.to_string(),
            reason: "Shard filename contains non-whitelisted characters".to_string(),
        });
    }
    Ok(())
}

/// Validates the 8-byte little-endian header length prefix of a `.safetensors` file [E01].
pub fn parse_and_validate_header_len(shard_name: &str, len_bytes: [u8; 8], total_file_size: u64) -> Result<u64> {
    validate_shard_filename(shard_name)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(LlmZipError::HeaderTooLarge {
            shard_name: shard_name.to_string(),
            header_len,
            max_allowed: MAX_SAFETENSORS_HEADER_BYTES,
        });
    }
    let min_file_size = 8u64.checked_add(header_len).ok_or_else(|| LlmZipError::HeaderTooLarge {
        shard_name: shard_name.to_string(),
        header_len,
        max_allowed: MAX_SAFETENSORS_HEADER_BYTES,
    })?;
    if min_file_size > total_file_size {
        return Err(LlmZipError::InvalidTensorBounds {
            shard_name: shard_name.to_string(),
            tensor_name: "<header>".to_string(),
            reason: format!(
                "Header length ({}) + 8 exceeds total shard file size ({})",
                header_len, total_file_size
            ),
        });
    }
    Ok(header_len)
}

/// Classifies a HuggingFace tensor name into `(layer_idx, TensorRole)`.
pub fn classify_hf_tensor(name: &str, default_num_experts: usize) -> (Option<usize>, TensorRole) {
    if name == "model.embed_tokens.weight" || name == "transformer.wte.weight" {
        return (None, TensorRole::Embedding);
    }
    if name == "lm_head.weight" || name == "output.weight" {
        return (None, TensorRole::LmHead);
    }
    if name == "model.norm.weight" || name == "transformer.ln_f.weight" {
        return (None, TensorRole::Norm);
    }

    let mut layer_idx = None;
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] == "layers" || parts[i] == "h" {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                layer_idx = Some(idx);
                break;
            }
        }
    }

    // Check MoE expert pattern: `...experts.{eid}.w1.weight`
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] == "experts" {
            if let Ok(eid) = parts[i + 1].parse::<usize>() {
                return (
                    layer_idx,
                    TensorRole::MoeExpert {
                        expert_idx: eid,
                        num_experts: default_num_experts.max(eid + 1),
                    },
                );
            }
        }
    }

    if name.contains("block_sparse_moe.gate.") || name.contains("mlp.gate.weight") {
        return (layer_idx, TensorRole::MoeRouter);
    }
    if name.contains("layernorm") || name.ends_with("norm.weight") {
        return (layer_idx, TensorRole::Norm);
    }
    if name.contains("self_attn.q_proj") || name.contains("attn.c_attn_q") {
        return (layer_idx, TensorRole::AttnQ);
    }
    if name.contains("self_attn.k_proj") {
        return (layer_idx, TensorRole::AttnK);
    }
    if name.contains("self_attn.v_proj") {
        return (layer_idx, TensorRole::AttnV);
    }
    if name.contains("self_attn.o_proj") || name.contains("attn.c_proj") {
        return (layer_idx, TensorRole::AttnO);
    }
    if name.contains("mlp.gate_proj") {
        return (layer_idx, TensorRole::FfnGate);
    }
    if name.contains("mlp.up_proj") {
        return (layer_idx, TensorRole::FfnUp);
    }
    if name.contains("mlp.down_proj") {
        return (layer_idx, TensorRole::FfnDown);
    }

    (layer_idx, TensorRole::Other)
}

/// Parses and validates a single `.safetensors` JSON header, enforcing non-overlapping bounds and exact coverage [E02].
pub fn parse_safetensors_header(
    shard: &ShardDescriptor,
    json_bytes: &[u8],
    starting_tensor_id: usize,
    default_num_experts: usize,
) -> Result<Vec<VirtualTensorEntry>> {
    validate_shard_filename(&shard.shard_name)?;
    if json_bytes.len() as u64 != shard.header_len {
        return Err(LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: "<header>".to_string(),
            reason: "Provided header slice length does not match declared header_len".to_string(),
        });
    }

    let root: BTreeMap<String, Value> = serde_json::from_slice(json_bytes)
        .map_err(|e| LlmZipError::MetadataParse(format!("Invalid SafeTensors JSON in {}: {}", shard.shard_name, e)))?;

    let data_base_offset = 8u64.checked_add(shard.header_len).ok_or_else(|| {
        LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: "<header>".to_string(),
            reason: "Overflow computing data base offset".to_string(),
        }
    })?;
    let expected_data_section_len = shard
        .total_file_size
        .checked_sub(data_base_offset)
        .ok_or_else(|| LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: "<header>".to_string(),
            reason: "Shard total size smaller than header".to_string(),
        })?;

    let mut parsed_entries = Vec::new();

    for (key, val) in root {
        if key == "__metadata__" {
            continue;
        }
        let obj = val.as_object().ok_or_else(|| LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: key.clone(),
            reason: "Tensor descriptor is not a JSON object".to_string(),
        })?;

        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Missing dtype field".to_string(),
            })?;
        let dtype = UpstreamDType::from_str_dtype(dtype_str).ok_or_else(|| {
            LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: format!("Unsupported upstream dtype '{}'", dtype_str),
            }
        })?;

        let shape_arr = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Missing shape array".to_string(),
            })?;
        let mut shape = Vec::with_capacity(shape_arr.len());
        for s in shape_arr {
            let dim = s.as_u64().ok_or_else(|| LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Negative or non-integer dimension in shape".to_string(),
            })?;
            let dim_usize = usize::try_from(dim).map_err(|_| LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Dimension exceeds usize".to_string(),
            })?;
            shape.push(dim_usize);
        }

        let offsets_arr = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Missing data_offsets array".to_string(),
            })?;
        if offsets_arr.len() != 2 {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "data_offsets must contain exactly 2 integers".to_string(),
            });
        }
        let start = offsets_arr[0].as_u64().ok_or_else(|| LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: key.clone(),
            reason: "Invalid start offset".to_string(),
        })?;
        let end = offsets_arr[1].as_u64().ok_or_else(|| LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: key.clone(),
            reason: "Invalid end offset".to_string(),
        })?;

        if start >= end || end > expected_data_section_len {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: format!(
                    "Invalid range [{}..{}] for data section of length {}",
                    start, end, expected_data_section_len
                ),
            });
        }

        let (layer_idx, role) = classify_hf_tensor(&key, default_num_experts);
        let stream_abs_start = data_base_offset.checked_add(start).ok_or_else(|| {
            LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Overflow computing stream_abs_start".to_string(),
            }
        })?;
        let stream_abs_end = data_base_offset.checked_add(end).ok_or_else(|| {
            LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Overflow computing stream_abs_end".to_string(),
            }
        })?;

        let entry = VirtualTensorEntry {
            tensor_id: 0,
            hf_name: key.clone(),
            shard_idx: shard.shard_idx,
            data_rel_start: start,
            data_rel_end: end,
            stream_abs_start,
            stream_abs_end,
            dtype,
            shape,
            layer_idx,
            role,
        };

        let elem_count = entry.num_elements()?;
        let expected_bytes = elem_count.checked_mul(dtype.byte_size()).ok_or_else(|| {
            LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key.clone(),
                reason: "Byte size multiplication overflow".to_string(),
            }
        })?;
        if end - start != expected_bytes {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: key,
                reason: format!(
                    "Declared byte range ({}) != shape * dtype byte size ({})",
                    end - start,
                    expected_bytes
                ),
            });
        }

        parsed_entries.push(entry);
    }

    if parsed_entries.is_empty() {
        return Err(LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: "<none>".to_string(),
            reason: "Shard contains zero tensors".to_string(),
        });
    }

    // Sort by physical offset in the shard and verify strict non-overlapping contiguous layout [E02].
    parsed_entries.sort_by_key(|e| e.data_rel_start);
    let mut cursor = 0u64;
    for (idx, entry) in parsed_entries.iter_mut().enumerate() {
        if entry.data_rel_start != cursor {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: shard.shard_name.clone(),
                tensor_name: entry.hf_name.clone(),
                reason: format!(
                    "Tensor offset overlap or gap detected: expected start at {}, got {}",
                    cursor, entry.data_rel_start
                ),
            });
        }
        cursor = entry.data_rel_end;
        entry.tensor_id = starting_tensor_id + idx;
    }

    if cursor != expected_data_section_len {
        return Err(LlmZipError::InvalidTensorBounds {
            shard_name: shard.shard_name.clone(),
            tensor_name: "<eof>".to_string(),
            reason: format!(
                "Trailing unreferenced bytes in shard: tensors end at {}, data section is {}",
                cursor, expected_data_section_len
            ),
        });
    }

    Ok(parsed_entries)
}

/// Builds the complete Virtual Tensor Address Space (VTAS) across all shards and analyzes topological order.
pub fn build_virtual_tensor_space(
    shards_with_headers: Vec<(ShardDescriptor, Vec<u8>)>,
    tie_word_embeddings_cfg: bool,
    default_num_experts: usize,
) -> Result<VirtualTensorSpace> {
    let mut shards = Vec::with_capacity(shards_with_headers.len());
    let mut all_tensors = Vec::new();

    for (shard, header_bytes) in shards_with_headers {
        let mut shard_tensors = parse_safetensors_header(
            &shard,
            &header_bytes,
            all_tensors.len(),
            default_num_experts,
        )?;
        if all_tensors.len() + shard_tensors.len() > MAX_TOTAL_TENSORS {
            return Err(LlmZipError::MetadataParse(format!(
                "Total tensor count exceeds safety limit {}",
                MAX_TOTAL_TENSORS
            )));
        }
        shards.push(shard);
        all_tensors.append(&mut shard_tensors);
    }

    // Analyze physical order vs topological execution order (Question 6 & E07).
    let mut is_strictly_topological = true;
    let mut out_of_order_prefetch_ids = Vec::new();
    let mut has_lm_head = false;
    let mut max_layer_seen: Option<usize> = None;

    for (pos, t) in all_tensors.iter().enumerate() {
        if t.role == TensorRole::LmHead {
            has_lm_head = true;
        }
        if t.role == TensorRole::Embedding && pos > 0 && max_layer_seen.is_some() {
            is_strictly_topological = false;
            out_of_order_prefetch_ids.push(t.tensor_id);
        }
        if let Some(l_idx) = t.layer_idx {
            if let Some(prev_max) = max_layer_seen {
                if l_idx < prev_max {
                    is_strictly_topological = false;
                    out_of_order_prefetch_ids.push(t.tensor_id);
                } else {
                    max_layer_seen = Some(l_idx);
                }
            } else {
                max_layer_seen = Some(l_idx);
            }
        }
    }

    let tie_word_embeddings = tie_word_embeddings_cfg || !has_lm_head;

    Ok(VirtualTensorSpace {
        shards,
        tensors: all_tensors,
        topology: TopologyCheckReport {
            is_strictly_topological,
            out_of_order_prefetch_ids,
            tie_word_embeddings,
        },
    })
}
