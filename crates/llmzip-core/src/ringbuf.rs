use crate::error::{LlmZipError, Result};
use std::collections::BTreeMap;

/// RoPE (Rotary Position Embedding) weight layout mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeWeightLayout {
    /// Standard half-split (NeoX / Qwen2 / Mistral HF native).
    SplitHalf,
    /// Interleaved adjacent pairs (Llama GGUF native).
    Interleaved,
}

/// Bounded in-memory ring buffer holding partial bytes across shard/chunk boundaries [E06]
/// and single-layer weights for QKV fusion or RoPE permutation.
pub struct LayerRingBuffer {
    max_budget_bytes: usize,
    current_bytes_held: usize,
    partial_tensor_id: Option<usize>,
    partial_tensor_expected: usize,
    partial_tensor_buf: Vec<u8>,
    layer_tensors: BTreeMap<String, Vec<f32>>,
}

impl LayerRingBuffer {
    pub fn new(max_budget_bytes: usize) -> Self {
        Self {
            max_budget_bytes,
            current_bytes_held: 0,
            partial_tensor_id: None,
            partial_tensor_expected: 0,
            partial_tensor_buf: Vec::new(),
            layer_tensors: BTreeMap::new(),
        }
    }

    pub fn current_bytes_held(&self) -> usize {
        self.current_bytes_held + self.partial_tensor_buf.len()
    }

    pub fn max_budget_bytes(&self) -> usize {
        self.max_budget_bytes
    }

    /// Feeds a chunk of raw bytes for `tensor_id` (total expected length `expected_len`).
    /// Returns `Some(complete_bytes)` once all bytes for `tensor_id` have been assembled across chunks/shards [E06].
    pub fn push_tensor_chunk(
        &mut self,
        tensor_id: usize,
        expected_len: usize,
        chunk: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        if expected_len > self.max_budget_bytes {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: "<ringbuf>".to_string(),
                tensor_name: format!("tensor_{}", tensor_id),
                reason: format!(
                    "Tensor byte length ({}) exceeds single-layer RAM ring buffer budget ({})",
                    expected_len, self.max_budget_bytes
                ),
            });
        }

        if self.partial_tensor_id != Some(tensor_id) {
            if self.partial_tensor_id.is_some() && !self.partial_tensor_buf.is_empty() {
                return Err(LlmZipError::InvalidTensorBounds {
                    shard_name: "<ringbuf>".to_string(),
                    tensor_name: format!("tensor_{}", tensor_id),
                    reason: "Interrupted partial tensor assembly before completion".to_string(),
                });
            }
            // Fast path: entire tensor arrived in a single chunk
            if chunk.len() == expected_len {
                return Ok(Some(chunk.to_vec()));
            }
            self.partial_tensor_id = Some(tensor_id);
            self.partial_tensor_expected = expected_len;
            self.partial_tensor_buf.clear();
            self.partial_tensor_buf.reserve(expected_len);
        }

        if self.partial_tensor_buf.len() + chunk.len() > self.partial_tensor_expected {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: "<ringbuf>".to_string(),
                tensor_name: format!("tensor_{}", tensor_id),
                reason: "Chunk overflowed expected tensor byte length".to_string(),
            });
        }

        self.partial_tensor_buf.extend_from_slice(chunk);

        if self.partial_tensor_buf.len() == self.partial_tensor_expected {
            self.partial_tensor_id = None;
            self.partial_tensor_expected = 0;
            let completed = std::mem::take(&mut self.partial_tensor_buf);
            Ok(Some(completed))
        } else {
            Ok(None)
        }
    }

    /// Stores a decoded FP32 tensor in the current layer's working set, enforcing the memory budget.
    pub fn store_layer_tensor(&mut self, name: String, data: Vec<f32>) -> Result<()> {
        let byte_cost = data.len() * 4;
        if self.current_bytes_held + byte_cost > self.max_budget_bytes {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: "<ringbuf>".to_string(),
                tensor_name: name,
                reason: format!(
                    "Layer working set ({} + {} bytes) exceeds RAM ring buffer budget ({})",
                    self.current_bytes_held, byte_cost, self.max_budget_bytes
                ),
            });
        }
        self.current_bytes_held += byte_cost;
        self.layer_tensors.insert(name, data);
        Ok(())
    }

    pub fn take_layer_tensor(&mut self, name: &str) -> Option<Vec<f32>> {
        if let Some(v) = self.layer_tensors.remove(name) {
            self.current_bytes_held = self.current_bytes_held.saturating_sub(v.len() * 4);
            Some(v)
        } else {
            None
        }
    }

    pub fn clear_layer(&mut self) {
        self.layer_tensors.clear();
        self.current_bytes_held = 0;
    }
}

/// Permutes `q_proj` or `k_proj` weights of shape `[num_heads * head_dim, hidden_dim]`
/// from HuggingFace split-half layout (`[first_half, second_half]`) to GGUF interleaved layout (`[even, odd]`).
pub fn permute_rope_weights_f32(
    weights: &[f32],
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(weights.len(), num_heads * head_dim * hidden_dim);
    let half = head_dim / 2;
    let mut out = vec![0.0f32; weights.len()];

    for h in 0..num_heads {
        let head_row_base = h * head_dim;
        for i in 0..half {
            let src_row_0 = head_row_base + i;
            let src_row_1 = head_row_base + half + i;
            let dst_row_0 = head_row_base + 2 * i;
            let dst_row_1 = head_row_base + 2 * i + 1;

            out[dst_row_0 * hidden_dim..(dst_row_0 + 1) * hidden_dim]
                .copy_from_slice(&weights[src_row_0 * hidden_dim..(src_row_0 + 1) * hidden_dim]);
            out[dst_row_1 * hidden_dim..(dst_row_1 + 1) * hidden_dim]
                .copy_from_slice(&weights[src_row_1 * hidden_dim..(src_row_1 + 1) * hidden_dim]);
        }
    }
    out
}

/// Fuses separate `q_proj`, `k_proj`, and `v_proj` row-major matrices into a single contiguous `qkv_proj` matrix in RAM.
pub fn fuse_qkv_weights_f32(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut fused = Vec::with_capacity(q.len() + k.len() + v.len());
    fused.extend_from_slice(q);
    fused.extend_from_slice(k);
    fused.extend_from_slice(v);
    fused
}
