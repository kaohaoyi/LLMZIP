use crate::error::Result;
use crate::preflight::{TensorRole, VirtualTensorSpace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default GGUF byte alignment (32 bytes).
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
/// Unverified magic header written before all shard SHA-256 hashes pass [E03].
pub const GGUF_MAGIC_UNVERIFIED: [u8; 4] = *b"UNVF";
/// Official GGUF magic header written atomically upon full verification [E03].
pub const GGUF_MAGIC_VERIFIED: [u8; 4] = *b"GGUF";
/// GGUF v3 format version.
pub const GGUF_VERSION_V3: u32 = 3;

/// Supported quantization types in LLMZIP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantFormat {
    /// GGUF Q4_K (Super-block of 256 elements = 144 bytes).
    GgufQ4K,
    /// GGUF Q6_K (Super-block of 256 elements = 210 bytes, used for attn_v / ffn_down in Q4_K_M).
    GgufQ6K,
    /// GGUF Q8_0 (Block of 32 elements = 34 bytes, used as fallback when dim % 256 != 0).
    GgufQ8_0,
    /// Unquantized FP16 (Block of 1 element = 2 bytes, fallback when dim % 32 != 0).
    GgufF16,
    /// Unquantized FP32 (Block of 1 element = 4 bytes, used for LayerNorm / RMSNorm / Router).
    GgufF32,
    /// AWQ INT4 Group-128 format (128 elements -> 64B qweight + 2B scale + 1B qzero = 67B effective or packed tensors).
    AwqInt4G128,
    /// NVIDIA Blackwell Two-Level NVFP4 Super-Block (256 elements -> 4B FP32 global + 16B FP8 E4M3 micro + 128B E2M1 = 148 B).
    Nvfp4Sb256,
    /// NVIDIA NVFP4 16-element Micro-Block (16 elements -> 1B FP8 E4M3 scale + 8B E2M1 = 9 B).
    Nvfp4B16,
    /// OCP MXFP4 32-element Block (32 elements -> 1B E8M0 shared exp + 16B E2M1 = 17 B).
    Mxfp4B32,
    /// FP8 E4M3FN 32-element Block (32 elements -> 2B FP16 scale + 32B E4M3 = 34 B).
    Fp8E4M3B32,
}

impl QuantFormat {
    /// Number of elements per quantization block ($B_Q$).
    pub const fn block_elements(self) -> usize {
        match self {
            Self::GgufQ4K | Self::GgufQ6K | Self::Nvfp4Sb256 => 256,
            Self::GgufQ8_0 | Self::Mxfp4B32 | Self::Fp8E4M3B32 => 32,
            Self::GgufF16 | Self::GgufF32 => 1,
            Self::AwqInt4G128 => 128,
            Self::Nvfp4B16 => 16,
        }
    }

    /// Number of bytes per quantization block ($S_Q$).
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::GgufQ4K => 144,
            Self::GgufQ6K => 210,
            Self::GgufQ8_0 => 34,
            Self::GgufF16 => 2,
            Self::GgufF32 => 4,
            Self::AwqInt4G128 => 68,
            Self::Nvfp4Sb256 => 148,
            Self::Nvfp4B16 => 9,
            Self::Mxfp4B32 => 17,
            Self::Fp8E4M3B32 => 34,
        }
    }

    /// Official GGUF tensor type enum ID.
    pub const fn gguf_type_id(self) -> u32 {
        match self {
            Self::GgufF32 => 0,
            Self::GgufF16 => 1,
            Self::GgufQ8_0 => 8,
            Self::GgufQ4K => 12,
            Self::GgufQ6K => 14,
            Self::AwqInt4G128 => 100, // Custom/AWQ container marker
            Self::Nvfp4Sb256 => 101,  // NVFP4 Super-block marker
            Self::Nvfp4B16 => 102,    // NVFP4 Micro-block marker
            Self::Mxfp4B32 => 103,    // OCP MXFP4 marker
            Self::Fp8E4M3B32 => 104,  // FP8 E4M3 marker
        }
    }
}

/// User-facing quantization preset profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantPreset {
    /// GGUF Q4_K_M: Q4_K for standard matrices, Q6_K for `attn_v`, `ffn_down`, `lm_head`, F32 for norms.
    GgufQ4KM,
    /// GGUF Q8_0: Q8_0 for all weight matrices, F32 for norms.
    GgufQ8_0,
    /// AWQ INT4 (group_size = 128).
    AwqW4G128,
    /// NVIDIA Blackwell NVFP4 (Super-block of 256 elements; fallback to 16-element micro-block).
    Nvfp4,
    /// OCP MXFP4 (32-element micro-block with E8M0 shared exponent).
    Mxfp4,
    /// FP8 E4M3FN (32-element block with FP16 scale).
    Fp8E4M3,
}

/// Describes where an upstream tensor writes its quantized payload in the pre-allocated output file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedWriteTarget {
    pub upstream_tensor_id: usize,
    pub output_tensor_name: String,
    pub assigned_quant: QuantFormat,
    /// Whether this tensor was downgraded due to non-aligned inner dimension [E05].
    pub fallback_applied: bool,
    /// Exact byte offset in the final file where this tensor (or MoE expert slice) must be written via `pwrite`.
    pub file_write_offset: u64,
    /// Exact byte length of the quantized payload for this upstream tensor.
    pub quantized_byte_len: u64,
    /// Optional secondary write offset when `tie_word_embeddings` duplicates `token_embd.weight` to `output.weight` [E07].
    pub tied_secondary_offset: Option<u64>,
}

/// Metadata entry for a GGUF top-level tensor (including stacked 3D MoE tensors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensorDirectoryEntry {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub quant_format: QuantFormat,
    /// Offset relative to the start of the GGUF `tensor_data` section.
    pub data_section_offset: u64,
    pub total_byte_len: u64,
}

/// Complete Deterministic Layout Plan computed before downloading any weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicLayoutPlan {
    pub preset: QuantPreset,
    pub alignment: u64,
    /// Pre-serialized header bytes (starting with `"UNVF"` unverified magic [E03]).
    pub unverified_header_bytes: Vec<u8>,
    /// Byte offset where the tensor data section begins (aligned to `alignment`).
    pub data_section_base_offset: u64,
    /// Exact total size of the final output file in bytes (for `fallocate` / `SetEndOfFile` [E08]).
    pub total_output_file_size: u64,
    /// Indexed by `upstream_tensor_id`.
    pub write_targets: Vec<PlannedWriteTarget>,
    /// Consolidated GGUF tensor directory (with MoE experts merged into 3D tensors).
    pub gguf_directory: Vec<GgufTensorDirectoryEntry>,
}

#[inline]
pub const fn align_up(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

/// Assigns the target `QuantFormat` for a tensor, applying automatic fallback `256 -> 32 -> 1` if `inner_dim` is not divisible [E05].
pub fn select_quant_format_with_fallback(
    preset: QuantPreset,
    role: TensorRole,
    shape: &[usize],
) -> (QuantFormat, bool) {
    // 1D tensors and Normalization / Router weights are always kept in F32.
    if shape.len() == 1 || matches!(role, TensorRole::Norm | TensorRole::MoeRouter) {
        return (QuantFormat::GgufF32, false);
    }

    let inner_dim = *shape.last().unwrap_or(&1);

    let desired = match preset {
        QuantPreset::GgufQ4KM => match role {
            TensorRole::AttnV | TensorRole::FfnDown | TensorRole::LmHead => QuantFormat::GgufQ6K,
            _ => QuantFormat::GgufQ4K,
        },
        QuantPreset::GgufQ8_0 => QuantFormat::GgufQ8_0,
        QuantPreset::AwqW4G128 => match role {
            TensorRole::Embedding | TensorRole::LmHead => QuantFormat::GgufF16,
            _ => QuantFormat::AwqInt4G128,
        },
        QuantPreset::Nvfp4 => match role {
            TensorRole::Embedding | TensorRole::LmHead => QuantFormat::GgufF16,
            _ => QuantFormat::Nvfp4Sb256,
        },
        QuantPreset::Mxfp4 => match role {
            TensorRole::Embedding | TensorRole::LmHead => QuantFormat::GgufF16,
            _ => QuantFormat::Mxfp4B32,
        },
        QuantPreset::Fp8E4M3 => match role {
            TensorRole::Embedding => QuantFormat::GgufF16,
            _ => QuantFormat::Fp8E4M3B32,
        },
    };

    // Check alignment against block_elements [E05].
    if inner_dim.is_multiple_of(desired.block_elements()) {
        return (desired, false);
    }

    // Automatic safe fallback hierarchy:
    // NVFP4 Sb256 -> Nvfp4B16 (16) -> GgufF16 (1)
    if preset == QuantPreset::Nvfp4 {
        if inner_dim.is_multiple_of(QuantFormat::Nvfp4B16.block_elements()) {
            return (QuantFormat::Nvfp4B16, true);
        } else {
            return (QuantFormat::GgufF16, true);
        }
    }

    if inner_dim.is_multiple_of(QuantFormat::GgufQ8_0.block_elements()) && preset != QuantPreset::AwqW4G128 {
        (QuantFormat::GgufQ8_0, true)
    } else {
        (QuantFormat::GgufF16, true)
    }
}

/// Computes the exact quantized byte size for a tensor of shape `shape` under `quant`.
pub fn compute_quantized_bytes(shape: &[usize], quant: QuantFormat) -> u64 {
    let inner_dim = *shape.last().unwrap_or(&1);
    let outer_elements: u64 = shape[..shape.len().saturating_sub(1)]
        .iter()
        .map(|&d| d as u64)
        .product();
    let blocks_per_row = (inner_dim / quant.block_elements()) as u64;
    outer_elements * blocks_per_row * quant.block_bytes()
}

/// Maps a HuggingFace tensor name to its standard GGUF name (merging MoE experts into 3D tensor names).
pub fn map_hf_name_to_gguf(hf_name: &str) -> String {
    if hf_name == "model.embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }
    if hf_name == "model.norm.weight" {
        return "output_norm.weight".to_string();
    }
    if hf_name == "lm_head.weight" {
        return "output.weight".to_string();
    }
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        if let Some((bid, tail)) = rest.split_once('.') {
            if let Some(exp_rest) = tail.strip_prefix("block_sparse_moe.experts.") {
                if let Some((_eid, w_tail)) = exp_rest.split_once('.') {
                    let mapped_w = match w_tail {
                        "w1.weight" => "ffn_gate_exps.weight",
                        "w2.weight" => "ffn_down_exps.weight",
                        "w3.weight" => "ffn_up_exps.weight",
                        other => other,
                    };
                    return format!("blk.{}.{}", bid, mapped_w);
                }
            }
            let mapped_tail = match tail {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "block_sparse_moe.gate.weight" => "ffn_gate_inp.weight",
                other => other,
            };
            return format!("blk.{}.{}", bid, mapped_tail);
        }
    }
    hf_name.to_string()
}

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Builds the complete deterministic layout plan (exact byte offsets for every tensor, MoE sub-offsets, and GGUF header).
pub fn build_deterministic_layout(
    vtas: &VirtualTensorSpace,
    preset: QuantPreset,
    arch_name: &str,
) -> Result<DeterministicLayoutPlan> {
    let alignment = GGUF_DEFAULT_ALIGNMENT;
    let mut gguf_dir_map: BTreeMap<String, (GgufTensorDirectoryEntry, Vec<(usize, usize)>)> = BTreeMap::new();
    let mut single_slice_sizes = Vec::with_capacity(vtas.tensors.len());
    let mut assigned_quants = Vec::with_capacity(vtas.tensors.len());

    for t in &vtas.tensors {
        let (quant, fallback) = select_quant_format_with_fallback(preset, t.role, &t.shape);
        let slice_bytes = compute_quantized_bytes(&t.shape, quant);
        single_slice_sizes.push(slice_bytes);
        assigned_quants.push((quant, fallback));

        let gguf_name = map_hf_name_to_gguf(&t.hf_name);

        match t.role {
            TensorRole::MoeExpert {
                expert_idx,
                num_experts,
            } => {
                let entry = gguf_dir_map.entry(gguf_name.clone()).or_insert_with(|| {
                    // GGUF dimensions are stored in reverse (inner-most first: [d_in, d_out, num_experts]).
                    let mut dims: Vec<u64> = t.shape.iter().rev().map(|&d| d as u64).collect();
                    dims.push(num_experts as u64);
                    (
                        GgufTensorDirectoryEntry {
                            name: gguf_name,
                            dimensions: dims,
                            quant_format: quant,
                            data_section_offset: 0,
                            total_byte_len: slice_bytes * (num_experts as u64),
                        },
                        Vec::new(),
                    )
                });
                entry.1.push((t.tensor_id, expert_idx));
                if expert_idx + 1 > entry.0.dimensions.last().copied().unwrap_or(1) as usize {
                    let actual_e = (expert_idx + 1) as u64;
                    if let Some(last) = entry.0.dimensions.last_mut() {
                        *last = actual_e;
                    }
                    entry.0.total_byte_len = slice_bytes * actual_e;
                }
            }
            _ => {
                let dims: Vec<u64> = t.shape.iter().rev().map(|&d| d as u64).collect();
                gguf_dir_map.insert(
                    gguf_name.clone(),
                    (
                        GgufTensorDirectoryEntry {
                            name: gguf_name,
                            dimensions: dims,
                            quant_format: quant,
                            data_section_offset: 0,
                            total_byte_len: slice_bytes,
                        },
                        vec![(t.tensor_id, 0)],
                    ),
                );
            }
        }
    }

    // If Weight Tying is active (`tie_word_embeddings`) and `output.weight` is not present, allocate an explicit slot for `output.weight` [E07].
    let mut tied_embd_tensor_id: Option<usize> = None;
    if vtas.topology.tie_word_embeddings && !gguf_dir_map.contains_key("output.weight") {
        if let Some(embd_t) = vtas.tensors.iter().find(|t| t.role == TensorRole::Embedding) {
            tied_embd_tensor_id = Some(embd_t.tensor_id);
            let (quant, _) = assigned_quants[embd_t.tensor_id];
            let slice_bytes = single_slice_sizes[embd_t.tensor_id];
            let dims: Vec<u64> = embd_t.shape.iter().rev().map(|&d| d as u64).collect();
            gguf_dir_map.insert(
                "output.weight".to_string(),
                (
                    GgufTensorDirectoryEntry {
                        name: "output.weight".to_string(),
                        dimensions: dims,
                        quant_format: quant,
                        data_section_offset: 0,
                        total_byte_len: slice_bytes,
                    },
                    Vec::new(),
                ),
            );
        }
    }

    // Assign relative offsets within the GGUF data section (aligned to 32 bytes).
    let mut data_rel_cursor = 0u64;
    let mut gguf_directory = Vec::with_capacity(gguf_dir_map.len());
    for (dir_entry, _contributors) in gguf_dir_map.values_mut() {
        data_rel_cursor = align_up(data_rel_cursor, alignment);
        dir_entry.data_section_offset = data_rel_cursor;
        data_rel_cursor += dir_entry.total_byte_len;
        gguf_directory.push(dir_entry.clone());
    }

    // Serialize the GGUF v3 binary header with "UNVF" magic lock [E03].
    let mut header_buf = Vec::with_capacity(4096);
    header_buf.extend_from_slice(&GGUF_MAGIC_UNVERIFIED);
    header_buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    header_buf.extend_from_slice(&(gguf_directory.len() as u64).to_le_bytes());
    // 1 KV metadata pair: `general.architecture` = `arch_name`
    header_buf.extend_from_slice(&1u64.to_le_bytes());
    write_gguf_string(&mut header_buf, "general.architecture");
    header_buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_METADATA_VALUE_TYPE_STRING = 8
    write_gguf_string(&mut header_buf, arch_name);

    // Serialize GGUF tensor info directory
    for entry in &gguf_directory {
        write_gguf_string(&mut header_buf, &entry.name);
        header_buf.extend_from_slice(&(entry.dimensions.len() as u32).to_le_bytes());
        for &d in &entry.dimensions {
            header_buf.extend_from_slice(&d.to_le_bytes());
        }
        header_buf.extend_from_slice(&entry.quant_format.gguf_type_id().to_le_bytes());
        header_buf.extend_from_slice(&entry.data_section_offset.to_le_bytes());
    }

    let data_section_base_offset = align_up(header_buf.len() as u64, alignment);
    header_buf.resize(data_section_base_offset as usize, 0u8);

    let total_output_file_size = data_section_base_offset + data_rel_cursor;

    // Build exact per-upstream-tensor write targets (including MoE strided sub-offsets: base + expert_idx * stride).
    let mut write_targets: Vec<Option<PlannedWriteTarget>> = vec![None; vtas.tensors.len()];
    let tied_output_abs_offset = gguf_dir_map
        .get("output.weight")
        .map(|(dir, _)| data_section_base_offset + dir.data_section_offset);

    for (name, (dir_entry, contributors)) in &gguf_dir_map {
        let base_abs_offset = data_section_base_offset + dir_entry.data_section_offset;
        for &(tensor_id, expert_idx) in contributors {
            let (quant, fallback) = assigned_quants[tensor_id];
            let slice_bytes = single_slice_sizes[tensor_id];
            let sub_offset = base_abs_offset + (expert_idx as u64) * slice_bytes;
            let tied_secondary_offset = if Some(tensor_id) == tied_embd_tensor_id {
                tied_output_abs_offset
            } else {
                None
            };
            write_targets[tensor_id] = Some(PlannedWriteTarget {
                upstream_tensor_id: tensor_id,
                output_tensor_name: name.clone(),
                assigned_quant: quant,
                fallback_applied: fallback,
                file_write_offset: sub_offset,
                quantized_byte_len: slice_bytes,
                tied_secondary_offset,
            });
        }
    }

    let resolved_targets: Vec<PlannedWriteTarget> = write_targets
        .into_iter()
        .enumerate()
        .map(|(idx, opt): (usize, Option<PlannedWriteTarget>)| {
            opt.ok_or_else(|| {
                crate::error::LlmZipError::MetadataParse(format!(
                    "Internal invariant: missing write target for tensor {}",
                    idx
                ))
            })
        })
        .collect::<crate::error::Result<Vec<_>>>()?;

    Ok(DeterministicLayoutPlan {
        preset,
        alignment,
        unverified_header_bytes: header_buf,
        data_section_base_offset,
        total_output_file_size,
        write_targets: resolved_targets,
        gguf_directory,
    })
}
