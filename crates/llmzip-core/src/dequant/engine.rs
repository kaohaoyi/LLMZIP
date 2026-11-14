use crate::error::{LlmZipError, Result};
use crate::layout::QuantFormat;
use crate::quant::gguf_simd::dequantize_to_f32;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DequantDtype {
    BF16,
    FP16,
    F32,
}

impl DequantDtype {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "fp16" | "f16" | "float16" => Ok(Self::FP16),
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            other => Err(LlmZipError::DequantError {
                tensor_name: String::new(),
                reason: format!("Unsupported dequant target dtype '{}'", other),
            }),
        }
    }

    pub fn safetensors_dtype_str(&self) -> &'static str {
        match self {
            Self::BF16 => "BF16",
            Self::FP16 => "F16",
            Self::F32 => "F32",
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::BF16 | Self::FP16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Converts IEEE 754 f32 slice to BF16 bytes (round-to-nearest-even).
pub fn f32_to_bf16_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &val in src {
        let u = val.to_bits();
        // Round to nearest even: add lsb of bf16 + 0x7fff
        let lsb = (u >> 16) & 1;
        let rounding_bias = 0x7FFF + lsb;
        let rounded = u.wrapping_add(rounding_bias);
        let bf16 = (rounded >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

/// Converts IEEE 754 f32 slice to IEEE 754 half-precision FP16 bytes.
pub fn f32_to_fp16_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &val in src {
        let u = val.to_bits();
        let sign = (u >> 31) & 0x1;
        let exp = ((u >> 23) & 0xFF) as i32;
        let mant = u & 0x7FFFFF;

        let h: u16 = if exp == 255 {
            // NaN or Infinity
            let nan_bit = if mant != 0 { 0x0200 } else { 0 };
            ((sign << 15) | 0x7C00 | nan_bit) as u16
        } else {
            let new_exp = exp - 127 + 15;
            if new_exp >= 31 {
                // Overflow to Infinity
                ((sign << 15) | 0x7C00) as u16
            } else if new_exp <= 0 {
                // Subnormal or Zero
                if 14 - new_exp > 24 {
                    (sign << 15) as u16
                } else {
                    let full_mant = mant | 0x800000;
                    let shifted = full_mant >> (1 - new_exp + 13);
                    ((sign << 15) | (shifted & 0x3FF)) as u16
                }
            } else {
                let shifted_mant = mant >> 13;
                ((sign << 15) | ((new_exp as u32) << 10) | (shifted_mant & 0x3FF)) as u16
            }
        };
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Converts IEEE 754 f32 slice to raw F32 bytes.
pub fn f32_to_f32_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for &val in src {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Dequantizes raw quantized payload bytes into target dtype bytes.
pub fn dequantize_to_dtype(
    quantized_bytes: &[u8],
    format: QuantFormat,
    target_dtype: DequantDtype,
) -> Vec<u8> {
    let f32_values = dequantize_to_f32(quantized_bytes, format);
    match target_dtype {
        DequantDtype::BF16 => f32_to_bf16_bytes(&f32_values),
        DequantDtype::FP16 => f32_to_fp16_bytes(&f32_values),
        DequantDtype::F32 => f32_to_f32_bytes(&f32_values),
    }
}

/// Maps GGUF tensor names back to standard HuggingFace Transformers tensor names.
pub fn gguf_to_hf_name(gguf_name: &str) -> String {
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some((layer_str, suffix)) = rest.split_once('.') {
            if let Ok(layer_idx) = layer_str.parse::<usize>() {
                let hf_suffix = match suffix {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    "attn_norm.weight" => "input_layernorm.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    other => other,
                };
                return format!("model.layers.{}.{}", layer_idx, hf_suffix);
            }
        }
    }

    gguf_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion_fidelity() {
        let floats = vec![0.0f32, 1.0f32, -1.0f32, 0.5f32, std::f32::consts::E];
        let bf16_bytes = f32_to_bf16_bytes(&floats);
        assert_eq!(bf16_bytes.len(), floats.len() * 2);

        let fp16_bytes = f32_to_fp16_bytes(&floats);
        assert_eq!(fp16_bytes.len(), floats.len() * 2);

        let f32_bytes = f32_to_f32_bytes(&floats);
        assert_eq!(f32_bytes.len(), floats.len() * 4);
    }

    #[test]
    fn test_gguf_to_hf_name_mapping() {
        assert_eq!(gguf_to_hf_name("token_embd.weight"), "model.embed_tokens.weight");
        assert_eq!(gguf_to_hf_name("blk.0.attn_q.weight"), "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(gguf_to_hf_name("blk.15.ffn_down.weight"), "model.layers.15.mlp.down_proj.weight");
        assert_eq!(gguf_to_hf_name("output_norm.weight"), "model.norm.weight");
        assert_eq!(gguf_to_hf_name("output.weight"), "lm_head.weight");
        assert_eq!(gguf_to_hf_name("custom_tensor"), "custom_tensor");
    }
}
