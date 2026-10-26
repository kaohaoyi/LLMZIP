use crate::quant::gguf_simd::{f16_to_f32, f32_to_f16_clamped};
use rayon::prelude::*;

/// Precomputed 8 non-negative E2M1 discrete values:
/// 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
pub const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Midpoint rounding thresholds for Round-to-Nearest-Even (RN):
/// [0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00]
pub const E2M1_MIDPOINTS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

/// Maximum finite value representable in E2M1 (6.0).
pub const E2M1_MAX_FINITE: f32 = 6.0;

/// Maximum finite value representable in FP8 E4M3FN (448.0).
pub const FP8_E4M3FN_MAX_FINITE: f32 = 448.0;

/// Maps a finite positive `f32` value to the closest 3-bit unsigned E2M1 code (0..7).
#[inline]
pub fn f32_to_e2m1_magnitude(val: f32) -> u8 {
    if val <= E2M1_MIDPOINTS[0] {
        0
    } else if val <= E2M1_MIDPOINTS[1] {
        1
    } else if val <= E2M1_MIDPOINTS[2] {
        2
    } else if val <= E2M1_MIDPOINTS[3] {
        3
    } else if val <= E2M1_MIDPOINTS[4] {
        4
    } else if val <= E2M1_MIDPOINTS[5] {
        5
    } else if val <= E2M1_MIDPOINTS[6] {
        6
    } else {
        7
    }
}

/// Encodes an `f32` into 4-bit E2M1 (bit 3: sign, bits 0..2: magnitude).
#[inline]
pub fn f32_to_e2m1(val: f32) -> u8 {
    if val.is_nan() {
        return 0;
    }
    let sign_bit = if val.is_sign_negative() && val != 0.0 { 0x08 } else { 0x00 };
    let mag = f32_to_e2m1_magnitude(val.abs());
    sign_bit | mag
}

/// Decodes 4-bit E2M1 (bit 3: sign, bits 0..2: magnitude) to `f32`.
#[inline]
pub fn e2m1_to_f32(code: u8) -> f32 {
    let mag = E2M1_VALUES[(code & 0x07) as usize];
    if (code & 0x08) != 0 {
        -mag
    } else {
        mag
    }
}

/// Encodes `f32` into FP8 E4M3FN (1 sign, 4 exp bias=7, 3 mantissa; 0x7F / 0xFF = NaN, max finite = 448.0).
#[inline]
pub fn f32_to_fp8_e4m3fn(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F; // Canonical NaN in E4M3FN
    }
    let sign_bit = if val.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let abs_val = val.abs();

    if abs_val >= 448.0 {
        return sign_bit | 0x7E; // Clamp to max finite (0b01111110 = 448.0)
    }

    // Smallest normal in E4M3FN is 2^(1 - 7) * 1.0 = 2^(-6) = 0.015625
    // Smallest subnormal step is 2^(-6) * (1/8) = 2^(-9) = 0.001953125
    if abs_val < 0.015625 {
        let sub_steps = (abs_val * 512.0).round() as u8;
        return sign_bit | (sub_steps.min(7));
    }

    let bits = abs_val.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x007F_FFFF;

    let target_exp = f32_exp + 7;

    if target_exp >= 15 {
        // Exponent 15: mantissa up to 6 is valid (7 is NaN; 0x78 | 6 = 0x7E = 448.0)
        let mant_bits = f32_mant >> 20;
        let rem = (f32_mant >> 19) & 1;
        let rounded = (mant_bits + rem).min(6) as u8;
        sign_bit | 0x78 | rounded
    } else {
        // Normal number in E4M3FN
        let mant_bits = f32_mant >> 20; // 3 bits of mantissa
        let rem = (f32_mant >> 19) & 1; // Rounding bit
        let mut rounded = mant_bits + rem;
        let mut exp = target_exp as u8;
        if rounded >= 8 {
            exp += 1;
            rounded = 0;
            if exp >= 15 {
                return sign_bit | 0x7E;
            }
        }
        sign_bit | (exp << 3) | (rounded as u8)
    }
}

/// Decodes FP8 E4M3FN to `f32`.
#[inline]
pub fn fp8_e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = (byte & 0x80) != 0;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    if exp == 15 && mant == 7 {
        return f32::NAN;
    }

    let val = if exp == 0 {
        // Subnormal: 2^(-6) * (mant / 8)
        (mant as f32) * (1.0 / 512.0)
    } else {
        // Normal: 2^(exp - 7) * (1 + mant / 8)
        let scale = 2.0f32.powi(exp as i32 - 7);
        scale * (1.0 + (mant as f32) / 8.0)
    };

    if sign {
        -val
    } else {
        val
    }
}

/// Encodes a maximum absolute value into OCP MXFP4 E8M0 shared power-of-two exponent (bias = 127).
#[inline]
pub fn f32_to_e8m0_shared_exp(amax: f32) -> (u8, f32) {
    if amax <= 1e-12 {
        return (0, 0.0);
    }
    // In OCP MX v1.0 standard, the shared exponent is:
    // E = clamp(floor(log2(max |x_i|)) - emax_elem + bias, 0, 255)
    // For E2M1, emax = 2, bias = 127.
    let f32_exp = ((amax.to_bits() >> 23) & 0xFF) as i32 - 127;
    let raw_exp = f32_exp - 2 + 127;
    let exp_clamped = raw_exp.clamp(0, 254) as u8;
    let scale = 2.0f32.powi(exp_clamped as i32 - 127);
    (exp_clamped, scale)
}

/// Decodes OCP E8M0 shared exponent byte to `f32` scale.
#[inline]
pub fn e8m0_to_f32(exp_byte: u8) -> f32 {
    if exp_byte == 0 {
        0.0
    } else if exp_byte == 255 {
        f32::NAN
    } else {
        2.0f32.powi(exp_byte as i32 - 127)
    }
}

// ============================================================================
// Block Quantizers & Dequantizers
// ============================================================================

/// Quantizes `src` into NVIDIA Blackwell Two-Level NVFP4 Super-Blocks (256 elements -> 148 Bytes).
/// Layout of 148-byte `Nvfp4Sb256`:
/// - `global_scale`: `f32` (4 Bytes, little-endian)
/// - `micro_scales`: `[u8; 16]` (16 Bytes, FP8 E4M3FN for each 16-element sub-block)
/// - `nibbles`: `[u8; 128]` (128 Bytes, packed 4-bit E2M1)
pub fn quantize_nvfp4_sb256(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 256, 0);
    let num_blocks = src.len() / 256;
    let mut out = vec![0u8; num_blocks * 148];

    let quant_one_sb = |sb: &[f32], out_slot: &mut [u8]| {
        let mut sub_amaxes = [0.0f32; 16];
        let mut max_sub_amax = 0.0f32;
        for k in 0..16 {
            let sub = &sb[k * 16..(k + 1) * 16];
            let mut s_max = 0.0f32;
            for &x in sub {
                let ax = x.abs();
                if ax > s_max {
                    s_max = ax;
                }
            }
            sub_amaxes[k] = s_max;
            if s_max > max_sub_amax {
                max_sub_amax = s_max;
            }
        }

        let global_scale = if max_sub_amax > 1e-12 {
            max_sub_amax / (E2M1_MAX_FINITE * 16.0)
        } else {
            1.0
        };
        let inv_global = 1.0 / global_scale;

        let mut micro_scales = [0u8; 16];
        let mut eff_micro = [0.0f32; 16];

        for k in 0..16 {
            let sub_amax = sub_amaxes[k];
            let desired_s = (sub_amax * inv_global) / E2M1_MAX_FINITE;
            let fp8_code = f32_to_fp8_e4m3fn(desired_s);
            micro_scales[k] = fp8_code;
            eff_micro[k] = fp8_e4m3fn_to_f32(fp8_code);
        }

        let mut nibbles = [0u8; 128];
        for k in 0..16 {
            let sub = &sb[k * 16..(k + 1) * 16];
            let block_scale = global_scale * eff_micro[k];
            let inv_block_scale = if block_scale > 1e-12 { 1.0 / block_scale } else { 0.0 };

            for j in 0..8 {
                let v0 = sub[2 * j];
                let v1 = sub[2 * j + 1];
                let q0 = f32_to_e2m1(v0 * inv_block_scale) & 0x0F;
                let q1 = f32_to_e2m1(v1 * inv_block_scale) & 0x0F;
                nibbles[k * 8 + j] = q0 | (q1 << 4);
            }
        }

        out_slot[0..4].copy_from_slice(&global_scale.to_le_bytes());
        out_slot[4..20].copy_from_slice(&micro_scales);
        out_slot[20..148].copy_from_slice(&nibbles);
    };

    if num_blocks >= 16 {
        out.par_chunks_mut(148 * 16)
            .enumerate()
            .for_each(|(batch_idx, batch_out)| {
                let start_b = batch_idx * 16;
                let blocks_in_batch = batch_out.len() / 148;
                for b in 0..blocks_in_batch {
                    let global_b = start_b + b;
                    let sb = &src[global_b * 256..(global_b + 1) * 256];
                    let slot = &mut batch_out[b * 148..(b + 1) * 148];
                    quant_one_sb(sb, slot);
                }
            });
    } else {
        for b in 0..num_blocks {
            let sb = &src[b * 256..(b + 1) * 256];
            let slot = &mut out[b * 148..(b + 1) * 148];
            quant_one_sb(sb, slot);
        }
    }

    out
}

/// Dequantizes NVIDIA NVFP4 Super-Blocks (148 Bytes -> 256 `f32` elements).
pub fn dequantize_nvfp4_sb256(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 148, 0);
    let num_blocks = bytes.len() / 148;
    let mut out = Vec::with_capacity(num_blocks * 256);

    for b in 0..num_blocks {
        let blk = &bytes[b * 148..(b + 1) * 148];
        let global_scale = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        let micro_scales = &blk[4..20];
        let nibbles = &blk[20..148];

        for k in 0..16 {
            let s_local = fp8_e4m3fn_to_f32(micro_scales[k]);
            let block_scale = global_scale * s_local;

            for j in 0..8 {
                let packed = nibbles[k * 8 + j];
                let q0 = packed & 0x0F;
                let q1 = packed >> 4;
                out.push(e2m1_to_f32(q0) * block_scale);
                out.push(e2m1_to_f32(q1) * block_scale);
            }
        }
    }

    out
}

/// Quantizes `src` into NVIDIA NVFP4 16-element Micro-Blocks (16 elements -> 9 Bytes).
/// Layout: `1B FP8 E4M3 scale + 8B packed E2M1`.
pub fn quantize_nvfp4_b16(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 16, 0);
    let num_blocks = src.len() / 16;
    let mut out = Vec::with_capacity(num_blocks * 9);

    for b in 0..num_blocks {
        let sub = &src[b * 16..(b + 1) * 16];
        let mut amax = 0.0f32;
        for &v in sub {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let scale_val = amax / E2M1_MAX_FINITE;
        let fp8_scale = f32_to_fp8_e4m3fn(scale_val);
        let eff_scale = fp8_e4m3fn_to_f32(fp8_scale).max(1e-12);
        let inv_scale = 1.0 / eff_scale;

        out.push(fp8_scale);
        for j in 0..8 {
            let q0 = f32_to_e2m1(sub[2 * j] * inv_scale) & 0x0F;
            let q1 = f32_to_e2m1(sub[2 * j + 1] * inv_scale) & 0x0F;
            out.push(q0 | (q1 << 4));
        }
    }

    out
}

/// Dequantizes NVIDIA NVFP4 16-element Micro-Blocks (9 Bytes -> 16 `f32` elements).
pub fn dequantize_nvfp4_b16(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 9, 0);
    let num_blocks = bytes.len() / 9;
    let mut out = Vec::with_capacity(num_blocks * 16);

    for b in 0..num_blocks {
        let blk = &bytes[b * 9..(b + 1) * 9];
        let scale = fp8_e4m3fn_to_f32(blk[0]);
        for j in 0..8 {
            let packed = blk[1 + j];
            let q0 = packed & 0x0F;
            let q1 = packed >> 4;
            out.push(e2m1_to_f32(q0) * scale);
            out.push(e2m1_to_f32(q1) * scale);
        }
    }

    out
}

/// Quantizes `src` into OCP MXFP4 32-element Blocks (32 elements -> 17 Bytes).
/// Layout: `1B E8M0 shared exponent + 16B packed E2M1`.
pub fn quantize_mxfp4_b32(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 32, 0);
    let num_blocks = src.len() / 32;
    let mut out = Vec::with_capacity(num_blocks * 17);

    for b in 0..num_blocks {
        let sub = &src[b * 32..(b + 1) * 32];
        let mut amax = 0.0f32;
        for &v in sub {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let (e8m0_code, scale) = f32_to_e8m0_shared_exp(amax);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        out.push(e8m0_code);
        for j in 0..16 {
            let q0 = f32_to_e2m1(sub[2 * j] * inv_scale) & 0x0F;
            let q1 = f32_to_e2m1(sub[2 * j + 1] * inv_scale) & 0x0F;
            out.push(q0 | (q1 << 4));
        }
    }

    out
}

/// Dequantizes OCP MXFP4 32-element Blocks (17 Bytes -> 32 `f32` elements).
pub fn dequantize_mxfp4_b32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 17, 0);
    let num_blocks = bytes.len() / 17;
    let mut out = Vec::with_capacity(num_blocks * 32);

    for b in 0..num_blocks {
        let blk = &bytes[b * 17..(b + 1) * 17];
        let scale = e8m0_to_f32(blk[0]);
        for j in 0..16 {
            let packed = blk[1 + j];
            let q0 = packed & 0x0F;
            let q1 = packed >> 4;
            out.push(e2m1_to_f32(q0) * scale);
            out.push(e2m1_to_f32(q1) * scale);
        }
    }

    out
}

/// Quantizes `src` into FP8 E4M3FN 32-element Blocks (32 elements -> 34 Bytes).
/// Layout: `2B FP16 scale + 32B FP8 E4M3FN weights`.
pub fn quantize_fp8_e4m3_b32(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 32, 0);
    let num_blocks = src.len() / 32;
    let mut out = Vec::with_capacity(num_blocks * 34);

    for b in 0..num_blocks {
        let sub = &src[b * 32..(b + 1) * 32];
        let mut amax = 0.0f32;
        for &v in sub {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let scale = amax / FP8_E4M3FN_MAX_FINITE;
        let scale_f16 = f32_to_f16_clamped(scale);
        let eff_scale = f16_to_f32(scale_f16).max(1e-12);
        let inv_scale = 1.0 / eff_scale;

        out.extend_from_slice(&scale_f16.to_le_bytes());
        for &x in sub {
            out.push(f32_to_fp8_e4m3fn(x * inv_scale));
        }
    }

    out
}

/// Dequantizes FP8 E4M3FN 32-element Blocks (34 Bytes -> 32 `f32` elements).
pub fn dequantize_fp8_e4m3_b32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 34, 0);
    let num_blocks = bytes.len() / 34;
    let mut out = Vec::with_capacity(num_blocks * 32);

    for b in 0..num_blocks {
        let blk = &bytes[b * 34..(b + 1) * 34];
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..32 {
            let q = blk[2 + i];
            out.push(fp8_e4m3fn_to_f32(q) * scale);
        }
    }

    out
}

/// Split payload for vLLM / SGLang / TensorRT-LLM `modelopt` native `.safetensors` export.
#[derive(Debug, Clone)]
pub struct Nvfp4SplitPayload {
    /// Packed 4-bit E2M1 weights (`out_features * in_features / 2` bytes).
    pub weight_u8: Vec<u8>,
    /// FP8 E4M3FN micro-scales (`out_features * (in_features / 16)` bytes).
    pub weight_scale_e4m3: Vec<u8>,
    /// Global FP32 scale factor.
    pub weight_scale_2_f32: f32,
}

/// Quantizes `src` into separate vLLM / SGLang / TensorRT-LLM tensors.
pub fn quantize_nvfp4_split_tensors(src: &[f32]) -> Nvfp4SplitPayload {
    assert_eq!(src.len() % 16, 0);
    let mut amax = 0.0f32;
    for &v in src {
        let av = v.abs();
        if av > amax {
            amax = av;
        }
    }

    let global_scale = (amax / (E2M1_MAX_FINITE * FP8_E4M3FN_MAX_FINITE)).max(1e-12);
    let inv_global = 1.0 / global_scale;

    let num_micros = src.len() / 16;
    let mut weight_scale_e4m3 = Vec::with_capacity(num_micros);
    let mut weight_u8 = Vec::with_capacity(src.len() / 2);

    for k in 0..num_micros {
        let sub = &src[k * 16..(k + 1) * 16];
        let mut sub_amax = 0.0f32;
        for &x in sub {
            let ax = x.abs();
            if ax > sub_amax {
                sub_amax = ax;
            }
        }
        let desired_s = (sub_amax * inv_global) / E2M1_MAX_FINITE;
        let fp8_code = f32_to_fp8_e4m3fn(desired_s);
        weight_scale_e4m3.push(fp8_code);
        let eff_s = fp8_e4m3fn_to_f32(fp8_code).max(1e-12);
        let inv_block = 1.0 / (global_scale * eff_s);

        for j in 0..8 {
            let q0 = f32_to_e2m1(sub[2 * j] * inv_block) & 0x0F;
            let q1 = f32_to_e2m1(sub[2 * j + 1] * inv_block) & 0x0F;
            weight_u8.push(q0 | (q1 << 4));
        }
    }

    Nvfp4SplitPayload {
        weight_u8,
        weight_scale_e4m3,
        weight_scale_2_f32: global_scale,
    }
}
