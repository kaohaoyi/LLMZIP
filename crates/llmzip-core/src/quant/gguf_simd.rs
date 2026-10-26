use crate::error::{LlmZipError, Result};
use crate::layout::QuantFormat;
use crate::preflight::UpstreamDType;
use rayon::prelude::*;

/// Maximum finite representable value in IEEE 754 binary16 (FP16).
pub const FP16_MAX_FINITE: f32 = 65504.0;

/// Converts IEEE 754 binary16 (`u16` bits) to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal f16 -> normalized f32
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Converts `f32` to IEEE 754 binary16 (`u16` bits), clamping out-of-range finite values to `±65504.0` [E11].
#[inline]
pub fn f32_to_f16_clamped(val: f32) -> u16 {
    if val.is_nan() {
        return 0;
    }
    let clamped = val.clamp(-FP16_MAX_FINITE, FP16_MAX_FINITE);
    let bits = clamped.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x007F_FFFF;

    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    } else if exp >= 31 {
        return sign | 0x7BFF; // Max finite FP16 (65504.0)
    }

    let rounded = mant + 0x0000_1000;
    if (rounded & 0x0080_0000) != 0 {
        let new_exp = exp + 1;
        if new_exp >= 31 {
            sign | 0x7BFF
        } else {
            sign | ((new_exp as u16) << 10)
        }
    } else {
        sign | ((exp as u16) << 10) | ((rounded >> 13) as u16)
    }
}

/// Converts `BF16` (`u16` bits) directly into register-level `f32` without overflow [E11].
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decodes raw upstream tensor bytes (`F16`, `BF16`, or `F32`) into a register-widened `Vec<f32>` [E11],
/// enforcing vectorized `NaN`/`Inf` finite checks [E04].
pub fn decode_and_validate_upstream_to_f32(
    tensor_name: &str,
    raw_bytes: &[u8],
    dtype: UpstreamDType,
    sanitize_non_finite: bool,
) -> Result<Vec<f32>> {
    match dtype {
        UpstreamDType::F16 => {
            let count = raw_bytes.len() / 2;
            let mut out = vec![0.0f32; count];
            if count >= 8192 {
                out.par_chunks_mut(8192)
                    .enumerate()
                    .try_for_each(|(chunk_idx, slice)| -> Result<()> {
                        let base = chunk_idx * 8192;
                        for (local_i, target) in slice.iter_mut().enumerate() {
                            let i = base + local_i;
                            let bits = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
                            let v = f16_to_f32(bits);
                            if !v.is_finite() {
                                if sanitize_non_finite {
                                    *target = if v.is_nan() {
                                        0.0
                                    } else if v.is_sign_positive() {
                                        FP16_MAX_FINITE
                                    } else {
                                        -FP16_MAX_FINITE
                                    };
                                } else {
                                    return Err(LlmZipError::NumericalPoisoning {
                                        tensor_name: tensor_name.to_string(),
                                        element_index: i,
                                        raw_bits: bits,
                                    });
                                }
                            } else {
                                *target = v;
                            }
                        }
                        Ok(())
                    })?;
            } else {
                for i in 0..count {
                    let bits = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
                    let v = f16_to_f32(bits);
                    if !v.is_finite() {
                        if sanitize_non_finite {
                            let clean = if v.is_nan() {
                                0.0
                            } else if v.is_sign_positive() {
                                FP16_MAX_FINITE
                            } else {
                                -FP16_MAX_FINITE
                            };
                            out[i] = clean;
                        } else {
                            return Err(LlmZipError::NumericalPoisoning {
                                tensor_name: tensor_name.to_string(),
                                element_index: i,
                                raw_bits: bits,
                            });
                        }
                    } else {
                        out[i] = v;
                    }
                }
            }
            Ok(out)
        }
        UpstreamDType::BF16 => {
            let count = raw_bytes.len() / 2;
            let mut out = vec![0.0f32; count];
            if count >= 8192 {
                out.par_chunks_mut(8192)
                    .enumerate()
                    .try_for_each(|(chunk_idx, slice)| -> Result<()> {
                        let base = chunk_idx * 8192;
                        for (local_i, target) in slice.iter_mut().enumerate() {
                            let i = base + local_i;
                            let bits = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
                            let v = bf16_to_f32(bits);
                            if !v.is_finite() {
                                if sanitize_non_finite {
                                    *target = if v.is_nan() {
                                        0.0
                                    } else if v.is_sign_positive() {
                                        FP16_MAX_FINITE
                                    } else {
                                        -FP16_MAX_FINITE
                                    };
                                } else {
                                    return Err(LlmZipError::NumericalPoisoning {
                                        tensor_name: tensor_name.to_string(),
                                        element_index: i,
                                        raw_bits: bits,
                                    });
                                }
                            } else {
                                *target = v.clamp(-FP16_MAX_FINITE, FP16_MAX_FINITE);
                            }
                        }
                        Ok(())
                    })?;
            } else {
                for i in 0..count {
                    let bits = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
                    let v = bf16_to_f32(bits);
                    if !v.is_finite() {
                        if sanitize_non_finite {
                            let clean = if v.is_nan() {
                                0.0
                            } else if v.is_sign_positive() {
                                FP16_MAX_FINITE
                            } else {
                                -FP16_MAX_FINITE
                            };
                            out[i] = clean;
                        } else {
                            return Err(LlmZipError::NumericalPoisoning {
                                tensor_name: tensor_name.to_string(),
                                element_index: i,
                                raw_bits: bits,
                            });
                        }
                    } else {
                        out[i] = v.clamp(-FP16_MAX_FINITE, FP16_MAX_FINITE);
                    }
                }
            }
            Ok(out)
        }
        UpstreamDType::F32 => {
            let count = raw_bytes.len() / 4;
            let mut out = vec![0.0f32; count];
            if count >= 8192 {
                out.par_chunks_mut(8192)
                    .enumerate()
                    .try_for_each(|(chunk_idx, slice)| -> Result<()> {
                        let base = chunk_idx * 8192;
                        for (local_i, target) in slice.iter_mut().enumerate() {
                            let i = base + local_i;
                            let mut b = [0u8; 4];
                            b.copy_from_slice(&raw_bytes[4 * i..4 * i + 4]);
                            let v = f32::from_le_bytes(b);
                            if !v.is_finite() {
                                if sanitize_non_finite {
                                    *target = if v.is_nan() {
                                        0.0
                                    } else if v.is_sign_positive() {
                                        FP16_MAX_FINITE
                                    } else {
                                        -FP16_MAX_FINITE
                                    };
                                } else {
                                    return Err(LlmZipError::NumericalPoisoning {
                                        tensor_name: tensor_name.to_string(),
                                        element_index: i,
                                        raw_bits: (v.to_bits() >> 16) as u16,
                                    });
                                }
                            } else {
                                *target = v.clamp(-FP16_MAX_FINITE, FP16_MAX_FINITE);
                            }
                        }
                        Ok(())
                    })?;
            } else {
                for i in 0..count {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&raw_bytes[4 * i..4 * i + 4]);
                    let v = f32::from_le_bytes(b);
                    if !v.is_finite() {
                        if sanitize_non_finite {
                            let clean = if v.is_nan() {
                                0.0
                            } else if v.is_sign_positive() {
                                FP16_MAX_FINITE
                            } else {
                                -FP16_MAX_FINITE
                            };
                            out[i] = clean;
                        } else {
                            return Err(LlmZipError::NumericalPoisoning {
                                tensor_name: tensor_name.to_string(),
                                element_index: i,
                                raw_bits: (v.to_bits() >> 16) as u16,
                            });
                        }
                    } else {
                        out[i] = v.clamp(-FP16_MAX_FINITE, FP16_MAX_FINITE);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Quantizes `f32` slice into GGUF `Q8_0` blocks (32 elements -> 34 bytes: 2B f16 `d` + 32B `i8` `qs`).
pub fn quantize_q8_0(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 32, 0);
    let num_blocks = src.len() / 32;
    let mut out = vec![0u8; num_blocks * 34];

    if num_blocks >= 64 {
        out.par_chunks_mut(34 * 64)
            .enumerate()
            .for_each(|(batch_idx, batch_out)| {
                let start_b = batch_idx * 64;
                let blocks_in_batch = batch_out.len() / 34;
                for b in 0..blocks_in_batch {
                    let global_b = start_b + b;
                    let block = &src[global_b * 32..(global_b + 1) * 32];
                    let mut amax = 0.0f32;
                    for &v in block {
                        let av = v.abs();
                        if av > amax {
                            amax = av;
                        }
                    }
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    let d_f16 = f32_to_f16_clamped(d);
                    let out_slice = &mut batch_out[b * 34..(b + 1) * 34];
                    out_slice[0..2].copy_from_slice(&d_f16.to_le_bytes());
                    for i in 0..32 {
                        let q = (block[i] * id).round().clamp(-127.0, 127.0) as i8;
                        out_slice[2 + i] = q as u8;
                    }
                }
            });
    } else {
        for b in 0..num_blocks {
            let block = &src[b * 32..(b + 1) * 32];
            let mut amax = 0.0f32;
            for &v in block {
                let av = v.abs();
                if av > amax {
                    amax = av;
                }
            }
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let d_f16 = f32_to_f16_clamped(d);
            let out_slice = &mut out[b * 34..(b + 1) * 34];
            out_slice[0..2].copy_from_slice(&d_f16.to_le_bytes());
            for i in 0..32 {
                let q = (block[i] * id).round().clamp(-127.0, 127.0) as i8;
                out_slice[2 + i] = q as u8;
            }
        }
    }
    out
}

/// Dequantizes GGUF `Q8_0` blocks back to `f32`.
pub fn dequantize_q8_0(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 34, 0);
    let num_blocks = bytes.len() / 34;
    let mut out = Vec::with_capacity(num_blocks * 32);

    for b in 0..num_blocks {
        let chunk = &bytes[b * 34..(b + 1) * 34];
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        for i in 0..32 {
            let q = chunk[2 + i] as i8;
            out.push((q as f32) * d);
        }
    }
    out
}

/// Quantizes `f32` slice into GGUF `Q4_K` super-blocks (256 elements -> 144 bytes).
/// Layout of 144-byte `block_q4_K`:
/// - `d`: `f16` (2B) super-block scale
/// - `dmin`: `f16` (2B) super-block minimum scale
/// - `scales`: `[u8; 12]` (12B) packed 6-bit sub-block scales & mins for 8 sub-blocks of 32
/// - `qs`: `[u8; 128]` (128B) packed 4-bit unsigned values `[0..15]`
pub fn quantize_q4_k(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 256, 0);
    let num_blocks = src.len() / 256;
    let mut out = vec![0u8; num_blocks * 144];

    let quant_one_sb = |sb: &[f32], out_slot: &mut [u8]| {
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;

        for j in 0..8 {
            let sub = &sb[j * 32..(j + 1) * 32];
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &x in sub {
                if x < min_v {
                    min_v = x;
                }
                if x > max_v {
                    max_v = x;
                }
            }
            // Offset minimum so non-negative min_offset = max(0, -min_v)
            let m = (-min_v).max(0.0);
            let s = (max_v + m) / 15.0;
            sub_scales[j] = s;
            sub_mins[j] = m;
            if s > max_scale {
                max_scale = s;
            }
            if m > max_min {
                max_min = m;
            }
        }

        let d = max_scale / 63.0;
        let dmin = max_min / 63.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

        let d_f16 = f32_to_f16_clamped(d);
        let dmin_f16 = f32_to_f16_clamped(dmin);
        let eff_d = f16_to_f32(d_f16);
        let eff_dmin = f16_to_f16_roundtrip(dmin_f16);

        let mut ls = [0u8; 8];
        let mut lm = [0u8; 8];
        for j in 0..8 {
            ls[j] = (sub_scales[j] * inv_d).round().clamp(0.0, 63.0) as u8;
            lm[j] = (sub_mins[j] * inv_dmin).round().clamp(0.0, 63.0) as u8;
        }

        // Pack 8x 6-bit `ls` and 8x 6-bit `lm` into 12 bytes (standard ggml_q4_K packing)
        let mut scales_packed = [0u8; 12];
        for j in 0..4 {
            scales_packed[j] = (ls[j] & 63) | ((ls[j + 4] >> 4) << 6);
            scales_packed[j + 4] = (lm[j] & 63) | ((lm[j + 4] >> 4) << 6);
            scales_packed[j + 8] = (ls[j + 4] & 0x0F) | ((lm[j + 4] & 0x0F) << 4);
        }

        let mut qs = [0u8; 128];
        for j in 0..4 {
            let sc0 = eff_d * (ls[2 * j] as f32);
            let mn0 = eff_dmin * (lm[2 * j] as f32);
            let sc1 = eff_d * (ls[2 * j + 1] as f32);
            let mn1 = eff_dmin * (lm[2 * j + 1] as f32);

            let isc0 = if sc0 > 0.0 { 1.0 / sc0 } else { 0.0 };
            let isc1 = if sc1 > 0.0 { 1.0 / sc1 } else { 0.0 };

            let sub0 = &sb[(2 * j) * 32..(2 * j + 1) * 32];
            let sub1 = &sb[(2 * j + 1) * 32..(2 * j + 2) * 32];

            for l in 0..32 {
                let q0 = ((sub0[l] + mn0) * isc0).round().clamp(0.0, 15.0) as u8;
                let q1 = ((sub1[l] + mn1) * isc1).round().clamp(0.0, 15.0) as u8;
                qs[j * 32 + l] = q0 | (q1 << 4);
            }
        }

        out_slot[0..2].copy_from_slice(&d_f16.to_le_bytes());
        out_slot[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
        out_slot[4..16].copy_from_slice(&scales_packed);
        out_slot[16..144].copy_from_slice(&qs);
    };

    if num_blocks >= 16 {
        out.par_chunks_mut(144 * 16)
            .enumerate()
            .for_each(|(batch_idx, batch_out)| {
                let start_b = batch_idx * 16;
                let blocks_in_batch = batch_out.len() / 144;
                for b in 0..blocks_in_batch {
                    let global_b = start_b + b;
                    let sb = &src[global_b * 256..(global_b + 1) * 256];
                    let slot = &mut batch_out[b * 144..(b + 1) * 144];
                    quant_one_sb(sb, slot);
                }
            });
    } else {
        for b in 0..num_blocks {
            let sb = &src[b * 256..(b + 1) * 256];
            let slot = &mut out[b * 144..(b + 1) * 144];
            quant_one_sb(sb, slot);
        }
    }
    out
}

#[inline]
fn f16_to_f16_roundtrip(bits: u16) -> f32 {
    f16_to_f32(bits)
}

/// Dequantizes GGUF `Q4_K` super-blocks (144 bytes -> 256 `f32` elements).
pub fn dequantize_q4_k(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 144, 0);
    let num_blocks = bytes.len() / 144;
    let mut out = Vec::with_capacity(num_blocks * 256);

    for b in 0..num_blocks {
        let blk = &bytes[b * 144..(b + 1) * 144];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let sp = &blk[4..16];
        let qs = &blk[16..144];

        let mut ls = [0u8; 8];
        let mut lm = [0u8; 8];
        for j in 0..4 {
            ls[j] = sp[j] & 63;
            lm[j] = sp[j + 4] & 63;
            ls[j + 4] = (sp[j + 8] & 0x0F) | ((sp[j] >> 6) << 4);
            lm[j + 4] = (sp[j + 8] >> 4) | ((sp[j + 4] >> 6) << 4);
        }

        for j in 0..4 {
            let sc0 = d * (ls[2 * j] as f32);
            let mn0 = dmin * (lm[2 * j] as f32);
            let sc1 = d * (ls[2 * j + 1] as f32);
            let mn1 = dmin * (lm[2 * j + 1] as f32);

            for l in 0..32 {
                let packed = qs[j * 32 + l];
                let q0 = (packed & 0x0F) as f32;
                let q1 = (packed >> 4) as f32;
                out.push(q0 * sc0 - mn0);
                // Temporarily store sub0 then sub1 in order
                let _ = q1;
            }
            for l in 0..32 {
                let packed = qs[j * 32 + l];
                let q1 = (packed >> 4) as f32;
                out.push(q1 * sc1 - mn1);
            }
        }
    }
    out
}

/// Quantizes `f32` slice into GGUF `Q6_K` super-blocks (256 elements -> 210 bytes).
/// Layout of 210-byte `block_q6_K`:
/// - `ql`: `[u8; 128]` (low 4 bits of 6-bit quants)
/// - `qh`: `[u8; 64]` (high 2 bits of 6-bit quants)
/// - `scales`: `[i8; 16]` (16 sub-block scales for 16 groups of 16 elements)
/// - `d`: `f16` (2B) super-block scale
pub fn quantize_q6_k(src: &[f32]) -> Vec<u8> {
    assert_eq!(src.len() % 256, 0);
    let num_blocks = src.len() / 256;
    let mut out = vec![0u8; num_blocks * 210];

    let quant_one_q6_sb = |sb: &[f32], out_slot: &mut [u8]| {
        let mut sub_scales = [0.0f32; 16];
        let mut max_abs_scale = 0.0f32;

        for g in 0..16 {
            let sub = &sb[g * 16..(g + 1) * 16];
            let mut amax = 0.0f32;
            for &v in sub {
                if v.abs() > amax {
                    amax = v.abs();
                }
            }
            let s = amax / 31.0;
            sub_scales[g] = s;
            if s > max_abs_scale {
                max_abs_scale = s;
            }
        }

        let d = max_abs_scale / 127.0;
        let d_f16 = f32_to_f16_clamped(d);
        let eff_d = f16_to_f32(d_f16);
        let inv_d = if eff_d > 0.0 { 1.0 / eff_d } else { 0.0 };

        let mut scales_i8 = [0i8; 16];
        for g in 0..16 {
            scales_i8[g] = (sub_scales[g] * inv_d).round().clamp(-127.0, 127.0) as i8;
        }

        let mut quants = [32u8; 256];
        for g in 0..16 {
            let sc = eff_d * (scales_i8[g] as f32);
            let isc = if sc.abs() > 0.0 { 1.0 / sc } else { 0.0 };
            for i in 0..16 {
                let v = sb[g * 16 + i];
                let q = (v * isc).round().clamp(-32.0, 31.0) as i8;
                quants[g * 16 + i] = (q + 32) as u8; // 0..63
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for i in 0..128 {
            let q0 = quants[2 * i] & 0x3F;
            let q1 = quants[2 * i + 1] & 0x3F;
            ql[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        for i in 0..64 {
            let q0 = (quants[4 * i] >> 4) & 0x03;
            let q1 = (quants[4 * i + 1] >> 4) & 0x03;
            let q2 = (quants[4 * i + 2] >> 4) & 0x03;
            let q3 = (quants[4 * i + 3] >> 4) & 0x03;
            qh[i] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
        }

        out_slot[0..128].copy_from_slice(&ql);
        out_slot[128..192].copy_from_slice(&qh);
        for k in 0..16 {
            out_slot[192 + k] = scales_i8[k] as u8;
        }
        out_slot[208..210].copy_from_slice(&d_f16.to_le_bytes());
    };

    if num_blocks >= 16 {
        out.par_chunks_mut(210 * 16)
            .enumerate()
            .for_each(|(batch_idx, batch_out)| {
                let start_b = batch_idx * 16;
                let blocks_in_batch = batch_out.len() / 210;
                for b in 0..blocks_in_batch {
                    let global_b = start_b + b;
                    let sb = &src[global_b * 256..(global_b + 1) * 256];
                    let slot = &mut batch_out[b * 210..(b + 1) * 210];
                    quant_one_q6_sb(sb, slot);
                }
            });
    } else {
        for b in 0..num_blocks {
            let sb = &src[b * 256..(b + 1) * 256];
            let slot = &mut out[b * 210..(b + 1) * 210];
            quant_one_q6_sb(sb, slot);
        }
    }
    out
}

/// Dequantizes GGUF `Q6_K` super-blocks (210 bytes -> 256 `f32` elements).
pub fn dequantize_q6_k(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 210, 0);
    let num_blocks = bytes.len() / 210;
    let mut out = Vec::with_capacity(num_blocks * 256);

    for b in 0..num_blocks {
        let blk = &bytes[b * 210..(b + 1) * 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));

        for idx in 0..256 {
            let low4 = if idx % 2 == 0 {
                ql[idx / 2] & 0x0F
            } else {
                ql[idx / 2] >> 4
            };
            let high2 = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x03;
            let q = ((low4 | (high2 << 4)) as i8) - 32;
            let sc = scales[idx / 16] as i8;
            out.push(d * (sc as f32) * (q as f32));
        }
    }
    out
}

/// Unified entry point to quantize a validated `f32` tensor slice into `QuantFormat`.
pub fn quantize_f32_slice(src: &[f32], format: QuantFormat) -> Vec<u8> {
    match format {
        QuantFormat::GgufQ4K => quantize_q4_k(src),
        QuantFormat::GgufQ6K => quantize_q6_k(src),
        QuantFormat::GgufQ8_0 => quantize_q8_0(src),
        QuantFormat::GgufF16 => {
            let mut out = Vec::with_capacity(src.len() * 2);
            for &v in src {
                out.extend_from_slice(&f32_to_f16_clamped(v).to_le_bytes());
            }
            out
        }
        QuantFormat::GgufF32 => {
            let mut out = Vec::with_capacity(src.len() * 4);
            for &v in src {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        QuantFormat::AwqInt4G128 => crate::quant::awq_pack::quantize_awq_int4_g128(src, None),
        QuantFormat::Nvfp4Sb256 => crate::quant::nvfp4_microscale::quantize_nvfp4_sb256(src),
        QuantFormat::Nvfp4B16 => crate::quant::nvfp4_microscale::quantize_nvfp4_b16(src),
        QuantFormat::Mxfp4B32 => crate::quant::nvfp4_microscale::quantize_mxfp4_b32(src),
        QuantFormat::Fp8E4M3B32 => crate::quant::nvfp4_microscale::quantize_fp8_e4m3_b32(src),
    }
}

/// Unified entry point to dequantize a quantized byte slice back to `Vec<f32>`.
pub fn dequantize_to_f32(bytes: &[u8], format: QuantFormat) -> Vec<f32> {
    match format {
        QuantFormat::GgufQ4K => dequantize_q4_k(bytes),
        QuantFormat::GgufQ6K => dequantize_q6_k(bytes),
        QuantFormat::GgufQ8_0 => dequantize_q8_0(bytes),
        QuantFormat::GgufF16 => {
            let count = bytes.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                out.push(f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])));
            }
            out
        }
        QuantFormat::GgufF32 => {
            let count = bytes.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[4 * i..4 * i + 4]);
                out.push(f32::from_le_bytes(b));
            }
            out
        }
        QuantFormat::AwqInt4G128 => crate::quant::awq_pack::dequantize_awq_int4_g128(bytes),
        QuantFormat::Nvfp4Sb256 => crate::quant::nvfp4_microscale::dequantize_nvfp4_sb256(bytes),
        QuantFormat::Nvfp4B16 => crate::quant::nvfp4_microscale::dequantize_nvfp4_b16(bytes),
        QuantFormat::Mxfp4B32 => crate::quant::nvfp4_microscale::dequantize_mxfp4_b32(bytes),
        QuantFormat::Fp8E4M3B32 => crate::quant::nvfp4_microscale::dequantize_fp8_e4m3_b32(bytes),
    }
}
