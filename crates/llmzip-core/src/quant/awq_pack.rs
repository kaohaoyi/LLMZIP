use crate::quant::gguf_simd::{f16_to_f32, f32_to_f16_clamped};

/// Group size for AWQ INT4 quantization.
pub const AWQ_GROUP_SIZE: usize = 128;
/// Bytes per AWQ 128-element group: 2B `f16` scale + 2B `f16` group_min + 64B packed 4-bit weights = 68 bytes.
pub const AWQ_GROUP_BYTES: usize = 68;

/// Quantizes `src` (whose length must be a multiple of 128) into AWQ INT4 group-128 format,
/// optionally applying per-input-channel activation scales `channel_scales` ($W \cdot \text{diag}(s)$).
pub fn quantize_awq_int4_g128(src: &[f32], channel_scales: Option<&[f32]>) -> Vec<u8> {
    assert_eq!(src.len() % AWQ_GROUP_SIZE, 0);
    let num_groups = src.len() / AWQ_GROUP_SIZE;
    let mut out = Vec::with_capacity(num_groups * AWQ_GROUP_BYTES);

    let inner_dim = channel_scales.map(|s| s.len()).unwrap_or(src.len());

    for g in 0..num_groups {
        let base_idx = g * AWQ_GROUP_SIZE;
        let grp = &src[base_idx..base_idx + AWQ_GROUP_SIZE];

        let mut scaled = [0.0f32; AWQ_GROUP_SIZE];
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        for i in 0..AWQ_GROUP_SIZE {
            let col = (base_idx + i) % inner_dim;
            let s_factor = channel_scales.map(|s| s[col]).unwrap_or(1.0);
            let v = grp[i] * s_factor;
            scaled[i] = v;
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }

        let min_f16 = f32_to_f16_clamped(min_v);
        let eff_min = f16_to_f32(min_f16);
        let range = (max_v - eff_min).max(0.0);
        let scale = range / 15.0;

        let scale_f16 = f32_to_f16_clamped(scale);
        let eff_scale = f16_to_f32(scale_f16);
        let eff_inv_scale = if eff_scale > 0.0 { 1.0 / eff_scale } else { 0.0 };

        out.extend_from_slice(&scale_f16.to_le_bytes());
        out.extend_from_slice(&min_f16.to_le_bytes());

        for i in 0..64 {
            let q0 = ((scaled[2 * i] - eff_min) * eff_inv_scale)
                .round()
                .clamp(0.0, 15.0) as u8;
            let q1 = ((scaled[2 * i + 1] - eff_min) * eff_inv_scale)
                .round()
                .clamp(0.0, 15.0) as u8;
            out.push(q0 | (q1 << 4));
        }
    }

    out
}

/// Dequantizes AWQ INT4 group-128 bytes back into `f32` weights.
pub fn dequantize_awq_int4_g128(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % AWQ_GROUP_BYTES, 0);
    let num_groups = bytes.len() / AWQ_GROUP_BYTES;
    let mut out = Vec::with_capacity(num_groups * AWQ_GROUP_SIZE);

    for g in 0..num_groups {
        let blk = &bytes[g * AWQ_GROUP_BYTES..(g + 1) * AWQ_GROUP_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let min_v = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let qw = &blk[4..68];

        for &packed in qw {
            let q0 = (packed & 0x0F) as f32;
            let q1 = (packed >> 4) as f32;
            out.push(min_v + q0 * scale);
            out.push(min_v + q1 * scale);
        }
    }

    out
}
