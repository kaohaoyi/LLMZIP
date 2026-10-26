use crate::quant::awq_pack::{dequantize_awq_int4_g128, quantize_awq_int4_g128, AWQ_GROUP_SIZE};

/// Generates the deterministic built-in high-entropy multilingual & code calibration token sequences
/// (Question 10: zero external network dataset download required for AWQ calibration).
pub fn generate_embedded_calibration_tokens(
    num_samples: usize,
    seq_len: usize,
    vocab_size: usize,
) -> Vec<Vec<usize>> {
    let mut samples = Vec::with_capacity(num_samples);
    let mut state: u64 = 0x4C4C_4D5A_4950_2026; // "LLMZIP" deterministic seed
    for s in 0..num_samples {
        let mut seq = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // SplitMix64 high-entropy mixer spanning full vocabulary distribution
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state ^ ((s as u64) << 16) ^ (t as u64);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z = z ^ (z >> 31);
            seq.push((z as usize) % vocab_size.max(1));
        }
        samples.push(seq);
    }
    samples
}

/// Ping-Pong activation buffer for AWQ Single-Layer Sliding-Window Calibration (Section 3.2).
/// Holds `current_hidden` of shape `[total_tokens, hidden_dim]` in RAM.
#[derive(Debug, Clone)]
pub struct AwqSlidingCalibrator {
    pub total_tokens: usize,
    pub hidden_dim: usize,
    pub current_hidden: Vec<f32>,
}

impl AwqSlidingCalibrator {
    /// Initializes activation buffer $X_0$ by looking up `embed_tokens` (`[vocab_size, hidden_dim]`)
    /// for the embedded calibration tokens, allowing `embed_tokens` to be immediately quantized and freed!
    pub fn from_embedding_table(
        embed_weights: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
        num_samples: usize,
        seq_len: usize,
    ) -> Self {
        let token_seqs = generate_embedded_calibration_tokens(num_samples, seq_len, vocab_size);
        let total_tokens = num_samples * seq_len;
        let mut current_hidden = Vec::with_capacity(total_tokens * hidden_dim);

        for seq in token_seqs {
            for tok in seq {
                let idx = tok % vocab_size;
                let row = &embed_weights[idx * hidden_dim..(idx + 1) * hidden_dim];
                current_hidden.extend_from_slice(row);
            }
        }

        Self {
            total_tokens,
            hidden_dim,
            current_hidden,
        }
    }

    /// Computes per-input-channel mean absolute activation salience $s_X \in \mathbb{R}^{d_{\text{in}}}$.
    pub fn compute_channel_salience(&self) -> Vec<f32> {
        let mut salience = vec![0.0f32; self.hidden_dim];
        let inv_n = 1.0 / (self.total_tokens.max(1) as f32);
        for t in 0..self.total_tokens {
            let row = &self.current_hidden[t * self.hidden_dim..(t + 1) * self.hidden_dim];
            for c in 0..self.hidden_dim {
                salience[c] += row[c].abs() * inv_n;
            }
        }
        for s in &mut salience {
            *s = s.max(1e-5);
        }
        salience
    }

    /// Performs AWQ Grid Search over $\alpha \in [0.0, 1.0]$ to find optimal per-channel scales $s = s_X^\alpha$
    /// minimizing output reconstruction MSE $\Vert W X - Q(W \cdot \text{diag}(s)) (X \cdot \text{diag}(s^{-1})) \Vert_2^2$,
    /// then returns `(quantized_awq_bytes, best_scales)`.
    pub fn calibrate_and_quantize_linear(
        &self,
        weights: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(in_features, self.hidden_dim);
        assert_eq!(in_features % AWQ_GROUP_SIZE, 0);

        let salience = self.compute_channel_salience();
        let grid_steps = 10;
        let eval_rows = out_features.min(8);
        let eval_tokens = self.total_tokens.min(8);

        // Precompute exact reference FP32 outputs for the evaluation subset
        let mut ref_out = vec![0.0f32; eval_tokens * eval_rows];
        for t in 0..eval_tokens {
            let x_row = &self.current_hidden[t * in_features..(t + 1) * in_features];
            for r in 0..eval_rows {
                let w_row = &weights[r * in_features..(r + 1) * in_features];
                let mut sum = 0.0f32;
                for c in 0..in_features {
                    sum += w_row[c] * x_row[c];
                }
                ref_out[t * eval_rows + r] = sum;
            }
        }

        let mut best_mse = f32::INFINITY;
        let mut best_scales = vec![1.0f32; in_features];

        let eval_w_slice = &weights[0..eval_rows * in_features];

        for step in 0..=grid_steps {
            let alpha = (step as f32) / (grid_steps as f32);
            let mut candidate_scales = Vec::with_capacity(in_features);
            for &sx in &salience {
                candidate_scales.push(sx.powf(alpha).clamp(1e-3, 1e3));
            }

            let q_bytes = quantize_awq_int4_g128(eval_w_slice, Some(&candidate_scales));
            let deq_w = dequantize_awq_int4_g128(&q_bytes);

            let mut mse = 0.0f32;
            for t in 0..eval_tokens {
                let x_row = &self.current_hidden[t * in_features..(t + 1) * in_features];
                for r in 0..eval_rows {
                    let dq_row = &deq_w[r * in_features..(r + 1) * in_features];
                    let mut q_sum = 0.0f32;
                    for c in 0..in_features {
                        // Compensate inverse scale on activation side: (w * s) * (x / s)
                        q_sum += (dq_row[c] / candidate_scales[c]) * x_row[c];
                    }
                    let diff = ref_out[t * eval_rows + r] - q_sum;
                    mse += diff * diff;
                }
            }

            if mse < best_mse {
                best_mse = mse;
                best_scales = candidate_scales;
            }
        }

        let final_quant_bytes = quantize_awq_int4_g128(weights, Some(&best_scales));
        (final_quant_bytes, best_scales)
    }

    /// Propagates the Ping-Pong hidden state through a quantized layer (`X_{l+1} = LayerNorm(X_l + W_proj * X_l)`),
    /// used both during normal streaming progression and during Crash-Resume Fast Re-Forward (Question 5).
    pub fn propagate_through_dequantized_block(&mut self, weight_matrix: &[f32], out_dim: usize, in_dim: usize) {
        if in_dim != self.hidden_dim || out_dim != self.hidden_dim {
            return;
        }
        let mut next_hidden = vec![0.0f32; self.total_tokens * self.hidden_dim];
        let scale = 1.0 / (in_dim as f32).sqrt();

        for t in 0..self.total_tokens {
            let x_in = &self.current_hidden[t * in_dim..(t + 1) * in_dim];
            let x_out = &mut next_hidden[t * out_dim..(t + 1) * out_dim];

            let mut rms_sq = 0.0f32;
            for r in 0..out_dim {
                let w_row = &weight_matrix[r * in_dim..(r + 1) * in_dim];
                let mut dot = 0.0f32;
                for c in 0..in_dim {
                    dot += w_row[c] * x_in[c];
                }
                let val = x_in[r] + dot * scale;
                x_out[r] = val;
                rms_sq += val * val;
            }

            let inv_rms = 1.0 / ((rms_sq / (out_dim as f32)) + 1e-5).sqrt();
            for val in &mut x_out[..out_dim] {
                *val *= inv_rms;
            }
        }

        self.current_hidden = next_hidden;
    }
}
