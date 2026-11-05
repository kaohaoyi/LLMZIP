use crate::error::{LlmZipError, Result};
use crate::hasher::StreamingSha256;
use crate::io_prealloc::PreallocatedOutputFile;
use crate::layout::{DeterministicLayoutPlan, QuantFormat, QuantPreset};
use crate::preflight::{TensorRole, VirtualTensorSpace};
use crate::quant::awq_calib::AwqSlidingCalibrator;
use crate::quant::gguf_simd::{decode_and_validate_upstream_to_f32, dequantize_to_f32, quantize_f32_slice};
use crate::ringbuf::LayerRingBuffer;
use crate::wal::MicroWal;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

/// Upstream HTTP status & range metadata for [E09] verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpResponseMeta {
    /// HTTP status code (e.g. `200` or `206`).
    pub status_code: u16,
    /// Parsed start offset from `Content-Range: bytes <start>-<end>/<total>` when `status_code == 206`.
    pub content_range_start: Option<u64>,
}

/// Validates an HTTP Range response against `requested_offset` [E09].
/// Returns the number of leading bytes that must be discarded from the incoming body stream
/// (`0` for a valid `206 Partial Content`, or `requested_offset` if a mirror ignores `Range` and returns `200 OK`).
pub fn verify_http_range_response(
    shard_name: &str,
    requested_offset: u64,
    resp: HttpResponseMeta,
) -> Result<u64> {
    if requested_offset == 0 {
        if resp.status_code == 200 {
            return Ok(0);
        }
        if resp.status_code == 206 {
            let actual = resp.content_range_start.unwrap_or(u64::MAX);
            if actual != 0 {
                return Err(LlmZipError::HttpRangeViolation {
                    shard_name: shard_name.to_string(),
                    expected_start: 0,
                    actual_start: actual,
                });
            }
            return Ok(0);
        }
    } else {
        if resp.status_code == 206 {
            let actual = resp.content_range_start.unwrap_or(u64::MAX);
            if actual != requested_offset {
                return Err(LlmZipError::HttpRangeViolation {
                    shard_name: shard_name.to_string(),
                    expected_start: requested_offset,
                    actual_start: actual,
                });
            }
            return Ok(0);
        }
        if resp.status_code == 200 {
            // Fallback for mirrors that ignore Range headers and return 200 OK full body [E09]:
            // discard the first `requested_offset` bytes before feeding the hasher & quantizer.
            return Ok(requested_offset);
        }
    }

    Err(LlmZipError::HttpRangeViolation {
        shard_name: shard_name.to_string(),
        expected_start: requested_offset,
        actual_start: resp.content_range_start.unwrap_or(u64::MAX),
    })
}

/// Network condition & hardware line profile for tuning chunk sizes, batch parallelism, and WAL disk flushing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineProfile {
    /// High-speed fiber (>500Mbps - 1Gbps+ or local disk):
    /// Max batch chunk size (1 MiB), full multi-threading, batched WAL disk flushes
    /// to avoid stalling the ultra-high network bandwidth.
    Fast,
    /// Standard broadband (100Mbps - 300Mbps):
    /// Balanced chunk size (256 KiB), adaptive worker threads, per-tensor WAL checkpoints.
    Standard,
    /// Flaky/Slow/Edge connections (<50Mbps or high packet loss):
    /// Small chunk size (64 KiB), strict per-chunk WAL sync, low CPU overhead
    /// to prioritize 0-byte redownload resume and connection stability.
    Slow,
    /// Auto-detect line quality and throughput.
    #[default]
    Auto,
}

impl LineProfile {
    pub fn chunk_size(&self) -> usize {
        match self {
            LineProfile::Fast | LineProfile::Auto => 1024 * 1024, // 1 MiB
            LineProfile::Standard => 256 * 1024,                  // 256 KiB
            LineProfile::Slow => 64 * 1024,                       // 64 KiB
        }
    }

    pub fn wal_sync_interval(&self) -> usize {
        match self {
            LineProfile::Fast | LineProfile::Auto => 16,
            LineProfile::Standard => 4,
            LineProfile::Slow => 1,
        }
    }
}

/// Runtime configuration for `StreamingQuantizeEngine`.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// If true, clamps `NaN`/`Inf` in upstream weights instead of aborting [E04].
    pub sanitize_non_finite: bool,
    /// Maximum RAM budget in bytes for the single-layer ring buffer.
    pub ring_buffer_budget_bytes: usize,
    /// Number of calibration sequences for AWQ local calibration.
    pub awq_calib_samples: usize,
    /// Sequence length for AWQ local calibration.
    pub awq_calib_seq_len: usize,
    /// Adaptive line profile for network speed and batch quantization.
    pub line_profile: LineProfile,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            sanitize_non_finite: false,
            ring_buffer_budget_bytes: 512 * 1024 * 1024, // 512 MiB default single-layer budget
            awq_calib_samples: 8,
            awq_calib_seq_len: 32,
            line_profile: LineProfile::default(),
        }
    }
}

/// Core HatchZip-inspired Streaming Quantization & In-Place Model Loader Engine.
pub struct StreamingQuantizeEngine {
    vtas: VirtualTensorSpace,
    plan: DeterministicLayoutPlan,
    config: EngineConfig,
    output_file: PreallocatedOutputFile,
    wal_path: PathBuf,
    wal_file: std::fs::File,
    wal: MicroWal,
    hasher: StreamingSha256,
    ring_buf: LayerRingBuffer,
    awq_calibrator: Option<AwqSlidingCalibrator>,
    stream_cursor: u64,
}

impl StreamingQuantizeEngine {
    /// Initializes a new streaming quantization session or resumes from an existing `.wal` checkpoint.
    pub fn new_or_resume(
        vtas: VirtualTensorSpace,
        plan: DeterministicLayoutPlan,
        output_path: &Path,
        config: EngineConfig,
    ) -> Result<Self> {
        let mut wal_os = output_path.as_os_str().to_os_string();
        wal_os.push(".wal");
        let wal_path = PathBuf::from(wal_os);

        let resume = wal_path.exists() && output_path.exists();
        let output_file = PreallocatedOutputFile::create_or_resume(output_path, &plan, resume)?;

        let (wal, hasher) = if resume {
            let loaded_wal = MicroWal::load_from_path(&wal_path)?;
            let restored_hasher = StreamingSha256::from_midstate(loaded_wal.sha256_midstate);
            (loaded_wal, restored_hasher)
        } else {
            let fresh_hasher = StreamingSha256::new();
            let fresh_wal = MicroWal::new(vtas.tensors.len(), fresh_hasher.export_midstate());
            (fresh_wal, fresh_hasher)
        };

        let mut wal_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&wal_path)?;
        wal.sync_to_file(&mut wal_file)?;

        let ring_buf = LayerRingBuffer::new(config.ring_buffer_budget_bytes);

        let initial_cursor = if resume { wal.shard_confirmed_offset } else { 0 };

        let mut engine = Self {
            vtas,
            plan,
            config,
            output_file,
            wal_path,
            wal_file,
            wal,
            hasher,
            ring_buf,
            awq_calibrator: None,
            stream_cursor: initial_cursor,
        };

        // If resuming an AWQ local calibration session and `embed_tokens` was already quantized to disk,
        // reconstruct the calibration activations from the on-disk quantized layers (Question 5: 0-byte re-download).
        if resume && engine.plan.preset == QuantPreset::AwqW4G128 {
            engine.recover_awq_calibration_state_from_disk()?;
        }

        Ok(engine)
    }

    /// Reconstructs AWQ calibration hidden states on crash resume by reading and dequantizing
    /// already-completed layers directly from the pre-allocated output file.
    fn recover_awq_calibration_state_from_disk(&mut self) -> Result<()> {
        for tensor in &self.vtas.tensors {
            if !self.wal.is_tensor_completed(tensor.tensor_id) {
                break;
            }
            let target = &self.plan.write_targets[tensor.tensor_id];
            let mut q_bytes = vec![0u8; target.quantized_byte_len as usize];
            self.output_file.pread_exact(target.file_write_offset, &mut q_bytes)?;
            let deq_f32 = dequantize_to_f32(&q_bytes, target.assigned_quant);

            if tensor.role == TensorRole::Embedding && tensor.shape.len() == 2 {
                let vocab_size = tensor.shape[0];
                let hidden_dim = tensor.shape[1];
                self.awq_calibrator = Some(AwqSlidingCalibrator::from_embedding_table(
                    &deq_f32,
                    vocab_size,
                    hidden_dim,
                    self.config.awq_calib_samples,
                    self.config.awq_calib_seq_len,
                ));
            } else if tensor.shape.len() == 2 {
                if let Some(ref mut calib) = self.awq_calibrator {
                    calib.propagate_through_dequantized_block(&deq_f32, tensor.shape[0], tensor.shape[1]);
                }
            }
        }
        Ok(())
    }

    pub fn current_wal(&self) -> &MicroWal {
        &self.wal
    }

    /// Returns `(current_shard_idx, confirmed_byte_offset)` for issuing the next HTTP `Range: bytes=<offset>-` request.
    pub fn resume_cursor(&self) -> (usize, u64) {
        (self.wal.current_shard_idx as usize, self.stream_cursor)
    }

    /// Streams a shard from `shard_bytes_from_offset` (which begins at `start_offset` in the shard),
    /// feeding the sequential SHA-256 hasher, quantizing completed tensors in-place, and checkpointing `.wal` after each tensor.
    /// If `stop_after_tensors` is `Some(k)`, simulates a network interruption after committing `k` tensors.
    pub fn stream_shard_slice(
        &mut self,
        shard_idx: usize,
        http_meta: HttpResponseMeta,
        shard_bytes_from_response: &[u8],
        chunk_size: usize,
        stop_after_tensors: Option<usize>,
    ) -> Result<usize> {
        let shard = self.vtas.shards[shard_idx].clone();
        let expected_offset = if self.wal.current_shard_idx as usize == shard_idx {
            self.stream_cursor
        } else {
            0
        };

        let discard_bytes = verify_http_range_response(&shard.shard_name, expected_offset, http_meta)?;
        if (discard_bytes as usize) > shard_bytes_from_response.len() {
            return Err(LlmZipError::HttpRangeViolation {
                shard_name: shard.shard_name.clone(),
                expected_start: expected_offset,
                actual_start: shard_bytes_from_response.len() as u64,
            });
        }

        let effective_stream = &shard_bytes_from_response[discard_bytes as usize..];
        let mut abs_cursor = expected_offset;
        let mut tensors_committed_this_call = 0usize;
        let mut wal_dirty = false;
        let sync_interval = self.config.line_profile.wal_sync_interval().max(1);

        let shard_tensor_ids: Vec<usize> = self
            .vtas
            .tensors
            .iter()
            .filter(|t| t.shard_idx == shard_idx)
            .map(|t| t.tensor_id)
            .collect();

        let mut active_idx = 0usize;

        for chunk in effective_stream.chunks(chunk_size.max(1)) {
            let chunk_start = abs_cursor;
            let chunk_end = chunk_start + (chunk.len() as u64);

            // 1. Feed header bytes (if we are below 8 + header_len) into the sequential SHA-256 hasher.
            let data_section_start = 8 + shard.header_len;
            if chunk_start < data_section_start {
                let header_end_in_chunk = data_section_start.min(chunk_end);
                let slice_len = (header_end_in_chunk - chunk_start) as usize;
                self.hasher.update(&chunk[..slice_len]);
            }

            // Advance active_idx past any tensors that ended before this chunk
            while active_idx < shard_tensor_ids.len() {
                let tid = shard_tensor_ids[active_idx];
                if self.vtas.tensors[tid].stream_abs_end <= chunk_start {
                    active_idx += 1;
                } else {
                    break;
                }
            }

            // 2. Process candidate overlapping tensors in physical order.
            for &tid in shard_tensor_ids.iter().skip(active_idx) {
                let t_meta = &self.vtas.tensors[tid];
                let t_start = t_meta.stream_abs_start;
                let t_end = t_meta.stream_abs_end;

                if chunk_end <= t_start {
                    // Tensor starts after this chunk, and subsequent tensors start even later -> break early!
                    break;
                }
                if chunk_start >= t_end {
                    continue;
                }

                let overlap_start = chunk_start.max(t_start);
                let overlap_end = chunk_end.min(t_end);
                let local_start = (overlap_start - chunk_start) as usize;
                let local_end = (overlap_end - chunk_start) as usize;
                let tensor_slice = &chunk[local_start..local_end];

                // Feed sequential SHA-256 hasher
                self.hasher.update(tensor_slice);

                let expected_tensor_bytes = (t_end - t_start) as usize;
                if let Some(full_raw_bytes) =
                    self.ring_buf.push_tensor_chunk(tid, expected_tensor_bytes, tensor_slice)?
                {
                    // Full tensor assembled in RAM Ring Buffer -> Validate, Quantize & In-Place Write!
                    self.quantize_and_commit_tensor(tid, &full_raw_bytes)?;

                    // Checkpoint WAL at exact tensor boundary (`t_end`)
                    self.wal.current_shard_idx = shard_idx as u32;
                    self.wal.shard_confirmed_offset = t_end;
                    self.wal.sha256_midstate = self.hasher.export_midstate();
                    self.wal.mark_tensor_completed(tid);
                    wal_dirty = true;

                    tensors_committed_this_call += 1;
                    if tensors_committed_this_call.is_multiple_of(sync_interval) {
                        self.output_file.sync_data()?;
                        self.wal.sync_to_file(&mut self.wal_file)?;
                        wal_dirty = false;
                    }

                    if let Some(limit) = stop_after_tensors {
                        if tensors_committed_this_call >= limit {
                            if wal_dirty {
                                self.output_file.sync_data()?;
                                self.wal.sync_to_file(&mut self.wal_file)?;
                            }
                            return Ok(tensors_committed_this_call);
                        }
                    }
                }
            }

            abs_cursor = chunk_end;
        }

        self.stream_cursor = abs_cursor;

        if wal_dirty {
            self.output_file.sync_data()?;
            self.wal.sync_to_file(&mut self.wal_file)?;
        }

        // If we reached EOF of the current shard, verify its full cryptographic LFS SHA-256 [E03]!
        if abs_cursor == shard.total_file_size {
            let actual_hex = self.hasher.finalize_hex();
            if !shard.expected_sha256_hex.is_empty()
                && !actual_hex.eq_ignore_ascii_case(&shard.expected_sha256_hex)
            {
                self.output_file.quarantine_and_abort()?;
                return Err(LlmZipError::Sha256Mismatch {
                    shard_name: shard.shard_name.clone(),
                    expected_hex: shard.expected_sha256_hex.clone(),
                    actual_hex,
                });
            }

            // Prepare WAL & hasher for the next shard if any remain
            if shard_idx + 1 < self.vtas.shards.len() {
                self.hasher = StreamingSha256::new();
                self.wal.current_shard_idx = (shard_idx + 1) as u32;
                self.wal.shard_confirmed_offset = 0;
                self.stream_cursor = 0;
                self.wal.sha256_midstate = self.hasher.export_midstate();
                self.wal.sync_to_file(&mut self.wal_file)?;
            } else if self.wal.all_completed() {
                // All shards verified and all tensors written -> Unlock "GGUF" Magic Header [E03]!
                self.output_file.commit_verified()?;
                let _ = fs::remove_file(&self.wal_path);
            }
        }

        Ok(tensors_committed_this_call)
    }

    fn quantize_and_commit_tensor(&mut self, tensor_id: usize, raw_bytes: &[u8]) -> Result<()> {
        let t_meta = &self.vtas.tensors[tensor_id];
        let target = &self.plan.write_targets[tensor_id];

        // 1. Decode upstream F16/BF16/F32 into FP32 registers [E11] & check NaN/Inf [E04].
        let f32_weights = decode_and_validate_upstream_to_f32(
            &t_meta.hf_name,
            raw_bytes,
            t_meta.dtype,
            self.config.sanitize_non_finite,
        )?;

        // 2. If in AWQ mode and this is the Embedding table, initialize the Single-Layer Sliding-Window Calibrator!
        if self.plan.preset == QuantPreset::AwqW4G128
            && t_meta.role == TensorRole::Embedding
            && t_meta.shape.len() == 2
        {
            self.awq_calibrator = Some(AwqSlidingCalibrator::from_embedding_table(
                &f32_weights,
                t_meta.shape[0],
                t_meta.shape[1],
                self.config.awq_calib_samples,
                self.config.awq_calib_seq_len,
            ));
        }

        // 3. Quantize according to assigned QuantFormat (using AWQ Sliding-Window Calibration if applicable).
        let quantized_bytes = if target.assigned_quant == QuantFormat::AwqInt4G128 && t_meta.shape.len() == 2 {
            let out_dim = t_meta.shape[0];
            let in_dim = t_meta.shape[1];
            if let Some(ref mut calib) = self.awq_calibrator {
                if calib.hidden_dim == in_dim {
                    let (q_bytes, _scales) = calib.calibrate_and_quantize_linear(&f32_weights, out_dim, in_dim);
                    let deq = dequantize_to_f32(&q_bytes, QuantFormat::AwqInt4G128);
                    calib.propagate_through_dequantized_block(&deq, out_dim, in_dim);
                    q_bytes
                } else {
                    quantize_f32_slice(&f32_weights, target.assigned_quant)
                }
            } else {
                quantize_f32_slice(&f32_weights, target.assigned_quant)
            }
        } else {
            quantize_f32_slice(&f32_weights, target.assigned_quant)
        };

        // 4. Write directly to the pre-allocated target offset (including MoE 3D sub-offsets).
        self.output_file
            .pwrite_all(target.file_write_offset, &quantized_bytes)?;

        // 5. If `tie_word_embeddings` is active, write the tied `output.weight` slot simultaneously [E07].
        if let Some(secondary_offset) = target.tied_secondary_offset {
            self.output_file.pwrite_all(secondary_offset, &quantized_bytes)?;
        }

        Ok(())
    }

    pub fn is_committed(&self) -> bool {
        self.output_file.is_committed()
    }

    pub fn commit_if_completed(&mut self) -> Result<bool> {
        if self.wal.all_completed() && !self.output_file.is_committed() {
            self.output_file.commit_verified()?;
            let _ = fs::remove_file(&self.wal_path);
            Ok(true)
        } else {
            Ok(self.output_file.is_committed())
        }
    }
}
