use crate::error::{LlmZipError, Result};
use crate::hf::client::{HfClient, HfFileEntry};
use crate::hf::range_scheduler::{generate_chunks, ParallelRangeScheduler, DEFAULT_CHUNK_SIZE};
use crate::layout::{build_deterministic_layout, QuantPreset};
use crate::preflight::{build_virtual_tensor_space, parse_and_validate_header_len, ShardDescriptor};
use crate::streamer::{EngineConfig, HttpResponseMeta, StreamingQuantizeEngine};
use reqwest::header::{CONTENT_RANGE, RANGE};
use std::path::PathBuf;
use std::sync::Arc;

/// Configuration for the Native Hugging Face Parallel Streamer.
#[derive(Debug, Clone)]
pub struct HfStreamerConfig {
    pub repo_id: String,
    pub revision: Option<String>,
    pub auth_token: Option<String>,
    pub concurrency: usize,
    pub chunk_size: u64,
    pub preset: QuantPreset,
    pub output_path: PathBuf,
    pub base_url: Option<String>,
    pub expected_sha256: Option<String>,
}

impl Default for HfStreamerConfig {
    fn default() -> Self {
        Self {
            repo_id: String::new(),
            revision: None,
            auth_token: None,
            concurrency: 4,
            chunk_size: DEFAULT_CHUNK_SIZE,
            preset: QuantPreset::GgufQ4KM,
            output_path: PathBuf::from("model.gguf"),
            base_url: None,
            expected_sha256: None,
        }
    }
}

/// Native Hugging Face parallel multi-connection streaming quantizer engine.
pub struct HfParallelStreamer {
    pub config: HfStreamerConfig,
    client: reqwest::Client,
    hf_client: HfClient,
}

impl HfParallelStreamer {
    pub fn new(config: HfStreamerConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://huggingface.co".to_string());
        let hf_client = HfClient::with_endpoint(config.auth_token.clone(), base_url);
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            config,
            client,
            hf_client,
        }
    }

    /// Fetches the initial SafeTensors header for a remote file via HTTP Range.
    async fn fetch_header_and_size(
        &self,
        file: &HfFileEntry,
    ) -> Result<(u64, Vec<u8>, u64)> {
        // Request first 64 KiB which covers 99.9% of SafeTensors headers
        let prefetch_size = 65_536u64;
        let range_hdr = format!("bytes=0-{}", prefetch_size - 1);

        let mut req = self.client.get(&file.url).header(RANGE, &range_hdr);
        if let Some(ref token) = self.config.auth_token {
            req = req.bearer_auth(token);
        }

        let resp = req.send().await.map_err(|e| LlmZipError::HfError {
            repo_id: self.config.repo_id.clone(),
            reason: format!("Failed to fetch SafeTensors header for {}: {}", file.filename, e),
        })?;

        let status = resp.status();
        if status != reqwest::StatusCode::PARTIAL_CONTENT && status != reqwest::StatusCode::OK {
            return Err(LlmZipError::HfError {
                repo_id: self.config.repo_id.clone(),
                reason: format!("Upstream HTTP {} fetching header for {}", status, file.filename),
            });
        }

        // Parse total file size from Content-Range or file metadata
        let mut total_file_size = file.size.unwrap_or(0);
        if let Some(cr) = resp.headers().get(CONTENT_RANGE).and_then(|v| v.to_str().ok()) {
            if let Some(slash_idx) = cr.rfind('/') {
                if let Ok(total) = cr[slash_idx + 1..].trim().parse::<u64>() {
                    total_file_size = total;
                }
            }
        }

        let initial_bytes = resp.bytes().await.map_err(|e| LlmZipError::HfError {
            repo_id: self.config.repo_id.clone(),
            reason: format!("Failed reading header response body: {}", e),
        })?;

        if initial_bytes.len() < 8 {
            return Err(LlmZipError::HfError {
                repo_id: self.config.repo_id.clone(),
                reason: format!("File {} is smaller than 8 bytes (invalid SafeTensors)", file.filename),
            });
        }

        if total_file_size == 0 {
            total_file_size = initial_bytes.len() as u64;
        }

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&initial_bytes[0..8]);
        let header_len = parse_and_validate_header_len(&file.filename, len_bytes, total_file_size)?;
        let required_bytes = 8 + header_len as usize;

        let mut full_header_buf = initial_bytes.to_vec();

        // If the 64 KiB buffer was not enough, fetch the remainder of the header
        if full_header_buf.len() < required_bytes {
            let next_start = full_header_buf.len() as u64;
            let next_end = required_bytes as u64 - 1;
            let rem_range = format!("bytes={}-{}", next_start, next_end);

            let mut rem_req = self.client.get(&file.url).header(RANGE, &rem_range);
            if let Some(ref token) = self.config.auth_token {
                rem_req = rem_req.bearer_auth(token);
            }
            let rem_resp = rem_req.send().await.map_err(|e| LlmZipError::HfError {
                repo_id: self.config.repo_id.clone(),
                reason: format!("Failed fetching header continuation for {}: {}", file.filename, e),
            })?;
            let rem_bytes = rem_resp.bytes().await.map_err(|e| LlmZipError::HfError {
                repo_id: self.config.repo_id.clone(),
                reason: format!("Failed reading header continuation: {}", e),
            })?;
            full_header_buf.extend_from_slice(&rem_bytes);
        }

        if full_header_buf.len() < required_bytes {
            return Err(LlmZipError::HfError {
                repo_id: self.config.repo_id.clone(),
                reason: format!("SafeTensors header truncated in {}", file.filename),
            });
        }

        let json_bytes = full_header_buf[8..required_bytes].to_vec();

        Ok((header_len, json_bytes, total_file_size))
    }

    /// Run the full multi-connection streaming download and quantization pipeline.
    pub async fn stream_to_gguf(&self) -> Result<PathBuf> {
        let descriptor = self
            .hf_client
            .get_model_info(&self.config.repo_id, self.config.revision.as_deref())
            .await?;

        // 1. Fetch preflight headers for each shard
        let mut shards_with_headers = Vec::new();
        for (idx, file) in descriptor.files.iter().enumerate() {
            let (header_len, header_json, total_file_size) =
                self.fetch_header_and_size(file).await?;

            let shard_desc = ShardDescriptor {
                shard_idx: idx,
                shard_name: file.filename.clone(),
                total_file_size,
                header_len,
                expected_sha256_hex: self.config.expected_sha256.clone().unwrap_or_default(),
            };

            shards_with_headers.push((shard_desc, header_json));
        }

        // 2. Build Virtual Tensor Address Space (VTAS)
        let vtas = build_virtual_tensor_space(shards_with_headers.clone(), false, 0)?;

        // 3. Build Deterministic Layout Plan
        let plan = build_deterministic_layout(
            &vtas,
            self.config.preset,
            "llama",
        )?;

        // 4. Initialize Streaming Engine
        let mut engine = StreamingQuantizeEngine::new_or_resume(
            vtas,
            plan,
            &self.config.output_path,
            EngineConfig::default(),
        )?;

        // 5. Download and quantize each shard via parallel range chunks
        for (shard_idx, file) in descriptor.files.iter().enumerate() {
            let shard_desc = &shards_with_headers[shard_idx].0;
            let chunks = generate_chunks(
                &shard_desc.shard_name,
                &file.url,
                shard_desc.total_file_size,
                self.config.chunk_size,
            );

            let scheduler = Arc::new(ParallelRangeScheduler::new(
                self.client.clone(),
                self.config.concurrency,
            ));
            let mut rx = scheduler.stream_ordered(chunks);

            while let Some(res) = rx.recv().await {
                let (_chunk_idx, chunk_bytes) = res?;
                let http_meta = HttpResponseMeta {
                    status_code: 206,
                    content_range_start: Some(engine.resume_cursor().1),
                };
                let chunk_len = chunk_bytes.len();
                engine.stream_shard_slice(
                    shard_idx,
                    http_meta,
                    &chunk_bytes,
                    chunk_len,
                    None,
                )?;
            }
        }

        // 6. Final verification & magic unlock
        engine.commit_if_completed()?;

        Ok(self.config.output_path.clone())
    }
}

