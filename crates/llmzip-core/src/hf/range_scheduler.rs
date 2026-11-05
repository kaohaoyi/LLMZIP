use crate::error::{LlmZipError, Result};
use bytes::Bytes;
use reqwest::header::RANGE;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

/// Default chunk size for parallel range downloading: 8 MB.
pub const DEFAULT_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

/// A discrete, non-overlapping HTTP byte range chunk request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRangeRequest {
    pub chunk_idx: usize,
    pub shard_name: String,
    pub file_url: String,
    pub byte_start: u64,
    pub byte_end: u64, // Inclusive
}

impl ChunkRangeRequest {
    pub fn len(&self) -> u64 {
        if self.byte_end >= self.byte_start {
            self.byte_end - self.byte_start + 1
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Partition a file of known size into sequential, non-overlapping `ChunkRangeRequest` slices.
pub fn generate_chunks(
    shard_name: &str,
    file_url: &str,
    file_size: u64,
    chunk_size: u64,
) -> Vec<ChunkRangeRequest> {
    if file_size == 0 {
        return Vec::new();
    }
    let actual_chunk_size = if chunk_size == 0 {
        DEFAULT_CHUNK_SIZE
    } else {
        chunk_size
    };

    let mut chunks = Vec::new();
    let mut offset = 0u64;
    let mut idx = 0usize;

    while offset < file_size {
        let end = (offset + actual_chunk_size - 1).min(file_size - 1);
        chunks.push(ChunkRangeRequest {
            chunk_idx: idx,
            shard_name: shard_name.to_string(),
            file_url: file_url.to_string(),
            byte_start: offset,
            byte_end: end,
        });
        offset = end + 1;
        idx += 1;
    }

    chunks
}

/// In-memory reassembly buffer that guarantees sequential ordered consumption
/// despite out-of-order network arrival from parallel download workers.
pub struct OrderedChunkReassembler {
    next_expected_chunk: usize,
    buffer: BTreeMap<usize, Bytes>,
}

impl OrderedChunkReassembler {
    pub fn new() -> Self {
        Self {
            next_expected_chunk: 0,
            buffer: BTreeMap::new(),
        }
    }

    /// Insert an arrived chunk into the reassembly buffer.
    pub fn push(&mut self, chunk_idx: usize, data: Bytes) {
        self.buffer.insert(chunk_idx, data);
    }

    /// Retrieve the next chunk if and only if it matches `next_expected_chunk`.
    pub fn pop_next(&mut self) -> Option<(usize, Bytes)> {
        if let Some(data) = self.buffer.remove(&self.next_expected_chunk) {
            let idx = self.next_expected_chunk;
            self.next_expected_chunk += 1;
            Some((idx, data))
        } else {
            None
        }
    }

    pub fn next_expected(&self) -> usize {
        self.next_expected_chunk
    }

    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_complete(&self, total_chunks: usize) -> bool {
        self.next_expected_chunk >= total_chunks && self.buffer.is_empty()
    }
}

impl Default for OrderedChunkReassembler {
    fn default() -> Self {
        Self::new()
    }
}

/// Fetch a single chunk with up to 3 retries.
pub async fn fetch_chunk_with_retry(
    client: &reqwest::Client,
    chunk: &ChunkRangeRequest,
    max_retries: usize,
) -> Result<Bytes> {
    let mut last_error = String::new();
    let range_header = format!("bytes={}-{}", chunk.byte_start, chunk.byte_end);

    for attempt in 0..=max_retries {
        if attempt > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(50 * (1 << attempt))).await;
        }

        let resp = match client
            .get(&chunk.file_url)
            .header(RANGE, &range_header)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                last_error = format!("HTTP send error: {}", e);
                continue;
            }
        };

        let status = resp.status();
        if status != reqwest::StatusCode::PARTIAL_CONTENT && status != reqwest::StatusCode::OK {
            last_error = format!("HTTP status {} on range {}", status, range_header);
            continue;
        }

        match resp.bytes().await {
            Ok(bytes) => {
                let expected_len = chunk.len();
                if bytes.len() as u64 != expected_len {
                    last_error = format!(
                        "Length mismatch: expected {} bytes, got {} bytes",
                        expected_len,
                        bytes.len()
                    );
                    continue;
                }
                return Ok(bytes);
            }
            Err(e) => {
                last_error = format!("Failed reading response bytes: {}", e);
                continue;
            }
        }
    }

    Err(LlmZipError::HfError {
        repo_id: chunk.shard_name.clone(),
        reason: format!("Range fetch failed after {} retries: {}", max_retries, last_error),
    })
}

/// Asynchronous parallel range scheduler.
/// Dispatches chunks across `concurrency` parallel tasks and yields chunks
/// strictly ordered by `chunk_idx` to maintain streaming invariants.
pub struct ParallelRangeScheduler {
    client: reqwest::Client,
    concurrency: usize,
}

impl ParallelRangeScheduler {
    pub fn new(client: reqwest::Client, concurrency: usize) -> Self {
        Self {
            client,
            concurrency: concurrency.max(1),
        }
    }

    /// Stream all chunks in strict sequential order (0, 1, 2, ...).
    /// Uses bounded semaphore to enforce the bounded RAM ceiling (Principle II).
    pub fn stream_ordered(
        self: Arc<Self>,
        chunks: Vec<ChunkRangeRequest>,
    ) -> mpsc::Receiver<Result<(usize, Bytes)>> {
        let total_chunks = chunks.len();
        // Bounded channel to prevent unbounded memory growth if downstream consumer is slower
        let (out_tx, out_rx) = mpsc::channel(self.concurrency * 2);

        tokio::spawn(async move {
            let (raw_tx, mut raw_rx) = mpsc::channel::<(usize, Result<Bytes>)>(self.concurrency * 2);
            let sem = Arc::new(Semaphore::new(self.concurrency));

            // Spawn background task to launch downloads bounded by concurrency
            let client = self.client.clone();
            tokio::spawn(async move {
                for chunk in chunks {
                    let permit = match sem.clone().acquire_owned().await {
                        Ok(p) => p,
                        Err(_) => break,
                    };
                    let raw_tx = raw_tx.clone();
                    let client = client.clone();

                    tokio::spawn(async move {
                        let idx = chunk.chunk_idx;
                        let res = fetch_chunk_with_retry(&client, &chunk, 3).await;
                        let _ = raw_tx.send((idx, res)).await;
                        drop(permit);
                    });
                }
            });

            // Reassembler collects out-of-order chunks and flushes strictly ordered
            let mut reassembler = OrderedChunkReassembler::new();

            while !reassembler.is_complete(total_chunks) {
                // First drain whatever ordered chunks are ready
                while let Some((idx, data)) = reassembler.pop_next() {
                    if out_tx.send(Ok((idx, data))).await.is_err() {
                        return; // Downstream receiver dropped
                    }
                }

                if reassembler.is_complete(total_chunks) {
                    break;
                }

                // Wait for next arrived chunk from workers
                match raw_rx.recv().await {
                    Some((idx, Ok(data))) => {
                        reassembler.push(idx, data);
                    }
                    Some((_idx, Err(err))) => {
                        let _ = out_tx.send(Err(err)).await;
                        return;
                    }
                    None => {
                        // All worker channels closed; flush any remaining
                        while let Some((idx, data)) = reassembler.pop_next() {
                            let _ = out_tx.send(Ok((idx, data))).await;
                        }
                        break;
                    }
                }
            }
        });

        out_rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_chunks_boundary() {
        let file_url = "https://example.com/model.safetensors";
        let shard_name = "model.safetensors";

        // Case 1: 0 size
        let chunks = generate_chunks(shard_name, file_url, 0, 1024);
        assert!(chunks.is_empty());

        // Case 2: exact multiple (2048 bytes with 1024 chunk size)
        let chunks = generate_chunks(shard_name, file_url, 2048, 1024);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_idx, 0);
        assert_eq!(chunks[0].byte_start, 0);
        assert_eq!(chunks[0].byte_end, 1023);
        assert_eq!(chunks[0].len(), 1024);

        assert_eq!(chunks[1].chunk_idx, 1);
        assert_eq!(chunks[1].byte_start, 1024);
        assert_eq!(chunks[1].byte_end, 2047);
        assert_eq!(chunks[1].len(), 1024);

        // Case 3: uneven remainder (2500 bytes with 1000 chunk size)
        let chunks = generate_chunks(shard_name, file_url, 2500, 1000);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].byte_start, 2000);
        assert_eq!(chunks[2].byte_end, 2499);
        assert_eq!(chunks[2].len(), 500);

        // Case 4: file smaller than chunk size
        let chunks = generate_chunks(shard_name, file_url, 500, 1024);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].byte_start, 0);
        assert_eq!(chunks[0].byte_end, 499);
        assert_eq!(chunks[0].len(), 500);
    }

    #[test]
    fn test_ordered_chunk_reassembler_out_of_order() {
        let mut reassembler = OrderedChunkReassembler::new();

        // Push out of order: 2, 0, 3, 1
        reassembler.push(2, Bytes::from_static(b"chunk2"));
        assert!(reassembler.pop_next().is_none());

        reassembler.push(0, Bytes::from_static(b"chunk0"));
        let first = reassembler.pop_next();
        assert_eq!(first, Some((0, Bytes::from_static(b"chunk0"))));
        // Next expected is 1, but we only have 2
        assert!(reassembler.pop_next().is_none());

        reassembler.push(3, Bytes::from_static(b"chunk3"));
        assert!(reassembler.pop_next().is_none());

        // Now push 1 -> should yield 1, then 2, then 3!
        reassembler.push(1, Bytes::from_static(b"chunk1"));
        assert_eq!(reassembler.pop_next(), Some((1, Bytes::from_static(b"chunk1"))));
        assert_eq!(reassembler.pop_next(), Some((2, Bytes::from_static(b"chunk2"))));
        assert_eq!(reassembler.pop_next(), Some((3, Bytes::from_static(b"chunk3"))));
        assert!(reassembler.pop_next().is_none());
        assert!(reassembler.is_complete(4));
    }
}
