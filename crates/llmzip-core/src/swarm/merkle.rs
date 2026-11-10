use bytes::Bytes;
use ring::digest::{digest, Context, SHA256};
use serde::{Deserialize, Serialize};
use crate::error::LlmZipError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkMeta {
    pub chunk_id: u32,
    pub size: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub model_id: String,
    pub total_size: u64,
    pub chunk_size: usize,
    pub chunks: Vec<ChunkMeta>,
    pub merkle_root: String,
}

pub struct Chunker;

impl Chunker {
    /// Splits raw byte data into chunks and computes Merkle root and chunk hashes
    pub fn split(model_id: &str, data: &[u8], chunk_size: usize) -> (ModelManifest, Vec<Bytes>) {
        let chunk_size = if chunk_size == 0 { 1024 * 1024 * 8 } else { chunk_size };
        let mut chunks_meta = Vec::new();
        let mut chunks_data = Vec::new();
        let mut root_hasher = Context::new(&SHA256);

        let mut offset = 0;
        let mut chunk_id = 0;
        let total_size = data.len() as u64;

        while offset < data.len() {
            let end = std::cmp::min(offset + chunk_size, data.len());
            let slice = &data[offset..end];
            let chunk_digest = digest(&SHA256, slice);
            let sha256_hex = hex_encode(chunk_digest.as_ref());

            root_hasher.update(chunk_digest.as_ref());

            chunks_meta.push(ChunkMeta {
                chunk_id,
                size: slice.len(),
                sha256: sha256_hex,
            });
            chunks_data.push(Bytes::copy_from_slice(slice));

            offset = end;
            chunk_id += 1;
        }

        let root_digest = root_hasher.finish();
        let merkle_root = hex_encode(root_digest.as_ref());

        let manifest = ModelManifest {
            model_id: model_id.to_string(),
            total_size,
            chunk_size,
            chunks: chunks_meta,
            merkle_root,
        };

        (manifest, chunks_data)
    }

    /// In-flight Poison Defense: verifies chunk integrity against expected hash
    pub fn verify_chunk(chunk: &[u8], meta: &ChunkMeta) -> Result<(), LlmZipError> {
        let actual_hash = hex_encode(digest(&SHA256, chunk).as_ref());
        if actual_hash != meta.sha256 {
            return Err(LlmZipError::SwarmError {
                peer: format!("chunk_{}", meta.chunk_id),
                reason: format!(
                    "In-flight poison defense: checksum mismatch for chunk {} (expected {}, got {})",
                    meta.chunk_id, meta.sha256, actual_hash
                ),
            });
        }
        Ok(())
    }
}

pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_merkle_split_and_poison_detection() {
        let dummy_data = vec![0xAB; 1024 * 1024 * 2 + 512]; // 2MB + 512B
        let (manifest, chunks) = Chunker::split("test-model-7b", &dummy_data, 1024 * 1024);
        assert_eq!(chunks.len(), 3);
        assert_eq!(manifest.chunks.len(), 3);
        assert!(!manifest.merkle_root.is_empty());

        // Verify valid chunk
        assert!(Chunker::verify_chunk(&chunks[0], &manifest.chunks[0]).is_ok());

        // Test poison tampering
        let mut tampered = chunks[0].to_vec();
        tampered[0] ^= 0xFF;
        let res = Chunker::verify_chunk(&tampered, &manifest.chunks[0]);
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("In-flight poison defense"));
    }
}
