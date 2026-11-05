use crate::cloud::client::{build_s3_store, S3Uri};
use crate::cloud::credentials::CloudCredentials;
use crate::error::{LlmZipError, Result};
use bytes::Bytes;
use futures::executor::block_on;
use object_store::path::Path as StorePath;
use object_store::{MultipartUpload, ObjectStore};
use std::sync::Arc;

pub const DEFAULT_S3_PART_SIZE: usize = 16 * 1024 * 1024; // 16 MiB
pub const MIN_S3_PART_SIZE: usize = 5 * 1024 * 1024;      // 5 MiB (S3 minimum)
pub const MAX_S3_PART_SIZE: usize = 64 * 1024 * 1024;     // 64 MiB

/// Zero-Temp-File streaming direct writer to S3 / Cloudflare R2 / MinIO.
/// Keeps chunks strictly in RAM (bounded to `part_size`), then flushes directly via S3 Multipart Upload.
pub struct S3MultipartStreamer {
    location: StorePath,
    part_size: usize,
    upload: Option<Box<dyn MultipartUpload>>,
    buffer: Vec<u8>,
    total_written: u64,
    parts_count: u32,
    completed: bool,
}

impl S3MultipartStreamer {
    /// Creates a new S3 streaming uploader using an arbitrary `ObjectStore` backend (e.g., InMemory for testing or S3).
    pub fn from_store(
        store: Arc<dyn ObjectStore>,
        location: StorePath,
        custom_part_size: Option<usize>,
    ) -> Result<Self> {
        let part_size = custom_part_size
            .unwrap_or(DEFAULT_S3_PART_SIZE)
            .clamp(MIN_S3_PART_SIZE, MAX_S3_PART_SIZE);

        let upload = block_on(store.put_multipart(&location)).map_err(|e| LlmZipError::S3Error {
            bucket: String::new(),
            key: location.to_string(),
            reason: format!("Failed to initiate S3 Multipart Upload: {}", e),
        })?;

        Ok(Self {
            location,
            part_size,
            upload: Some(upload),
            buffer: Vec::with_capacity(part_size),
            total_written: 0,
            parts_count: 0,
            completed: false,
        })
    }

    /// Creates a new S3 streaming uploader for the specified `s3://` destination.
    pub fn new(uri: &str, creds: Option<&CloudCredentials>, custom_part_size: Option<usize>) -> Result<Self> {
        let parsed = S3Uri::parse(uri)?;
        let store = build_s3_store(&parsed, creds)?;
        let location = StorePath::from(parsed.key.as_str());
        Self::from_store(store, location, custom_part_size)
    }

    /// Appends incoming streaming bytes to the in-memory buffer.
    /// When the buffer reaches `part_size`, it is flushed directly to S3 as a Multipart chunk.
    pub fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        let mut offset = 0;
        while offset < chunk.len() {
            let space = self.part_size - self.buffer.len();
            let to_copy = space.min(chunk.len() - offset);
            self.buffer.extend_from_slice(&chunk[offset..offset + to_copy]);
            offset += to_copy;

            if self.buffer.len() >= self.part_size {
                self.flush_buffer()?;
            }
        }
        Ok(())
    }

    /// Flushes the current in-memory buffer to S3 as a new part.
    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let upload = self.upload.as_mut().ok_or_else(|| LlmZipError::S3Error {
            bucket: String::new(),
            key: self.location.to_string(),
            reason: "Multipart upload was already finalized or aborted".to_string(),
        })?;

        let bytes = Bytes::copy_from_slice(&self.buffer);
        let len = bytes.len() as u64;

        block_on(upload.put_part(bytes.into())).map_err(|e| LlmZipError::S3Error {
            bucket: String::new(),
            key: self.location.to_string(),
            reason: format!("UploadPart {} failed: {}", self.parts_count + 1, e),
        })?;

        self.parts_count += 1;
        self.total_written += len;
        self.buffer.clear();
        Ok(())
    }

    /// Flushes any remaining bytes in memory and completes the S3 Multipart Upload.
    pub fn complete(mut self) -> Result<u64> {
        if !self.buffer.is_empty() {
            self.flush_buffer()?;
        }

        let mut upload = self.upload.take().ok_or_else(|| LlmZipError::S3Error {
            bucket: String::new(),
            key: self.location.to_string(),
            reason: "Multipart upload already consumed".to_string(),
        })?;

        block_on(upload.complete()).map_err(|e| LlmZipError::S3Error {
            bucket: String::new(),
            key: self.location.to_string(),
            reason: format!("CompleteMultipartUpload failed: {}", e),
        })?;

        self.completed = true;
        Ok(self.total_written)
    }

    /// Explicitly aborts the in-flight S3 Multipart Upload to prevent orphaned storage charges.
    pub fn abort(&mut self) -> Result<()> {
        if let Some(mut upload) = self.upload.take() {
            block_on(upload.abort()).map_err(|e| LlmZipError::S3Error {
                bucket: String::new(),
                key: self.location.to_string(),
                reason: format!("AbortMultipartUpload failed: {}", e),
            })?;
        }
        self.completed = true;
        Ok(())
    }

    /// Returns the number of parts flushed so far.
    pub fn parts_count(&self) -> u32 {
        self.parts_count
    }

    /// Returns the total bytes flushed so far.
    pub fn total_written(&self) -> u64 {
        self.total_written
    }
}

impl Drop for S3MultipartStreamer {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part_size_clamping() {
        assert_eq!(
            16 * 1024 * 1024,
            DEFAULT_S3_PART_SIZE.clamp(MIN_S3_PART_SIZE, MAX_S3_PART_SIZE)
        );
        // Too small clamped to 5MB
        assert_eq!(
            MIN_S3_PART_SIZE,
            (1024 * 1024).clamp(MIN_S3_PART_SIZE, MAX_S3_PART_SIZE)
        );
        // Too large clamped to 64MB
        assert_eq!(
            MAX_S3_PART_SIZE,
            (128 * 1024 * 1024).clamp(MIN_S3_PART_SIZE, MAX_S3_PART_SIZE)
        );
    }
}
