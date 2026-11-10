use std::fmt;
use std::io;

/// Comprehensive error and security violation taxonomy covering E01 through E12.
#[derive(Debug)]
pub enum LlmZipError {
    /// [E01] Malicious oversized SafeTensors JSON header (> 100 MB or 0 bytes).
    HeaderTooLarge {
        shard_name: String,
        header_len: u64,
        max_allowed: u64,
    },
    /// [E02] Tensor offset overlap, inversion (`start > end`), out-of-bounds, or shape integer overflow.
    InvalidTensorBounds {
        shard_name: String,
        tensor_name: String,
        reason: String,
    },
    /// [E03] Cryptographic SHA-256 mismatch at shard EOF (MITM tampering or corrupted upstream).
    Sha256Mismatch {
        shard_name: String,
        expected_hex: String,
        actual_hex: String,
    },
    /// [E04] Numerical poisoning (`NaN` or `Inf`) detected in upstream weight tensor when sanitization is disabled.
    NumericalPoisoning {
        tensor_name: String,
        element_index: usize,
        raw_bits: u16,
    },
    /// [E05] Invalid tensor shape or zero-dimension tensor.
    InvalidShape {
        tensor_name: String,
        shape: Vec<usize>,
    },
    /// [E08 / IO] Underlying filesystem or pre-allocation error (e.g., ENOSPC).
    Io(io::Error),
    /// [E09] Upstream HTTP Range protocol violation (e.g., mismatched 206 Content-Range).
    HttpRangeViolation {
        shard_name: String,
        expected_start: u64,
        actual_start: u64,
    },
    /// [E10] Concurrent process lock conflict on the target `.unverified` or `.wal` file.
    FileLocked {
        path: String,
    },
    /// [E12] Path traversal or forbidden format (e.g., `.bin` pickle file or `../` in shard filename).
    SecurityPathViolation {
        offending_path: String,
        reason: String,
    },
    /// WAL corruption or CRC32 mismatch on crash recovery.
    WalCorrupted {
        reason: String,
    },
    /// JSON parsing or metadata error.
    MetadataParse(String),
    /// S3 or Cloud Object Storage error (e.g. upload part failure, bucket not found).
    S3Error {
        bucket: String,
        key: String,
        reason: String,
    },
    /// Cloud authentication expired or forbidden (403 Forbidden).
    AuthExpired {
        reason: String,
    },
    /// Dequantization or SafeTensors inverse export error.
    DequantError {
        tensor_name: String,
        reason: String,
    },
    /// Hugging Face Hub API or repository error.
    HfError {
        repo_id: String,
        reason: String,
    },
    /// Inference target plugin error (Ollama, LM Studio, Jan, SGLang, Strata).
    PluginError {
        target: String,
        reason: String,
    },
    /// Named Pipe / Unix FIFO IPC streaming error.
    PipeError {
        pipe_path: String,
        reason: String,
    },
    /// Flat P2P Swarm distribution error (e.g. peer timeout, protocol mismatch, poison chunk).
    SwarmError {
        peer: String,
        reason: String,
    },
    /// Cluster streaming direct-write error (vLLM / SGLang IPC, shared memory).
    ClusterError {
        target: String,
        reason: String,
    },
}

impl fmt::Display for LlmZipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderTooLarge {
                shard_name,
                header_len,
                max_allowed,
            } => write!(
                f,
                "[E01 Security] Shard '{}' header size {} bytes exceeds limit {} bytes",
                shard_name, header_len, max_allowed
            ),
            Self::InvalidTensorBounds {
                shard_name,
                tensor_name,
                reason,
            } => write!(
                f,
                "[E02 Security] Invalid tensor bounds for '{}' in shard '{}': {}",
                tensor_name, shard_name, reason
            ),
            Self::Sha256Mismatch {
                shard_name,
                expected_hex,
                actual_hex,
            } => write!(
                f,
                "[E03 Security] Cryptographic SHA-256 mismatch on shard '{}': expected {}, got {}",
                shard_name, expected_hex, actual_hex
            ),
            Self::NumericalPoisoning {
                tensor_name,
                element_index,
                raw_bits,
            } => write!(
                f,
                "[E04 Security] Numerical poisoning (NaN/Inf, raw=0x{:04X}) in tensor '{}' at index {}",
                raw_bits, tensor_name, element_index
            ),
            Self::InvalidShape { tensor_name, shape } => write!(
                f,
                "[E05 Layout] Invalid tensor shape for '{}': {:?}",
                tensor_name, shape
            ),
            Self::Io(err) => write!(f, "[E08 I/O] {}", err),
            Self::HttpRangeViolation {
                shard_name,
                expected_start,
                actual_start,
            } => write!(
                f,
                "[E09 Network] HTTP Range violation on '{}': expected byte offset {}, got {}",
                shard_name, expected_start, actual_start
            ),
            Self::FileLocked { path } => write!(
                f,
                "[E10 Concurrency] Target file '{}' is already locked by another LLMZIP process",
                path
            ),
            Self::SecurityPathViolation {
                offending_path,
                reason,
            } => write!(
                f,
                "[E12 Security] Forbidden path or format '{}': {}",
                offending_path, reason
            ),
            Self::WalCorrupted { reason } => write!(f, "[WAL Recovery] {}", reason),
            Self::MetadataParse(msg) => write!(f, "[Metadata] {}", msg),
            Self::S3Error { bucket, key, reason } => write!(f, "[S3 Error] s3://{}/{}: {}", bucket, key, reason),
            Self::AuthExpired { reason } => write!(f, "[Auth Error] Credentials expired or unauthorized: {}", reason),
            Self::DequantError { tensor_name, reason } => write!(f, "[Dequant Error] Tensor '{}': {}", tensor_name, reason),
            Self::HfError { repo_id, reason } => write!(f, "[HF Error] Repo '{}': {}", repo_id, reason),
            Self::PluginError { target, reason } => write!(f, "[Plugin Error] Target '{}': {}", target, reason),
            Self::PipeError { pipe_path, reason } => write!(f, "[Pipe Error] Pipe '{}': {}", pipe_path, reason),
            Self::SwarmError { peer, reason } => write!(f, "[Swarm Error] Peer '{}': {}", peer, reason),
            Self::ClusterError { target, reason } => write!(f, "[Cluster Error] Target '{}': {}", target, reason),
        }
    }
}

impl std::error::Error for LlmZipError {}

impl From<io::Error> for LlmZipError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, LlmZipError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v030_error_display() {
        let hf_err = LlmZipError::HfError {
            repo_id: "Qwen/Qwen2.5-7B".to_string(),
            reason: "401 Unauthorized".to_string(),
        };
        assert!(hf_err.to_string().contains("[HF Error] Repo 'Qwen/Qwen2.5-7B'"));

        let plugin_err = LlmZipError::PluginError {
            target: "ollama".to_string(),
            reason: "directory not found".to_string(),
        };
        assert!(plugin_err.to_string().contains("[Plugin Error] Target 'ollama'"));

        let pipe_err = LlmZipError::PipeError {
            pipe_path: "pipe://test".to_string(),
            reason: "broken pipe".to_string(),
        };
        assert!(pipe_err.to_string().contains("[Pipe Error] Pipe 'pipe://test'"));
    }

    #[test]
    fn test_v040_error_display() {
        let swarm_err = LlmZipError::SwarmError {
            peer: "127.0.0.1:6881".to_string(),
            reason: "poison chunk rejected".to_string(),
        };
        assert!(swarm_err.to_string().contains("[Swarm Error] Peer '127.0.0.1:6881'"));

        let cluster_err = LlmZipError::ClusterError {
            target: "vllm".to_string(),
            reason: "shm buffer overflow".to_string(),
        };
        assert!(cluster_err.to_string().contains("[Cluster Error] Target 'vllm'"));
    }
}
