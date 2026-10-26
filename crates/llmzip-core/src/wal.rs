use crate::error::{LlmZipError, Result};
use crate::hasher::{crc32_ieee, Sha256MidState};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const WAL_MAGIC: [u8; 8] = *b"LZIPWAL1";
pub const WAL_VERSION: u32 = 1;
/// Fixed header size before the bitset: 8 (magic) + 4 (version) + 4 (shard_idx) + 8 (offset) + 112 (sha256_mid) + 4 (total_tensors) + 4 (crc32) = 144 bytes.
pub const WAL_FIXED_HEADER_BYTES: usize = 144;

/// `< 4KB` Micro-Journal (`.wal`) state for Zero-Re-Download atomic crash recovery (Question 5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroWal {
    pub current_shard_idx: u32,
    pub shard_confirmed_offset: u64,
    pub sha256_midstate: Sha256MidState,
    pub total_tensors: u32,
    pub completed_bitset: Vec<u64>,
}

impl MicroWal {
    pub fn new(total_tensors: usize, initial_midstate: Sha256MidState) -> Self {
        let num_words = total_tensors.div_ceil(64);
        Self {
            current_shard_idx: 0,
            shard_confirmed_offset: 0,
            sha256_midstate: initial_midstate,
            total_tensors: total_tensors as u32,
            completed_bitset: vec![0u64; num_words],
        }
    }

    pub fn mark_tensor_completed(&mut self, tensor_id: usize) {
        let word_idx = tensor_id / 64;
        let bit_idx = tensor_id % 64;
        if let Some(word) = self.completed_bitset.get_mut(word_idx) {
            *word |= 1u64 << bit_idx;
        }
    }

    pub fn is_tensor_completed(&self, tensor_id: usize) -> bool {
        let word_idx = tensor_id / 64;
        let bit_idx = tensor_id % 64;
        self.completed_bitset
            .get(word_idx)
            .map(|&w| (w & (1u64 << bit_idx)) != 0)
            .unwrap_or(false)
    }

    pub fn completed_count(&self) -> usize {
        self.completed_bitset.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn all_completed(&self) -> bool {
        (0..self.total_tensors as usize).all(|id| self.is_tensor_completed(id))
    }

    /// Serializes the WAL state into a compact binary record (`< 4KB` for up to 30,000 tensors) with IEEE CRC32 protection.
    pub fn to_bytes(&self) -> Vec<u8> {
        let bitset_byte_len = self.completed_bitset.len() * 8;
        let mut buf = Vec::with_capacity(WAL_FIXED_HEADER_BYTES + bitset_byte_len);

        buf.extend_from_slice(&WAL_MAGIC);
        buf.extend_from_slice(&WAL_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.current_shard_idx.to_le_bytes());
        buf.extend_from_slice(&self.shard_confirmed_offset.to_le_bytes());
        buf.extend_from_slice(&self.sha256_midstate.to_bytes());
        buf.extend_from_slice(&self.total_tensors.to_le_bytes());

        // Placeholder for CRC32 at offset 140..144
        buf.extend_from_slice(&0u32.to_le_bytes());

        for &word in &self.completed_bitset {
            buf.extend_from_slice(&word.to_le_bytes());
        }

        // Compute CRC32 over bytes 0..140 and 144..end
        let mut crc_input = Vec::with_capacity(buf.len() - 4);
        crc_input.extend_from_slice(&buf[..140]);
        crc_input.extend_from_slice(&buf[144..]);
        let crc = crc32_ieee(&crc_input);
        buf[140..144].copy_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Deserializes and verifies the WAL binary record.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < WAL_FIXED_HEADER_BYTES {
            return Err(LlmZipError::WalCorrupted {
                reason: format!(
                    "WAL file too short: {} bytes (minimum {})",
                    bytes.len(),
                    WAL_FIXED_HEADER_BYTES
                ),
            });
        }
        if bytes[0..8] != WAL_MAGIC {
            return Err(LlmZipError::WalCorrupted {
                reason: "Invalid WAL magic header".to_string(),
            });
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != WAL_VERSION {
            return Err(LlmZipError::WalCorrupted {
                reason: format!("Unsupported WAL version {}", version),
            });
        }
        let current_shard_idx = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let shard_confirmed_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let mut mid_bytes = [0u8; Sha256MidState::BYTE_SIZE];
        mid_bytes.copy_from_slice(&bytes[24..136]);
        let sha256_midstate = Sha256MidState::from_bytes(&mid_bytes).ok_or_else(|| LlmZipError::WalCorrupted {
            reason: "Invalid SHA-256 tail_len in WAL mid-state".to_string(),
        })?;

        let total_tensors = u32::from_le_bytes(bytes[136..140].try_into().unwrap());
        let expected_crc = u32::from_le_bytes(bytes[140..144].try_into().unwrap());

        let expected_words = (total_tensors as usize).div_ceil(64);
        if bytes.len() != WAL_FIXED_HEADER_BYTES + expected_words * 8 {
            return Err(LlmZipError::WalCorrupted {
                reason: "WAL bitset payload length mismatch".to_string(),
            });
        }

        let mut crc_input = Vec::with_capacity(bytes.len() - 4);
        crc_input.extend_from_slice(&bytes[..140]);
        crc_input.extend_from_slice(&bytes[144..]);
        let actual_crc = crc32_ieee(&crc_input);
        if actual_crc != expected_crc {
            return Err(LlmZipError::WalCorrupted {
                reason: format!(
                    "WAL CRC32 mismatch: expected 0x{:08X}, got 0x{:08X}",
                    expected_crc, actual_crc
                ),
            });
        }

        let mut completed_bitset = Vec::with_capacity(expected_words);
        for i in 0..expected_words {
            let start = WAL_FIXED_HEADER_BYTES + i * 8;
            let w = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
            completed_bitset.push(w);
        }

        Ok(Self {
            current_shard_idx,
            shard_confirmed_offset,
            sha256_midstate,
            total_tensors,
            completed_bitset,
        })
    }

    /// Atomically writes and `fsync`s the WAL state to an open file handle.
    pub fn sync_to_file(&self, file: &mut File) -> Result<()> {
        let data = self.to_bytes();
        file.seek(SeekFrom::Start(0))?;
        file.set_len(data.len() as u64)?;
        file.write_all(&data)?;
        file.sync_data()?;
        Ok(())
    }

    /// Loads and verifies a `.wal` file from disk.
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let mut f = OpenOptions::new().read(true).open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }
}
