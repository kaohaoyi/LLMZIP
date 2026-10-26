use crate::error::{LlmZipError, Result};
use crate::layout::{DeterministicLayoutPlan, GGUF_MAGIC_UNVERIFIED, GGUF_MAGIC_VERIFIED};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Manages the pre-allocated output file, exclusive concurrency lock [E10],
/// random-offset `pwrite` operations, and the cryptographic Magic Commit Lock (`UNVF` -> `GGUF`) [E03].
pub struct PreallocatedOutputFile {
    file: File,
    output_path: PathBuf,
    lock_path: PathBuf,
    total_size: u64,
    is_committed: bool,
}

impl PreallocatedOutputFile {
    /// Creates or opens a pre-allocated output file, acquiring an exclusive process lock [E10]
    /// and pre-allocating the exact final byte size on disk [E08].
    pub fn create_or_resume(output_path: &Path, plan: &DeterministicLayoutPlan, resume: bool) -> Result<Self> {
        let mut lock_os = output_path.as_os_str().to_os_string();
        lock_os.push(".lock");
        let lock_path = PathBuf::from(lock_os);

        // Acquire atomic exclusive lock file [E10].
        match OpenOptions::new().write(true).create_new(true).open(&lock_path) {
            Ok(mut lock_file) => {
                let pid = std::process::id();
                let _ = writeln!(lock_file, "pid={}", pid);
                let _ = lock_file.sync_all();
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(LlmZipError::FileLocked {
                    path: output_path.display().to_string(),
                });
            }
            Err(e) => return Err(LlmZipError::Io(e)),
        }

        let file_res = (|| -> Result<File> {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(!resume)
                .open(output_path)?;

            // Pre-allocate exact target byte size on disk immediately [E08].
            file.set_len(plan.total_output_file_size)?;

            if !resume {
                // Write the unverified GGUF header (starting with "UNVF" magic [E03]).
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&plan.unverified_header_bytes)?;

                // Force physical tail allocation check to catch ENOSPC upfront [E08].
                if plan.total_output_file_size > plan.unverified_header_bytes.len() as u64 {
                    file.seek(SeekFrom::Start(plan.total_output_file_size - 1))?;
                    file.write_all(&[0u8])?;
                }
                file.sync_all()?;
            } else {
                // Verify the existing file starts with "UNVF" or "GGUF"
                let mut magic = [0u8; 4];
                file.seek(SeekFrom::Start(0))?;
                file.read_exact(&mut magic)?;
                if magic != GGUF_MAGIC_UNVERIFIED && magic != GGUF_MAGIC_VERIFIED {
                    return Err(LlmZipError::MetadataParse(
                        "Resume target file does not have a valid UNVF/GGUF header".to_string(),
                    ));
                }
            }
            Ok(file)
        })();

        match file_res {
            Ok(file) => Ok(Self {
                file,
                output_path: output_path.to_path_buf(),
                lock_path,
                total_size: plan.total_output_file_size,
                is_committed: false,
            }),
            Err(err) => {
                let _ = fs::remove_file(&lock_path);
                Err(err)
            }
        }
    }

    /// Writes `buf` at exact byte `offset` in the pre-allocated file without creating any temporary files.
    pub fn pwrite_all(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        let end = offset.checked_add(buf.len() as u64).ok_or_else(|| {
            LlmZipError::InvalidTensorBounds {
                shard_name: "<output>".to_string(),
                tensor_name: "<write>".to_string(),
                reason: "Write offset overflow".to_string(),
            }
        })?;
        if end > self.total_size {
            return Err(LlmZipError::InvalidTensorBounds {
                shard_name: "<output>".to_string(),
                tensor_name: "<write>".to_string(),
                reason: format!(
                    "Attempted pwrite [{}..{}] beyond pre-allocated file size {}",
                    offset, end, self.total_size
                ),
            });
        }
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(buf)?;
        Ok(())
    }

    /// Reads `buf.len()` bytes from exact byte `offset` in the output file (used by AWQ crash-resume fast re-forward).
    pub fn pread_exact(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(buf)?;
        Ok(())
    }

    /// Flushes data to disk before updating `.wal`.
    pub fn sync_data(&mut self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }

    /// Atomically flips the file header magic from `"UNVF"` to `"GGUF"` after all shard SHA-256 hashes pass [E03].
    pub fn commit_verified(&mut self) -> Result<()> {
        self.file.sync_all()?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&GGUF_MAGIC_VERIFIED)?;
        self.file.sync_all()?;
        self.is_committed = true;
        Ok(())
    }

    /// Quarantine / poison the file on security violation (e.g. SHA-256 mismatch [E03]) by zeroing the header and truncating.
    pub fn quarantine_and_abort(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(b"POISONED")?;
        self.file.set_len(8)?;
        self.file.sync_all()?;
        self.is_committed = false;
        Ok(())
    }

    pub fn is_committed(&self) -> bool {
        self.is_committed
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }
}

impl Drop for PreallocatedOutputFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}
