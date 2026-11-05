use crate::error::{LlmZipError, Result};
#[cfg(not(windows))]
use std::path::PathBuf;

/// Converts a `pipe://<name>` URI into the platform-specific pipe path.
/// - Windows: `\\.\pipe\<name>`
/// - Unix: `/tmp/<name>`
pub fn parse_pipe_uri(uri: &str) -> String {
    let clean = uri.trim();
    if clean.starts_with("pipe://") {
        let name = &clean["pipe://".len()..];
        #[cfg(windows)]
        {
            format!(r"\\.\pipe\{}", name)
        }
        #[cfg(not(windows))]
        {
            format!("/tmp/{}", name)
        }
    } else {
        clean.to_string()
    }
}

/// Cross-platform Named Pipe (Windows) and Unix FIFO (Linux/macOS) abstraction.
pub struct LlmZipPipe {
    pub path: String,
    #[cfg(windows)]
    server: Option<tokio::net::windows::named_pipe::NamedPipeServer>,
    #[cfg(not(windows))]
    fifo_path: Option<PathBuf>,
    #[cfg(not(windows))]
    file: Option<tokio::fs::File>,
}

impl LlmZipPipe {
    /// Creates a new named pipe server at the specified URI or platform path.
    pub fn create(pipe_uri: &str) -> Result<Self> {
        let path = parse_pipe_uri(pipe_uri);

        #[cfg(windows)]
        {
            let server = tokio::net::windows::named_pipe::ServerOptions::new()
                .first_pipe_instance(true)
                .create(&path)
                .map_err(|e| LlmZipError::PipeError {
                    pipe_path: path.clone(),
                    reason: format!("Failed creating Windows Named Pipe: {}", e),
                })?;

            Ok(Self {
                path,
                server: Some(server),
            })
        }

        #[cfg(not(windows))]
        {
            let fifo_buf = PathBuf::from(&path);
            if fifo_buf.exists() {
                let _ = std::fs::remove_file(&fifo_buf);
            }
            // Create Unix FIFO using mkfifo command or C ABI
            let status = std::process::Command::new("mkfifo")
                .arg(&path)
                .status()
                .map_err(|e| LlmZipError::PipeError {
                    pipe_path: path.clone(),
                    reason: format!("Failed executing mkfifo: {}", e),
                })?;

            if !status.success() {
                return Err(LlmZipError::PipeError {
                    pipe_path: path.clone(),
                    reason: "mkfifo command failed".to_string(),
                });
            }

            Ok(Self {
                path,
                fifo_path: Some(fifo_buf),
                file: None,
            })
        }
    }

    /// Asynchronously waits for a client (e.g. `llama-cli`) to connect to the pipe.
    pub async fn wait_for_reader(&mut self) -> Result<()> {
        #[cfg(windows)]
        {
            if let Some(ref mut server) = self.server {
                server.connect().await.map_err(|e| LlmZipError::PipeError {
                    pipe_path: self.path.clone(),
                    reason: format!("Client connection failed: {}", e),
                })?;
            }
            Ok(())
        }

        #[cfg(not(windows))]
        {
            let f = tokio::fs::OpenOptions::new()
                .write(true)
                .open(&self.path)
                .await
                .map_err(|e| LlmZipError::PipeError {
                    pipe_path: self.path.clone(),
                    reason: format!("Failed opening FIFO for writing: {}", e),
                })?;
            self.file = Some(f);
            Ok(())
        }
    }

    /// Asynchronously writes bytes into the pipe, catching BrokenPipe on reader disconnect.
    pub async fn write_all(&mut self, data: &[u8]) -> Result<()> {
        #[cfg(windows)]
        {
            use tokio::io::AsyncWriteExt;
            if let Some(ref mut server) = self.server {
                server.write_all(data).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::BrokenPipe {
                        LlmZipError::PipeError {
                            pipe_path: self.path.clone(),
                            reason: "Client disconnected unexpectedly (Broken pipe)".to_string(),
                        }
                    } else {
                        LlmZipError::PipeError {
                            pipe_path: self.path.clone(),
                            reason: format!("Write to pipe failed: {}", e),
                        }
                    }
                })?;
            }
            Ok(())
        }

        #[cfg(not(windows))]
        {
            use tokio::io::AsyncWriteExt;
            if let Some(ref mut f) = self.file {
                f.write_all(data).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::BrokenPipe {
                        LlmZipError::PipeError {
                            pipe_path: self.path.clone(),
                            reason: "Client disconnected unexpectedly (Broken pipe)".to_string(),
                        }
                    } else {
                        LlmZipError::PipeError {
                            pipe_path: self.path.clone(),
                            reason: format!("Write to FIFO failed: {}", e),
                        }
                    }
                })?;
            }
            Ok(())
        }
    }

    /// Flushes any buffered bytes in the pipe.
    pub async fn flush(&mut self) -> Result<()> {
        #[cfg(windows)]
        {
            use tokio::io::AsyncWriteExt;
            if let Some(ref mut server) = self.server {
                server.flush().await.map_err(|e| LlmZipError::PipeError {
                    pipe_path: self.path.clone(),
                    reason: format!("Pipe flush failed: {}", e),
                })?;
            }
            Ok(())
        }

        #[cfg(not(windows))]
        {
            use tokio::io::AsyncWriteExt;
            if let Some(ref mut f) = self.file {
                f.flush().await.map_err(|e| LlmZipError::PipeError {
                    pipe_path: self.path.clone(),
                    reason: format!("FIFO flush failed: {}", e),
                })?;
            }
            Ok(())
        }
    }
}

impl Drop for LlmZipPipe {
    fn drop(&mut self) {
        #[cfg(not(windows))]
        {
            if let Some(ref p) = self.fifo_path {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pipe_uri() {
        let uri = "pipe://llama_model";
        let parsed = parse_pipe_uri(uri);
        #[cfg(windows)]
        assert_eq!(parsed, r"\\.\pipe\llama_model");
        #[cfg(not(windows))]
        assert_eq!(parsed, "/tmp/llama_model");

        let raw = "/custom/path/to/fifo";
        assert_eq!(parse_pipe_uri(raw), raw);
    }
}
