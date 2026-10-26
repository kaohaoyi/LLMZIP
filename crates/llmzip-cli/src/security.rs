use std::fmt;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, PartialEq, Eq)]
pub enum SecurityError {
    EmptyPath,
    InvalidExtension(String),
    PathTraversalDetected,
    DisallowedOverwrite(String),
    IoError(String),
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::EmptyPath => write!(f, "Security violation: target path cannot be empty"),
            SecurityError::InvalidExtension(ext) => {
                write!(f, "Security violation: Invalid extension '{}'. Only .gguf and .awq files are permitted.", ext)
            }
            SecurityError::PathTraversalDetected => {
                write!(f, "Security violation: Path traversal ('..') detected in output path")
            }
            SecurityError::DisallowedOverwrite(msg) => write!(f, "Security violation: Overwrite disallowed: {}", msg),
            SecurityError::IoError(msg) => write!(f, "I/O Security error: {}", msg),
        }
    }
}

impl std::error::Error for SecurityError {}

/// 驗證輸出檔案路徑的安全性，防止路徑穿越與非法副檔名覆蓋
pub fn validate_target_path(path: &Path) -> Result<PathBuf, SecurityError> {
    if path.as_os_str().is_empty() {
        return Err(SecurityError::EmptyPath);
    }

    // 檢查是否有 ParentDir (..) 試圖路徑穿越
    for comp in path.components() {
        if let Component::ParentDir = comp {
            return Err(SecurityError::PathTraversalDetected);
        }
    }

    // 副檔名強制白名單：.gguf, .awq
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if ext != "gguf" && ext != "awq" {
        return Err(SecurityError::InvalidExtension(ext));
    }

    // 如果檔案已存在且為目錄，禁止覆蓋
    if path.exists() && path.is_dir() {
        return Err(SecurityError::DisallowedOverwrite(
            "Target path points to an existing directory".to_string(),
        ));
    }

    Ok(path.to_path_buf())
}
