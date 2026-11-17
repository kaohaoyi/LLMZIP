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
            SecurityError::EmptyPath => {
                write!(f, "Security violation: target path cannot be empty")
            }
            SecurityError::InvalidExtension(ext) => {
                write!(f, "Security violation: Invalid extension '{}'. Only .gguf and .awq files are permitted.", ext)
            }
            SecurityError::PathTraversalDetected => {
                write!(f, "Security violation: Path traversal, absolute root, prefix drive or UNC detected in output path")
            }
            SecurityError::DisallowedOverwrite(msg) => {
                write!(f, "Security violation: Overwrite disallowed: {}", msg)
            }
            SecurityError::IoError(msg) => write!(f, "I/O Security error: {}", msg),
        }
    }
}

impl std::error::Error for SecurityError {}

const DOS_DEVICE_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

fn is_dos_device(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    DOS_DEVICE_NAMES.contains(&stem.as_str())
}

/// 驗證輸出檔案路徑的安全性，防止路徑穿越、Windows 盤符逃逸、UNC 攻擊、DOS 裝置與非法副檔名
pub fn validate_target_path(path: &Path) -> Result<PathBuf, SecurityError> {
    if path.as_os_str().is_empty() {
        return Err(SecurityError::EmptyPath);
    }

    if path.as_os_str().to_string_lossy().contains('\0') {
        return Err(SecurityError::PathTraversalDetected);
    }

    // 1. 禁止絕對路徑與帶根目錄路徑（例如 /etc/... 或 \System32\...）
    if path.is_absolute() || path.has_root() {
        return Err(SecurityError::PathTraversalDetected);
    }

    // 2. 逐一檢查組件：禁止 ParentDir (..)、Prefix (C:, \\server\share, \\?\) 與 RootDir
    for comp in path.components() {
        match comp {
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => {
                return Err(SecurityError::PathTraversalDetected);
            }
            Component::Normal(os_name) => {
                let name = os_name.to_string_lossy();
                if is_dos_device(&name) {
                    return Err(SecurityError::PathTraversalDetected);
                }
            }
            Component::CurDir => {}
        }
    }

    // 3. 檢查符號鏈接 / Junction
    let mut check_buf = PathBuf::new();
    for comp in path.components() {
        check_buf.push(comp);
        if let Ok(meta) = std::fs::symlink_metadata(&check_buf) {
            if meta.file_type().is_symlink() {
                return Err(SecurityError::PathTraversalDetected);
            }
        }
    }

    // 4. 副檔名強制白名單：.gguf, .awq
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if ext != "gguf" && ext != "awq" {
        return Err(SecurityError::InvalidExtension(ext));
    }

    // 5. 如果檔案已存在且為目錄，禁止覆蓋
    if path.exists() && path.is_dir() {
        return Err(SecurityError::DisallowedOverwrite(
            "Target path points to an existing directory".to_string(),
        ));
    }

    Ok(path.to_path_buf())
}

/// 驗證輸入檔案或目錄路徑的安全性，防止空位元組注入與 DOS 特殊裝置
pub fn validate_source_path(path: &Path) -> Result<PathBuf, SecurityError> {
    if path.as_os_str().is_empty() {
        return Err(SecurityError::EmptyPath);
    }

    if path.as_os_str().to_string_lossy().contains('\0') {
        return Err(SecurityError::PathTraversalDetected);
    }

    for comp in path.components() {
        if let Component::Normal(os_name) = comp {
            let name = os_name.to_string_lossy();
            if is_dos_device(&name) {
                return Err(SecurityError::PathTraversalDetected);
            }
        }
    }

    Ok(path.to_path_buf())
}
