use crate::error::{LlmZipError, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;

/// Redact sensitive token strings to prevent leaks in logs, WAL, and terminal displays.
pub fn redact_token(token: &str) -> String {
    if token.len() <= 8 {
        "[REDACTED]".to_string()
    } else {
        format!("{}...[REDACTED]...{}", &token[..4], &token[token.len() - 4..])
    }
}

/// Shard file metadata from Hugging Face repository snapshot.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfFileEntry {
    pub filename: String,
    pub url: String,
    pub size: Option<u64>,
}

impl fmt::Debug for HfFileEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfFileEntry")
            .field("filename", &self.filename)
            .field("url", &self.url)
            .field("size", &self.size)
            .finish()
    }
}

/// Hugging Face repository descriptor with redacted token in Debug representation.
#[derive(Clone, Serialize, Deserialize)]
pub struct HfRepoDescriptor {
    pub repo_id: String,
    pub revision: String,
    pub files: Vec<HfFileEntry>,
    pub auth_token: Option<String>,
}

impl fmt::Debug for HfRepoDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfRepoDescriptor")
            .field("repo_id", &self.repo_id)
            .field("revision", &self.revision)
            .field("files", &self.files)
            .field(
                "auth_token",
                &self.auth_token.as_ref().map(|t| redact_token(t)),
            )
            .finish()
    }
}

/// Hugging Face Hub HTTP API client for model snapshot discovery and direct downloading.
pub struct HfClient {
    client: reqwest::Client,
    base_url: String,
    auth_token: Option<String>,
}

#[derive(Deserialize)]
struct SiblingsEntry {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct HfModelInfoResponse {
    siblings: Option<Vec<SiblingsEntry>>,
}

impl HfClient {
    /// Create a new HfClient resolving token from parameter or environment (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`).
    pub fn new(auth_token: Option<String>) -> Self {
        let token = auth_token
            .or_else(|| env::var("HF_TOKEN").ok())
            .or_else(|| env::var("HUGGING_FACE_HUB_TOKEN").ok())
            .filter(|t| !t.trim().is_empty());
        Self::with_endpoint(token, "https://huggingface.co".to_string())
    }

    /// Create client with a custom base endpoint (essential for mock testing and enterprise proxies).
    pub fn with_endpoint(auth_token: Option<String>, base_url: String) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("LLMZIP-NativeStreamer/0.3.0"),
        );
        if let Some(ref token) = auth_token {
            if let Ok(mut val) = HeaderValue::from_str(&format!("Bearer {}", token)) {
                val.set_sensitive(true);
                headers.insert(AUTHORIZATION, val);
            }
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token,
        }
    }

    /// Format resolve download URL for a file in repository.
    pub fn get_download_url(&self, repo_id: &str, revision: &str, filename: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.base_url,
            repo_id.trim_matches('/'),
            revision,
            filename
        )
    }

    /// Query the Hugging Face Hub API for model files and parse safe-tensor shard list.
    pub async fn get_model_info(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<HfRepoDescriptor> {
        let rev = revision.unwrap_or("main");
        let api_url = format!("{}/api/models/{}", self.base_url, repo_id);

        let mut req = self.client.get(&api_url);
        if rev != "main" {
            req = req.query(&[("revision", rev)]);
        }

        let resp = req.send().await.map_err(|e| LlmZipError::HfError {
            repo_id: repo_id.to_string(),
            reason: format!("Failed to connect to Hugging Face Hub API: {}", e),
        })?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(LlmZipError::HfError {
                repo_id: repo_id.to_string(),
                reason: "Authentication required or forbidden (HTTP 401/403). Set HF_TOKEN environment variable.".to_string(),
            });
        }

        if !status.is_success() {
            return Err(LlmZipError::HfError {
                repo_id: repo_id.to_string(),
                reason: format!("Hugging Face API returned HTTP status {}", status),
            });
        }

        let model_info: HfModelInfoResponse = resp.json().await.map_err(|e| LlmZipError::HfError {
            repo_id: repo_id.to_string(),
            reason: format!("Failed to parse Hugging Face model metadata JSON: {}", e),
        })?;

        let mut files = Vec::new();
        if let Some(siblings) = model_info.siblings {
            for sibling in siblings {
                if sibling.rfilename.ends_with(".safetensors") {
                    let dl_url = self.get_download_url(repo_id, rev, &sibling.rfilename);
                    files.push(HfFileEntry {
                        filename: sibling.rfilename,
                        url: dl_url,
                        size: sibling.size,
                    });
                }
            }
        }

        // Sort files deterministically (e.g., model-00001-of-00004.safetensors, model-00002...)
        files.sort_by(|a, b| a.filename.cmp(&b.filename));

        if files.is_empty() {
            return Err(LlmZipError::HfError {
                repo_id: repo_id.to_string(),
                reason: "No .safetensors weights found in repository snapshot".to_string(),
            });
        }

        Ok(HfRepoDescriptor {
            repo_id: repo_id.to_string(),
            revision: rev.to_string(),
            files,
            auth_token: self.auth_token.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_redaction() {
        assert_eq!(redact_token("short"), "[REDACTED]");
        let token = "hf_1234567890abcdef";
        let redacted = redact_token(token);
        assert!(redacted.starts_with("hf_1"));
        assert!(redacted.ends_with("cdef"));
        assert!(!redacted.contains("567890"));
    }

    #[test]
    fn test_repo_descriptor_debug_redaction() {
        let descriptor = HfRepoDescriptor {
            repo_id: "Qwen/Qwen2.5-7B".to_string(),
            revision: "main".to_string(),
            files: vec![HfFileEntry {
                filename: "model.safetensors".to_string(),
                url: "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/model.safetensors".to_string(),
                size: Some(1024),
            }],
            auth_token: Some("hf_secret_token_12345".to_string()),
        };

        let debug_str = format!("{:?}", descriptor);
        assert!(debug_str.contains("[REDACTED]"));
        assert!(!debug_str.contains("secret_token"));
    }

    #[test]
    fn test_get_download_url() {
        let client = HfClient::with_endpoint(None, "https://huggingface.co/".to_string());
        let url = client.get_download_url("meta-llama/Llama-3-8B", "main", "model-00001-of-00004.safetensors");
        assert_eq!(
            url,
            "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/model-00001-of-00004.safetensors"
        );
    }
}
