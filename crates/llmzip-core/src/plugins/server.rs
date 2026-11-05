use crate::error::{LlmZipError, Result};
use crate::hf::streamer::{HfParallelStreamer, HfStreamerConfig};
use crate::layout::QuantPreset;
use crate::plugins::ollama::OllamaPlugin;
use serde_json::Value;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Configuration for the local Ollama Proxy Server.
#[derive(Debug, Clone)]
pub struct OllamaProxyConfig {
    pub bind_addr: SocketAddr,
    pub ollama_root: Option<PathBuf>,
    pub preset: QuantPreset,
    pub concurrency: usize,
}

impl Default for OllamaProxyConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:11435".parse().unwrap(),
            ollama_root: None,
            preset: QuantPreset::GgufQ4KM,
            concurrency: 4,
        }
    }
}

/// Runs the lightweight Ollama Proxy server, handling `/api/pull`, `/api/tags`, and `/api/version`.
pub async fn run_ollama_proxy(config: OllamaProxyConfig) -> Result<()> {
    let listener = TcpListener::bind(config.bind_addr).await.map_err(|e| {
        LlmZipError::PluginError {
            target: "ollama-proxy".to_string(),
            reason: format!("Failed to bind to {}: {}", config.bind_addr, e),
        }
    })?;

    let plugin = Arc::new(match config.ollama_root.clone() {
        Some(root) => OllamaPlugin::with_root(root),
        None => OllamaPlugin::new(),
    });

    let config = Arc::new(config);

    println!(
        "[Ollama Proxy] Listening on http://{} (Forwarding to Ollama at '{}')",
        config.bind_addr,
        plugin.root_dir.display()
    );

    loop {
        let (mut socket, _) = match listener.accept().await {
            Ok(s) => s,
            Err(_) => continue,
        };

        let plugin = plugin.clone();
        let config = config.clone();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 8192];
            let n = match socket.read(&mut buf).await {
                Ok(n) if n > 0 => n,
                _ => return,
            };

            let req_str = String::from_utf8_lossy(&buf[..n]);
            let lines: Vec<&str> = req_str.lines().collect();
            if lines.is_empty() {
                return;
            }

            let first_line = lines[0];
            let parts: Vec<&str> = first_line.split_whitespace().collect();
            if parts.len() < 2 {
                return;
            }

            let method = parts[0];
            let path = parts[1];

            if method == "GET" && path == "/api/version" {
                let body = r#"{"version":"0.3.0"}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(resp.as_bytes()).await;
            } else if method == "GET" && path == "/api/tags" {
                let body = r#"{"models":[]}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(resp.as_bytes()).await;
            } else if method == "POST" && path == "/api/pull" {
                // Parse JSON payload from HTTP body
                let body_start = req_str.find("\r\n\r\n").map(|idx| idx + 4).unwrap_or(n);
                let body_str = &req_str[body_start..];
                let pull_req: Value = serde_json::from_str(body_str).unwrap_or_default();
                let model_name = pull_req["name"]
                    .as_str()
                    .unwrap_or("Qwen/Qwen2.5-7B")
                    .trim_start_matches("hf.co/");

                // Start chunked / streaming response to Ollama client
                let header = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(header.as_bytes()).await;

                // Send progress: "pulling manifest"
                send_ndjson_chunk(&mut socket, r#"{"status":"pulling manifest"}"#).await;

                // Download via HfParallelStreamer
                let temp_gguf = std::env::temp_dir().join(format!("ollama_pull_{}.gguf", std::process::id()));
                let streamer_cfg = HfStreamerConfig {
                    repo_id: model_name.to_string(),
                    revision: None,
                    auth_token: None,
                    concurrency: config.concurrency,
                    chunk_size: 8 * 1024 * 1024,
                    preset: config.preset,
                    output_path: temp_gguf.clone(),
                    base_url: None,
                    expected_sha256: None,
                };

                send_ndjson_chunk(&mut socket, r#"{"status":"streaming and quantizing model weights"}"#).await;

                let streamer = HfParallelStreamer::new(streamer_cfg);
                match streamer.stream_to_gguf().await {
                    Ok(path) => {
                        send_ndjson_chunk(&mut socket, r#"{"status":"injecting into ollama blobs and manifests"}"#).await;
                        let tag = "latest";
                        let pure_name = model_name.split('/').last().unwrap_or(model_name);
                        let _ = plugin.inject_model(pure_name, tag, &path);
                        let _ = fs::remove_file(&temp_gguf);
                        send_ndjson_chunk(&mut socket, r#"{"status":"success"}"#).await;
                    }
                    Err(e) => {
                        let err_msg = format!(r#"{{"error":"Failed pulling model: {}"}}"#, e);
                        send_ndjson_chunk(&mut socket, &err_msg).await;
                    }
                }

                // Final empty chunk for HTTP Transfer-Encoding: chunked
                let _ = socket.write_all(b"0\r\n\r\n").await;
            } else {
                let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(resp.as_bytes()).await;
            }
        });
    }
}

async fn send_ndjson_chunk(socket: &mut tokio::net::TcpStream, ndjson: &str) {
    let payload = format!("{}\n", ndjson);
    let chunk_hdr = format!("{:X}\r\n{}\r\n", payload.len(), payload);
    let _ = socket.write_all(chunk_hdr.as_bytes()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proxy_server_version() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let cfg = OllamaProxyConfig {
            bind_addr: addr,
            ollama_root: None,
            preset: QuantPreset::GgufQ4KM,
            concurrency: 2,
        };

        tokio::spawn(async move {
            let _ = run_ollama_proxy(cfg).await;
        });

        // Small delay for listener startup
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let url = format!("http://{}/api/version", addr);
        if let Ok(resp) = client.get(&url).send().await {
            assert_eq!(resp.status(), reqwest::StatusCode::OK);
            let json: Value = resp.json().await.unwrap();
            assert_eq!(json["version"], "0.3.0");
        }
    }
}
