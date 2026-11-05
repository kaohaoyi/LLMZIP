use llmzip_core::hf::streamer::{HfParallelStreamer, HfStreamerConfig};
use llmzip_core::layout::QuantPreset;
use std::fs;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Spins up a minimal mock HTTP server supporting `/api/models/...` and HTTP Range slicing.
async fn run_mock_hf_server(raw_safetensors: Arc<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(_) => break,
            };

            let data = raw_safetensors.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let n = match socket.read(&mut buf).await {
                    Ok(n) if n > 0 => n,
                    _ => return,
                };
                let req_str = String::from_utf8_lossy(&buf[..n]);

                if req_str.starts_with("GET /api/models/") {
                    let total_size = data.len();
                    let body = format!(
                        r#"{{"siblings": [{{"rfilename": "model.safetensors", "size": {}}}]}}"#,
                        total_size
                    );
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                } else if req_str.contains("/resolve/main/model.safetensors") {
                    let total_size = data.len();
                    // Parse Range header: "Range: bytes=start-end"
                    let mut start = 0usize;
                    let mut end = total_size.saturating_sub(1);

                    for line in req_str.lines() {
                        if line.to_ascii_lowercase().starts_with("range: bytes=") {
                            let range_val = line["range: bytes=".len()..].trim();
                            let parts: Vec<&str> = range_val.split('-').collect();
                            if let Ok(s) = parts[0].parse::<usize>() {
                                start = s;
                            }
                            if parts.len() > 1 && !parts[1].is_empty() {
                                if let Ok(e) = parts[1].parse::<usize>() {
                                    end = e.min(total_size - 1);
                                }
                            }
                            break;
                        }
                    }

                    if start <= end && start < total_size {
                        let chunk = &data[start..=end];
                        let resp = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            start,
                            end,
                            total_size,
                            chunk.len()
                        );
                        let _ = socket.write_all(resp.as_bytes()).await;
                        let _ = socket.write_all(chunk).await;
                    } else {
                        let resp = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                        let _ = socket.write_all(resp.as_bytes()).await;
                    }
                } else {
                    let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = socket.write_all(resp.as_bytes()).await;
                }
            });
        }
    });

    format!("http://{}", addr)
}

#[tokio::test]
async fn test_hf_parallel_streaming_mock() {
    // 1. Build synthetic SafeTensors bytes
    let header_json = br#"{"model.norm.weight":{"dtype":"F32","shape":[32],"data_offsets":[0,128]}}"#;
    let header_len = header_json.len() as u64;

    let mut raw_data = Vec::new();
    raw_data.extend_from_slice(&header_len.to_le_bytes());
    raw_data.extend_from_slice(header_json);
    for _ in 0..32 {
        raw_data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let server_url = run_mock_hf_server(Arc::new(raw_data)).await;

    let mut out_path = std::env::temp_dir();
    out_path.push("test_hf_stream_out.gguf");
    if out_path.exists() {
        let _ = fs::remove_file(&out_path);
    }

    let config = HfStreamerConfig {
        repo_id: "test-org/test-model".to_string(),
        revision: Some("main".to_string()),
        auth_token: Some("hf_test_token_1234".to_string()),
        concurrency: 4,
        chunk_size: 48, // Intentionally small chunk size to test multi-chunk reassembly
        preset: QuantPreset::GgufQ4KM,
        output_path: out_path.clone(),
        base_url: Some(server_url),
        expected_sha256: None,
    };

    let streamer = HfParallelStreamer::new(config);
    let result = streamer.stream_to_gguf().await;
    assert!(result.is_ok(), "Streaming failed: {:?}", result.err());

    assert!(out_path.exists());
    let gguf_bytes = fs::read(&out_path).unwrap();
    assert!(gguf_bytes.len() > 4);
    assert_eq!(&gguf_bytes[0..4], b"GGUF");

    let _ = fs::remove_file(&out_path);
}
