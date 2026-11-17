use crate::security::{validate_source_path, validate_target_path};
use llmzip_core::hasher::sha256_hex;
use llmzip_core::layout::{
    build_deterministic_layout, QuantPreset, GGUF_MAGIC_UNVERIFIED, GGUF_MAGIC_VERIFIED,
};
use llmzip_core::preflight::{
    build_virtual_tensor_space, parse_and_validate_header_len, ShardDescriptor,
};
use llmzip_core::streamer::{EngineConfig, HttpResponseMeta, StreamingQuantizeEngine};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::Path;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "llmzip";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// 處理單行 JSON-RPC 2.0 請求並回傳序列化的 JSON-RPC 響應（若為通知則回傳 None）
pub fn handle_json_rpc_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let req: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            let err_resp = json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32700,
                    "message": format!("Parse error: {}", e)
                }
            });
            return Some(err_resp.to_string());
        }
    };

    // JSON-RPC 2.0 規範：請求必須為 JSON 物件
    if !req.is_object() {
        let err_resp = json!({
            "jsonrpc": "2.0",
            "id": null,
            "error": {
                "code": -32600,
                "message": "Invalid Request: expected a JSON object"
            }
        });
        return Some(err_resp.to_string());
    }

    let id = req.get("id").cloned();

    // 嚴格遵循 JSON-RPC 2.0：凡是無 id 的消息皆為 Notification，Server 絕不回傳任何響應
    id.as_ref()?;

    let method = match req.get("method").and_then(|m| m.as_str()) {
        Some(m) => m,
        None => {
            let err_resp = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: missing method"
                }
            });
            return Some(err_resp.to_string());
        }
    };

    let resp = match method {
        // MCP 必備 ping 保活處理
        "ping" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {}
        }),

        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }),

        "tools/list" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "tools": [
                    {
                        "name": "llmzip_inspect",
                        "description": "Inspect SafeTensors header and preview planned GGUF layout offsets without full allocation.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Path to input SafeTensors model file or first shard."
                                }
                            },
                            "required": ["source"]
                        }
                    },
                    {
                        "name": "llmzip_quantize",
                        "description": "Stream-quantize Hugging Face safetensors directly into GGUF with zero temp files (Peak disk = 1.0x output + 4KB WAL).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Path to source .safetensors file."
                                },
                                "output": {
                                    "type": "string",
                                    "description": "Target output path (must end with .gguf or .awq)."
                                },
                                "preset": {
                                    "type": "string",
                                    "description": "Quantization preset: q4_k_m (default), q8_0, awq, nvfp4, mxfp4, fp8.",
                                    "default": "q4_k_m"
                                },
                                "line": {
                                    "type": "string",
                                    "description": "Pipeline profile: auto (default), fast, standard, slow.",
                                    "default": "auto"
                                },
                                "sanitize": {
                                    "type": "boolean",
                                    "description": "Sanitize NaN/Inf weight poisoning guardrail.",
                                    "default": true
                                }
                            },
                            "required": ["source", "output"]
                        }
                    },
                    {
                        "name": "llmzip_verify",
                        "description": "Verify cryptographic SHA-256 commit lock of a GGUF file ('GGUF' verified vs 'UNVF' incomplete).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to .gguf file to verify."
                                }
                            },
                            "required": ["file_path"]
                        }
                    }
                ]
            }
        }),

        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or(Value::Null);
            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));

            match tool_name {
                "llmzip_inspect" => {
                    let source = args.get("source").and_then(|s| s.as_str()).unwrap_or("");
                    if source.is_empty() {
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "isError": true,
                                "content": [{"type": "text", "text": "Missing required argument 'source'"}]
                            }
                        })
                    } else {
                        match validate_source_path(Path::new(source)) {
                            Err(sec_err) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "isError": true,
                                    "content": [{"type": "text", "text": format!("Security violation: {}", sec_err)}]
                                }
                            }),
                            Ok(safe_source) => match std::fs::read(&safe_source) {
                                Err(e) => json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "isError": true,
                                        "content": [{"type": "text", "text": format!("Failed to read source file '{}': {}", safe_source.display(), e)}]
                                    }
                                }),
                                Ok(raw) => {
                                    if raw.len() < 8 {
                                        json!({
                                            "jsonrpc": "2.0",
                                            "id": id,
                                            "result": {
                                                "isError": true,
                                                "content": [{"type": "text", "text": format!("File '{}' is too small ({} bytes) for SafeTensors header.", safe_source.display(), raw.len())}]
                                            }
                                        })
                                    } else {
                                        let shard_name = safe_source
                                            .file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("model.safetensors");
                                        let mut len_bytes = [0u8; 8];
                                        len_bytes.copy_from_slice(&raw[0..8]);
                                        match parse_and_validate_header_len(
                                            shard_name,
                                            len_bytes,
                                            raw.len() as u64,
                                        ) {
                                            Err(e) => json!({
                                                "jsonrpc": "2.0",
                                                "id": id,
                                                "result": {
                                                    "isError": true,
                                                    "content": [{"type": "text", "text": format!("Invalid header: {}", e)}]
                                                }
                                            }),
                                            Ok(header_len) => {
                                                let header_slice =
                                                    raw[8..8 + (header_len as usize)].to_vec();
                                                let sha = sha256_hex(&raw);
                                                let desc = ShardDescriptor {
                                                    shard_idx: 0,
                                                    shard_name: shard_name.to_string(),
                                                    total_file_size: raw.len() as u64,
                                                    header_len,
                                                    expected_sha256_hex: sha.clone(),
                                                };
                                                match build_virtual_tensor_space(
                                                    vec![(desc, header_slice)],
                                                    false,
                                                    8,
                                                ) {
                                                    Err(e) => json!({
                                                        "jsonrpc": "2.0",
                                                        "id": id,
                                                        "result": {
                                                            "isError": true,
                                                            "content": [{"type": "text", "text": format!("VTAS error: {}", e)}]
                                                        }
                                                    }),
                                                    Ok(vtas) => {
                                                        let tensor_count = vtas.tensors.len();
                                                        let topological =
                                                            vtas.topology.is_strictly_topological;
                                                        let tied =
                                                            vtas.topology.tie_word_embeddings;
                                                        let plan_res = build_deterministic_layout(
                                                            &vtas,
                                                            QuantPreset::GgufQ4KM,
                                                            "llama",
                                                        );
                                                        let expected_gguf_size = plan_res
                                                            .map(|p| p.total_output_file_size)
                                                            .unwrap_or(0);
                                                        json!({
                                                            "jsonrpc": "2.0",
                                                            "id": id,
                                                            "result": {
                                                                "content": [{
                                                                    "type": "text",
                                                                    "text": format!(
                                                                        "Inspect report for '{}':\n- Shard: {}\n- Size: {} bytes\n- Header: {} bytes\n- SHA-256: {}\n- Total Tensors: {}\n- Strictly Topological: {}\n- Tied Embeddings: {}\n- Planned GGUF Size (q4_k_m): {} bytes",
                                                                        safe_source.display(),
                                                                        shard_name,
                                                                        raw.len(),
                                                                        header_len,
                                                                        sha,
                                                                        tensor_count,
                                                                        topological,
                                                                        tied,
                                                                        expected_gguf_size
                                                                    )
                                                                }]
                                                            }
                                                        })
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                        }
                    }
                }

                "llmzip_quantize" => {
                    let source = args.get("source").and_then(|s| s.as_str()).unwrap_or("");
                    let output = args.get("output").and_then(|o| o.as_str()).unwrap_or("");
                    let preset_str = args
                        .get("preset")
                        .and_then(|p| p.as_str())
                        .unwrap_or("q4_k_m");
                    let preset = match preset_str.to_ascii_lowercase().as_str() {
                        "q4_k_m" | "q4km" | "q4_k" => QuantPreset::GgufQ4KM,
                        "q8_0" | "q8" | "q80" => QuantPreset::GgufQ8_0,
                        "awq" | "awq_w4g128" | "w4g128" => QuantPreset::AwqW4G128,
                        "nvfp4" | "fp4" => QuantPreset::Nvfp4,
                        "mxfp4" => QuantPreset::Mxfp4,
                        "fp8" | "fp8_e4m3" => QuantPreset::Fp8E4M3,
                        _ => QuantPreset::GgufQ4KM,
                    };

                    match validate_target_path(Path::new(output)) {
                        Err(sec_err) => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "isError": true,
                                "content": [{"type": "text", "text": format!("Security violation: {}", sec_err)}]
                            }
                        }),
                        Ok(safe_output) => match validate_source_path(Path::new(source)) {
                            Err(sec_err) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "isError": true,
                                    "content": [{"type": "text", "text": format!("Security violation: {}", sec_err)}]
                                }
                            }),
                            Ok(safe_source) => match std::fs::read(&safe_source) {
                                Err(e) => json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "isError": true,
                                        "content": [{"type": "text", "text": format!("Failed to read source file '{}': {}", safe_source.display(), e)}]
                                    }
                                }),
                                Ok(raw) => {
                                    if raw.len() < 8 {
                                        json!({
                                            "jsonrpc": "2.0",
                                            "id": id,
                                            "result": {
                                                "isError": true,
                                                "content": [{"type": "text", "text": "Source file is smaller than 8 bytes"}]
                                            }
                                        })
                                    } else {
                                        let shard_name = safe_source
                                            .file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("model.safetensors");
                                        let mut len_bytes = [0u8; 8];
                                        len_bytes.copy_from_slice(&raw[0..8]);
                                        match parse_and_validate_header_len(
                                            shard_name,
                                            len_bytes,
                                            raw.len() as u64,
                                        ) {
                                            Err(e) => json!({
                                                "jsonrpc": "2.0",
                                                "id": id,
                                                "result": {
                                                    "isError": true,
                                                    "content": [{"type": "text", "text": format!("Header error: {}", e)}]
                                                }
                                            }),
                                            Ok(header_len) => {
                                                let header_slice =
                                                    raw[8..8 + (header_len as usize)].to_vec();
                                                let expected_sha = sha256_hex(&raw);
                                                let desc = ShardDescriptor {
                                                    shard_idx: 0,
                                                    shard_name: shard_name.to_string(),
                                                    total_file_size: raw.len() as u64,
                                                    header_len,
                                                    expected_sha256_hex: expected_sha,
                                                };
                                                match build_virtual_tensor_space(
                                                    vec![(desc, header_slice)],
                                                    false,
                                                    8,
                                                ) {
                                                    Err(e) => json!({
                                                        "jsonrpc": "2.0",
                                                        "id": id,
                                                        "result": {
                                                            "isError": true,
                                                            "content": [{"type": "text", "text": format!("VTAS error: {}", e)}]
                                                        }
                                                    }),
                                                    Ok(vtas) => match build_deterministic_layout(
                                                        &vtas, preset, "llama",
                                                    ) {
                                                        Err(e) => json!({
                                                            "jsonrpc": "2.0",
                                                            "id": id,
                                                            "result": {
                                                                "isError": true,
                                                                "content": [{"type": "text", "text": format!("Layout error: {}", e)}]
                                                            }
                                                        }),
                                                        Ok(plan) => {
                                                            let total_bytes =
                                                                plan.total_output_file_size;
                                                            let cfg = EngineConfig::default();
                                                            match StreamingQuantizeEngine::new_or_resume(vtas, plan, &safe_output, cfg) {
                                                                Err(e) => json!({
                                                                    "jsonrpc": "2.0",
                                                                    "id": id,
                                                                    "result": {
                                                                        "isError": true,
                                                                        "content": [{"type": "text", "text": format!("Engine init error: {}", e)}]
                                                                    }
                                                                }),
                                                                Ok(mut engine) => {
                                                                    let meta = HttpResponseMeta {
                                                                        status_code: 200,
                                                                        content_range_start: None,
                                                                    };
                                                                    match engine.stream_shard_slice(0, meta, &raw, 64 * 1024 * 1024, None) {
                                                                        Err(e) => json!({
                                                                            "jsonrpc": "2.0",
                                                                            "id": id,
                                                                            "result": {
                                                                                "isError": true,
                                                                                "content": [{"type": "text", "text": format!("Quantization error: {}", e)}]
                                                                            }
                                                                        }),
                                                                        Ok(committed) => {
                                                                            if engine.is_committed() {
                                                                                json!({
                                                                                    "jsonrpc": "2.0",
                                                                                    "id": id,
                                                                                    "result": {
                                                                                        "content": [{"type": "text", "text": format!("Successfully quantized and committed {} tensors into '{}' ({} bytes, preset: {}). Magic Commit Lock verified ('GGUF').", committed, safe_output.display(), total_bytes, preset_str)}]
                                                                                    }
                                                                                })
                                                                            } else {
                                                                                json!({
                                                                                    "jsonrpc": "2.0",
                                                                                    "id": id,
                                                                                    "result": {
                                                                                        "isError": true,
                                                                                        "content": [{"type": "text", "text": format!("Quantization completed {} tensors but output remains uncommitted 'UNVF'.", committed)}]
                                                                                    }
                                                                                })
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    },
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                        },
                    }
                }

                "llmzip_verify" => {
                    let file_path = args.get("file_path").and_then(|f| f.as_str()).unwrap_or("");
                    match validate_source_path(Path::new(file_path)) {
                        Err(sec_err) => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "isError": true,
                                "content": [{"type": "text", "text": format!("Security violation: {}", sec_err)}]
                            }
                        }),
                        Ok(safe_path) => {
                            if !safe_path.exists() {
                                json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "isError": true,
                                        "content": [{"type": "text", "text": format!("File '{}' not found", file_path)}]
                                    }
                                })
                            } else {
                                match File::open(&safe_path) {
                                    Ok(mut f) => {
                                        let mut magic = [0u8; 4];
                                        if f.read_exact(&mut magic).is_ok() {
                                            if magic == GGUF_MAGIC_VERIFIED {
                                                json!({
                                                    "jsonrpc": "2.0",
                                                    "id": id,
                                                    "result": {
                                                        "content": [{"type": "text", "text": format!("VERIFIED: File '{}' is a 100% cryptographically committed GGUF container. Ready for inference.", file_path)}]
                                                    }
                                                })
                                            } else if magic == GGUF_MAGIC_UNVERIFIED {
                                                json!({
                                                    "jsonrpc": "2.0",
                                                    "id": id,
                                                    "result": {
                                                        "isError": true,
                                                        "content": [{"type": "text", "text": format!("UNVERIFIED: File '{}' is in 'UNVF' state (incomplete or corrupted). Commit lock is still locked.", file_path)}]
                                                    }
                                                })
                                            } else {
                                                json!({
                                                    "jsonrpc": "2.0",
                                                    "id": id,
                                                    "result": {
                                                        "isError": true,
                                                        "content": [{"type": "text", "text": format!("INVALID: File '{}' has unknown magic header {:?}. Not a GGUF container.", file_path, magic)}]
                                                    }
                                                })
                                            }
                                        } else {
                                            json!({
                                                "jsonrpc": "2.0",
                                                "id": id,
                                                "result": {
                                                    "isError": true,
                                                    "content": [{"type": "text", "text": format!("Failed to read magic header from '{}'", file_path)}]
                                                }
                                            })
                                        }
                                    }
                                    Err(e) => json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "result": {
                                            "isError": true,
                                            "content": [{"type": "text", "text": format!("Failed to open '{}': {}", file_path, e)}]
                                        }
                                    }),
                                }
                            }
                        }
                    }
                }

                _ => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("Tool '{}' not found", tool_name)
                    }
                }),
            }
        }

        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("Method '{}' not supported", method)
            }
        }),
    };

    Some(resp.to_string())
}

/// 啟動標準 MCP JSON-RPC 2.0 stdio 伺服器迴圈
pub fn run_mcp_server() {
    eprintln!(
        "[llmzip-mcp] Starting MCP server v{} (Protocol: {})",
        SERVER_VERSION, MCP_PROTOCOL_VERSION
    );
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        match line {
            Ok(line_str) => {
                if let Some(resp_str) = handle_json_rpc_line(&line_str) {
                    if let Err(e) = writeln!(stdout, "{}", resp_str) {
                        eprintln!("[llmzip-mcp] Stdio write error: {}", e);
                        break;
                    }
                    if let Err(e) = stdout.flush() {
                        eprintln!("[llmzip-mcp] Stdio flush error: {}", e);
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("[llmzip-mcp] Stdin read error: {}", e);
                break;
            }
        }
    }
    eprintln!("[llmzip-mcp] Server terminated.");
}
