use llmzip_cli::mcp::handle_json_rpc_line;
use llmzip_cli::security::{validate_target_path, SecurityError};
use serde_json::{json, Value};
use std::path::Path;

#[test]
fn test_security_path_validation() {
    // 1. 合法路徑
    assert!(validate_target_path(Path::new("output.gguf")).is_ok());
    assert!(validate_target_path(Path::new("models/my_model.awq")).is_ok());

    // 2. 非法副檔名
    match validate_target_path(Path::new("script.sh")) {
        Err(SecurityError::InvalidExtension(_)) => {}
        other => panic!("Expected InvalidExtension, got {:?}", other),
    }

    match validate_target_path(Path::new("malicious.exe")) {
        Err(SecurityError::InvalidExtension(_)) => {}
        other => panic!("Expected InvalidExtension, got {:?}", other),
    }

    // 3. 路徑穿越攻擊
    match validate_target_path(Path::new("../../etc/passwd.gguf")) {
        Err(SecurityError::PathTraversalDetected) => {}
        other => panic!("Expected PathTraversalDetected, got {:?}", other),
    }

    // 4. 空路徑
    match validate_target_path(Path::new("")) {
        Err(SecurityError::EmptyPath) => {}
        other => panic!("Expected EmptyPath, got {:?}", other),
    }
}

#[test]
fn test_mcp_initialize() {
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-agent",
                "version": "1.0.0"
            }
        }
    });

    let resp_str = handle_json_rpc_line(&req.to_string()).expect("Should return a response");
    let resp: Value = serde_json::from_str(&resp_str).expect("Valid JSON response");

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
    assert!(resp["result"]["capabilities"]["tools"].is_object());
    assert_eq!(resp["result"]["serverInfo"]["name"], "llmzip");
}

#[test]
fn test_mcp_tools_list() {
    let req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });

    let resp_str = handle_json_rpc_line(&req.to_string()).expect("Should return a response");
    let resp: Value = serde_json::from_str(&resp_str).expect("Valid JSON response");

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 2);

    let tools = resp["result"]["tools"].as_array().expect("Tools must be an array");
    let tool_names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("Tool must have name"))
        .collect();

    assert!(tool_names.contains(&"llmzip_inspect"), "Must expose llmzip_inspect");
    assert!(tool_names.contains(&"llmzip_quantize"), "Must expose llmzip_quantize");
    assert!(tool_names.contains(&"llmzip_verify"), "Must expose llmzip_verify");
}

#[test]
fn test_mcp_security_block_on_call() {
    let req = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "llmzip_quantize",
            "arguments": {
                "source": "dummy.safetensors",
                "output": "../../root/evil.sh"
            }
        }
    });

    let resp_str = handle_json_rpc_line(&req.to_string()).expect("Should return response");
    let resp: Value = serde_json::from_str(&resp_str).expect("Valid JSON response");

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 3);
    assert_eq!(resp["result"]["isError"], true, "Should be flagged as error");
    let content = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(content.contains("Security violation") || content.contains("Path traversal") || content.contains("Invalid extension"));
}
