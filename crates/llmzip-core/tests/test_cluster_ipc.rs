use llmzip_core::cluster::ipc::{ClusterIpcChannel, TensorIpcDescriptor};
use llmzip_core::cluster::tensor_parallel::{TensorParallelScheduler, TpSplitKind};
use llmzip_core::cluster::vllm::VllmClusterAdapter;
use llmzip_core::cluster::sglang::SglangClusterAdapter;

#[tokio::test]
async fn test_cluster_ipc_streaming_and_descriptors() {
    let channel_name = "test_vllm_ipc_stream";
    let (server, client) = ClusterIpcChannel::create_pair(channel_name).await.expect("IPC pair create failed");

    let descriptor = TensorIpcDescriptor {
        tensor_name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        shape: vec![4096, 4096],
        dtype: "F16".to_string(),
        offset: 0,
        size: 4096 * 4096 * 2,
    };

    // Server sends descriptor
    server.send_descriptor(&descriptor).await.expect("Send descriptor failed");

    // Client receives descriptor
    let received = client.recv_descriptor().await.expect("Recv descriptor failed");
    assert_eq!(received.tensor_name, descriptor.tensor_name);
    assert_eq!(received.shape, descriptor.shape);
    assert_eq!(received.size, descriptor.size);

    // Stream tensor payload chunk
    let test_data = vec![0x33u8; 1024 * 64];
    server.send_payload(&test_data).await.expect("Send payload failed");
    let recv_data = client.recv_payload(test_data.len()).await.expect("Recv payload failed");
    assert_eq!(recv_data, test_data);
}

#[test]
fn test_tensor_parallel_sharding_column_and_row() {
    let tp = TensorParallelScheduler::new(4); // TP = 4 ranks

    // Column parallel: e.g. [4096, 2048] split along dim 0 -> 1024 rows per rank
    let col_shape = vec![4096, 2048];
    for rank in 0..4 {
        let (start, end) = tp.get_slice_range(&col_shape, TpSplitKind::ColumnParallel, rank).unwrap();
        assert_eq!(start, rank * 1024);
        assert_eq!(end, (rank + 1) * 1024);
    }

    // Row parallel: e.g. [2048, 4096] split along dim 1 -> 1024 cols per rank
    let row_shape = vec![2048, 4096];
    for rank in 0..4 {
        let (start, end) = tp.get_slice_range(&row_shape, TpSplitKind::RowParallel, rank).unwrap();
        assert_eq!(start, rank * 1024);
        assert_eq!(end, (rank + 1) * 1024);
    }
}

#[test]
fn test_vllm_and_sglang_adapter_metadata() {
    let vllm_adapter = VllmClusterAdapter::new("Qwen/Qwen2.5-7B", 4);
    let vllm_manifest = vllm_adapter.generate_manifest(vec![
        ("model.layers.0.self_attn.q_proj.weight".to_string(), vec![4096, 4096]),
        ("model.layers.0.self_attn.o_proj.weight".to_string(), vec![4096, 4096]),
    ]);
    assert_eq!(vllm_manifest.tp_world_size, 4);
    assert!(vllm_manifest.tensor_keys.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));

    let sglang_adapter = SglangClusterAdapter::new("deepseek-ai/DeepSeek-V3", 8);
    let sglang_cfg = sglang_adapter.generate_config();
    assert_eq!(sglang_cfg.tp_size, 8);
    assert_eq!(sglang_cfg.model_id, "deepseek-ai/DeepSeek-V3");
}
