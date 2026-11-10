use llmzip_core::swarm::{Chunker, SwarmLeecher, SwarmSeeder};
use std::sync::Arc;

#[tokio::test]
async fn test_p2p_swarm_server_and_fetch_roundtrip() {
    let dummy_weights = vec![0x42u8; 1024 * 512]; // 512 KB
    let seeder = Arc::new(SwarmSeeder::new());
    let manifest = seeder.seed_raw_bytes("test-llama-8b", &dummy_weights, 1024 * 128).await;
    assert_eq!(manifest.chunks.len(), 4);

    let server_addr = seeder.start("127.0.0.1:0").await.expect("Seeder start failed");

    // Leecher fetches single chunk
    let chunk0 = SwarmLeecher::fetch_chunk(
        &server_addr.to_string(),
        "test-llama-8b",
        &manifest.chunks[0],
    )
    .await
    .expect("Fetch chunk failed");

    assert_eq!(chunk0.len(), 1024 * 128);
    assert_eq!(chunk0.as_ref(), &dummy_weights[0..1024 * 128]);

    // Leecher fetches full model
    let all_chunks = SwarmLeecher::pull_model(
        &[server_addr.to_string()],
        &manifest,
    )
    .await
    .expect("Pull model failed");

    assert_eq!(all_chunks.len(), 4);
    let mut reconstructed = Vec::new();
    for c in all_chunks {
        reconstructed.extend_from_slice(&c);
    }
    assert_eq!(reconstructed, dummy_weights);
}

#[tokio::test]
async fn test_p2p_swarm_poison_chunk_rejection() {
    let dummy_weights = vec![0x11u8; 1024 * 64];
    let (mut manifest, _) = Chunker::split("poison-model", &dummy_weights, 1024 * 64);
    
    // Deliberately corrupt expected hash in manifest
    manifest.chunks[0].sha256 = "0000000000000000000000000000000000000000000000000000000000000000".to_string();

    let seeder = Arc::new(SwarmSeeder::new());
    seeder.store.insert_manifest(manifest.clone()).await;
    seeder.store.insert_chunk("poison-model", 0, bytes::Bytes::from(dummy_weights)).await;
    let server_addr = seeder.start("127.0.0.1:0").await.expect("Seeder start failed");

    let res = SwarmLeecher::fetch_chunk(
        &server_addr.to_string(),
        "poison-model",
        &manifest.chunks[0],
    )
    .await;

    assert!(res.is_err());
    let err_msg = res.unwrap_err().to_string();
    assert!(err_msg.contains("In-flight poison defense"));
}
