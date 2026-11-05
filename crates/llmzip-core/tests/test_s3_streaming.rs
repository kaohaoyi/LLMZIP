use bytes::Bytes;
use futures::executor::block_on;
use llmzip_core::cloud::s3_streamer::{S3MultipartStreamer, MIN_S3_PART_SIZE};
use object_store::memory::InMemory;
use object_store::path::Path as StorePath;
use object_store::ObjectStore;
use std::sync::Arc;

#[test]
fn test_s3_multipart_streaming_chunking_and_completion() {
    let store = Arc::new(InMemory::new());
    let path = StorePath::from("models/test-model.gguf");

    // Configure 5 MiB part size
    let part_size = MIN_S3_PART_SIZE;
    let mut streamer = S3MultipartStreamer::from_store(store.clone(), path.clone(), Some(part_size))
        .expect("Failed to initialize streamer");

    // Write 12 MiB of test data (2 full 5 MiB parts + 1 2 MiB tail part)
    let total_bytes = 12 * 1024 * 1024;
    let chunk_size = 1024 * 1024; // 1 MiB chunks
    let test_chunk = vec![0xABu8; chunk_size];

    for _ in 0..12 {
        streamer.write_chunk(&test_chunk).expect("write_chunk failed");
    }

    assert_eq!(streamer.parts_count(), 2); // 2 parts flushed during loop

    let total_written = streamer.complete().expect("complete failed");
    assert_eq!(total_written, total_bytes as u64);

    // Verify stored object in store
    let get_res = block_on(store.get(&path)).expect("Object not found in store");
    let bytes: Bytes = block_on(get_res.bytes()).expect("Failed to read stored bytes");
    assert_eq!(bytes.len(), total_bytes);
    assert!(bytes.iter().all(|&b| b == 0xAB));
}

#[test]
fn test_s3_multipart_abort_on_error_or_drop() {
    let store = Arc::new(InMemory::new());
    let path = StorePath::from("models/aborted-model.gguf");

    {
        let mut streamer = S3MultipartStreamer::from_store(store.clone(), path.clone(), Some(MIN_S3_PART_SIZE))
            .expect("Failed to initialize streamer");

        let chunk = vec![0xEFu8; 6 * 1024 * 1024]; // 6 MiB triggers 1 part flush
        streamer.write_chunk(&chunk).expect("write_chunk failed");
        assert_eq!(streamer.parts_count(), 1);

        // Explicitly abort
        streamer.abort().expect("abort failed");
    }

    // After abort, object must NOT exist in storage
    let res = block_on(store.get(&path));
    assert!(res.is_err(), "Object must not exist after abort");
}

#[test]
fn test_s3_multipart_raii_drop_cleanup() {
    let store = Arc::new(InMemory::new());
    let path = StorePath::from("models/dropped-model.gguf");

    {
        let mut streamer = S3MultipartStreamer::from_store(store.clone(), path.clone(), Some(MIN_S3_PART_SIZE))
            .expect("Failed to initialize streamer");

        let chunk = vec![0x42u8; 6 * 1024 * 1024];
        streamer.write_chunk(&chunk).expect("write_chunk failed");
        assert_eq!(streamer.parts_count(), 1);

        // Streamer is dropped without calling complete() -> RAII drop calls abort()
    }

    let res = block_on(store.get(&path));
    assert!(res.is_err(), "Object must not exist when streamer dropped without complete");
}
