use llmzip_core::pipe::named_pipe::{parse_pipe_uri, LlmZipPipe};
use tokio::io::AsyncReadExt;

#[tokio::test]
async fn test_pipe_roundtrip_and_broken_pipe() {
    let pipe_uri = "pipe://llmzip_unit_test_pipe";
    let pipe_path = parse_pipe_uri(pipe_uri);

    let mut pipe = LlmZipPipe::create(pipe_uri).expect("Failed to create pipe");

    // Client connection task
    let client_path = pipe_path.clone();
    let client_handle = tokio::spawn(async move {
        #[cfg(windows)]
        {
            let mut client = tokio::net::windows::named_pipe::ClientOptions::new()
                .open(&client_path)
                .expect("Client failed to open pipe");

            let mut buf = vec![0u8; 16];
            client.read_exact(&mut buf).await.expect("Failed to read");
            buf
        }

        #[cfg(not(windows))]
        {
            let mut file = tokio::fs::OpenOptions::new()
                .read(true)
                .open(&client_path)
                .await
                .expect("Failed opening FIFO");
            let mut buf = vec![0u8; 16];
            file.read_exact(&mut buf).await.expect("Failed to read");
            buf
        }
    });

    // Server waits for reader and writes 16 bytes
    pipe.wait_for_reader().await.expect("Wait for reader failed");
    let test_data = b"0123456789ABCDEF";
    pipe.write_all(test_data).await.expect("Write all failed");
    pipe.flush().await.expect("Flush failed");

    let received = client_handle.await.expect("Client task panicked");
    assert_eq!(&received, test_data);
}
