use bytes::{BufMut, BytesMut};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use crate::error::LlmZipError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorIpcDescriptor {
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: u64,
    pub size: usize,
}

pub struct ClusterIpcSender {
    stream: Arc<Mutex<TcpStream>>,
}

pub struct ClusterIpcReceiver {
    stream: Arc<Mutex<TcpStream>>,
}

pub struct ClusterIpcChannel;

impl ClusterIpcChannel {
    /// Creates a connected pair of IPC endpoints (Sender and Receiver)
    pub async fn create_pair(_name: &str) -> Result<(ClusterIpcSender, ClusterIpcReceiver), LlmZipError> {
        let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|e| LlmZipError::ClusterError {
            target: "ipc".to_string(),
            reason: format!("Failed to bind local IPC listener: {}", e),
        })?;
        let addr = listener.local_addr().map_err(LlmZipError::Io)?;

        let connect_fut = TcpStream::connect(addr);
        let accept_fut = listener.accept();

        let (client_res, accept_res) = tokio::join!(connect_fut, accept_fut);
        let client_stream = client_res.map_err(|e| LlmZipError::ClusterError {
            target: "ipc".to_string(),
            reason: format!("Failed to connect IPC client: {}", e),
        })?;
        let (server_stream, _) = accept_res.map_err(|e| LlmZipError::ClusterError {
            target: "ipc".to_string(),
            reason: format!("Failed to accept IPC server stream: {}", e),
        })?;

        Ok((
            ClusterIpcSender {
                stream: Arc::new(Mutex::new(server_stream)),
            },
            ClusterIpcReceiver {
                stream: Arc::new(Mutex::new(client_stream)),
            },
        ))
    }
}

impl ClusterIpcSender {
    pub async fn send_descriptor(&self, desc: &TensorIpcDescriptor) -> Result<(), LlmZipError> {
        let payload = serde_json::to_vec(desc).map_err(|e| LlmZipError::ClusterError {
            target: "ipc".to_string(),
            reason: format!("Failed to serialize descriptor: {}", e),
        })?;
        let mut buf = BytesMut::with_capacity(4 + payload.len());
        buf.put_u32(payload.len() as u32);
        buf.put_slice(&payload);

        let mut stream = self.stream.lock().await;
        stream.write_all(&buf).await.map_err(LlmZipError::Io)?;
        stream.flush().await.map_err(LlmZipError::Io)?;
        Ok(())
    }

    pub async fn send_payload(&self, data: &[u8]) -> Result<(), LlmZipError> {
        let mut stream = self.stream.lock().await;
        stream.write_all(data).await.map_err(LlmZipError::Io)?;
        stream.flush().await.map_err(LlmZipError::Io)?;
        Ok(())
    }
}

impl ClusterIpcReceiver {
    pub async fn recv_descriptor(&self) -> Result<TensorIpcDescriptor, LlmZipError> {
        let mut stream = self.stream.lock().await;
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await.map_err(LlmZipError::Io)?;
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).await.map_err(LlmZipError::Io)?;

        serde_json::from_slice(&payload).map_err(|e| LlmZipError::ClusterError {
            target: "ipc".to_string(),
            reason: format!("Failed to deserialize descriptor: {}", e),
        })
    }

    pub async fn recv_payload(&self, size: usize) -> Result<Vec<u8>, LlmZipError> {
        let mut stream = self.stream.lock().await;
        let mut buf = vec![0u8; size];
        stream.read_exact(&mut buf).await.map_err(LlmZipError::Io)?;
        Ok(buf)
    }
}
