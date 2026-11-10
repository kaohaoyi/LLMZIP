use bytes::{BufMut, BytesMut};
use serde::{Deserialize, Serialize};
use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use crate::swarm::merkle::ModelManifest;

pub const SWARM_MAGIC: &[u8; 4] = b"LZSW";

/// Wire protocol messages for Flat P2P Swarm
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SwarmMessage {
    Ping,
    Pong,
    AnnounceManifest(ModelManifest),
    RequestChunk { model_id: String, chunk_id: u32 },
    ChunkData { model_id: String, chunk_id: u32, data: Vec<u8> },
    HavePiece { model_id: String, chunk_id: u32 },
    Error(String),
}

/// Tokio TCP length-prefixed framing helper: [MAGIC: 4B][LEN: 4B BE][PAYLOAD]
pub async fn write_frame(stream: &mut TcpStream, msg: &SwarmMessage) -> io::Result<()> {
    let payload = serde_json::to_vec(msg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let mut buf = BytesMut::with_capacity(8 + payload.len());
    buf.put_slice(SWARM_MAGIC);
    buf.put_u32(payload.len() as u32);
    buf.put_slice(&payload);
    stream.write_all(&buf).await?;
    stream.flush().await?;
    Ok(())
}

pub async fn read_frame(stream: &mut TcpStream) -> io::Result<SwarmMessage> {
    let mut header = [0u8; 8];
    stream.read_exact(&mut header).await?;
    
    // Validate magic header
    if &header[0..4] != SWARM_MAGIC {
        // Fallback: check if standard 4-byte BE length prefix was used (interop with raw length prefix)
        let len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let mut rest_of_payload = vec![0u8; len - 4];
        stream.read_exact(&mut rest_of_payload).await?;
        let mut full_payload = Vec::with_capacity(len);
        full_payload.extend_from_slice(&header[4..8]);
        full_payload.extend_from_slice(&rest_of_payload);
        return serde_json::from_slice(&full_payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e));
    }

    let len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await?;
    serde_json::from_slice(&payload).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_message_serde() {
        let msg = SwarmMessage::RequestChunk {
            model_id: "test-model".to_string(),
            chunk_id: 42,
        };
        let bytes = serde_json::to_vec(&msg).unwrap();
        let decoded: SwarmMessage = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(msg, decoded);
    }
}
