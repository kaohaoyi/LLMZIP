use bytes::Bytes;
use tokio::net::TcpStream;
use crate::error::LlmZipError;
use crate::swarm::merkle::{ChunkMeta, Chunker, ModelManifest};
use crate::swarm::protocol::{read_frame, write_frame, SwarmMessage};

pub struct SwarmLeecher;

impl SwarmLeecher {
    /// Connects to a peer and pulls a specific chunk by its metadata with in-flight poison verification
    pub async fn fetch_chunk(
        peer_addr: &str,
        model_id: &str,
        meta: &ChunkMeta,
    ) -> Result<Bytes, LlmZipError> {
        let mut stream = TcpStream::connect(peer_addr)
            .await
            .map_err(|e| LlmZipError::SwarmError {
                peer: peer_addr.to_string(),
                reason: format!("Failed to connect to peer: {}", e),
            })?;

        write_frame(
            &mut stream,
            &SwarmMessage::RequestChunk {
                model_id: model_id.to_string(),
                chunk_id: meta.chunk_id,
            },
        )
        .await
        .map_err(|e| LlmZipError::SwarmError {
            peer: peer_addr.to_string(),
            reason: format!("Failed to send chunk request: {}", e),
        })?;

        let response = read_frame(&mut stream)
            .await
            .map_err(|e| LlmZipError::SwarmError {
                peer: peer_addr.to_string(),
                reason: format!("Failed to read chunk response: {}", e),
            })?;

        match response {
            SwarmMessage::ChunkData { data, chunk_id, .. } => {
                if chunk_id != meta.chunk_id {
                    return Err(LlmZipError::SwarmError {
                        peer: peer_addr.to_string(),
                        reason: format!("Mismatched chunk ID in response: expected {}, got {}", meta.chunk_id, chunk_id),
                    });
                }
                // Verify chunk integrity with poison defense
                Chunker::verify_chunk(&data, meta)?;
                Ok(Bytes::from(data))
            }
            SwarmMessage::Error(err_msg) => Err(LlmZipError::SwarmError {
                peer: peer_addr.to_string(),
                reason: err_msg,
            }),
            _ => Err(LlmZipError::SwarmError {
                peer: peer_addr.to_string(),
                reason: "Unexpected protocol response from peer".to_string(),
            }),
        }
    }

    /// Pulls all chunks of a manifest across a pool of peer addresses
    pub async fn pull_model(
        peers: &[String],
        manifest: &ModelManifest,
    ) -> Result<Vec<Bytes>, LlmZipError> {
        if peers.is_empty() {
            return Err(LlmZipError::SwarmError {
                peer: "none".to_string(),
                reason: "No peer addresses available for swarm pull".to_string(),
            });
        }

        let mut collected_chunks = Vec::with_capacity(manifest.chunks.len());
        for (i, meta) in manifest.chunks.iter().enumerate() {
            let peer = &peers[i % peers.len()];
            let chunk_data = Self::fetch_chunk(peer, &manifest.model_id, meta).await?;
            collected_chunks.push(chunk_data);
        }

        Ok(collected_chunks)
    }
}
