use bytes::Bytes;
use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use crate::error::LlmZipError;
use crate::swarm::merkle::{Chunker, ModelManifest};
use crate::swarm::protocol::{read_frame, write_frame, SwarmMessage};

/// In-memory zero-disk store for seeded model chunks
pub struct SwarmMemoryStore {
    manifests: RwLock<HashMap<String, ModelManifest>>,
    chunks: RwLock<HashMap<(String, u32), Bytes>>,
}

impl Default for SwarmMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmMemoryStore {
    pub fn new() -> Self {
        Self {
            manifests: RwLock::new(HashMap::new()),
            chunks: RwLock::new(HashMap::new()),
        }
    }

    pub async fn insert_manifest(&self, manifest: ModelManifest) {
        self.manifests.write().await.insert(manifest.model_id.clone(), manifest);
    }

    pub async fn get_manifest(&self, model_id: &str) -> Option<ModelManifest> {
        self.manifests.read().await.get(model_id).cloned()
    }

    pub async fn insert_chunk(&self, model_id: &str, chunk_id: u32, data: Bytes) {
        self.chunks.write().await.insert((model_id.to_string(), chunk_id), data);
    }

    pub async fn get_chunk(&self, model_id: &str, chunk_id: u32) -> Option<Bytes> {
        self.chunks.read().await.get(&(model_id.to_string(), chunk_id)).cloned()
    }
}

/// Tokio-based asynchronous TCP P2P Seeder node
pub struct SwarmSeeder {
    pub store: Arc<SwarmMemoryStore>,
}

impl Default for SwarmSeeder {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmSeeder {
    pub fn new() -> Self {
        Self {
            store: Arc::new(SwarmMemoryStore::new()),
        }
    }

    /// Seeds raw model weights in-memory with zero disk temporaries
    pub async fn seed_raw_bytes(&self, model_id: &str, data: &[u8], chunk_size: usize) -> ModelManifest {
        let (manifest, chunks) = Chunker::split(model_id, data, chunk_size);
        self.store.insert_manifest(manifest.clone()).await;
        for (i, chunk) in chunks.into_iter().enumerate() {
            self.store.insert_chunk(model_id, i as u32, chunk).await;
        }
        manifest
    }

    /// Starts the asynchronous TCP seeder server daemon
    pub async fn start(self: Arc<Self>, bind_addr: &str) -> Result<SocketAddr, LlmZipError> {
        let listener = TcpListener::bind(bind_addr).await.map_err(|e| LlmZipError::SwarmError {
            peer: bind_addr.to_string(),
            reason: format!("Failed to bind TCP listener: {}", e),
        })?;
        let local_addr = listener.local_addr().map_err(LlmZipError::Io)?;

        let seeder = Arc::clone(&self);
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _peer_addr)) => {
                        let seeder_clone = Arc::clone(&seeder);
                        tokio::spawn(async move {
                            if let Err(e) = seeder_clone.handle_peer(stream).await {
                                eprintln!("[SwarmSeeder] Connection error: {:?}", e);
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("[SwarmSeeder] Accept error: {:?}", e);
                        break;
                    }
                }
            }
        });

        Ok(local_addr)
    }

    async fn handle_peer(&self, mut stream: TcpStream) -> io::Result<()> {
        loop {
            let msg = match read_frame(&mut stream).await {
                Ok(m) => m,
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            match msg {
                SwarmMessage::Ping => {
                    write_frame(&mut stream, &SwarmMessage::Pong).await?;
                }
                SwarmMessage::AnnounceManifest(manifest) => {
                    self.store.insert_manifest(manifest).await;
                    write_frame(&mut stream, &SwarmMessage::Pong).await?;
                }
                SwarmMessage::RequestChunk { model_id, chunk_id } => {
                    if let Some(bytes) = self.store.get_chunk(&model_id, chunk_id).await {
                        write_frame(
                            &mut stream,
                            &SwarmMessage::ChunkData {
                                model_id,
                                chunk_id,
                                data: bytes.to_vec(),
                            },
                        )
                        .await?;
                    } else {
                        write_frame(
                            &mut stream,
                            &SwarmMessage::Error(format!("Chunk {} not found", chunk_id)),
                        )
                        .await?;
                    }
                }
                SwarmMessage::HavePiece { .. }
                | SwarmMessage::ChunkData { .. }
                | SwarmMessage::Pong
                | SwarmMessage::Error(_) => {}
            }
        }
        Ok(())
    }
}
