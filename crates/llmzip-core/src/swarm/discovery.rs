use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SwarmPeer {
    pub peer_id: String,
    pub address: String,
}

pub struct PeerDiscovery {
    peers: Arc<RwLock<Vec<SwarmPeer>>>,
}

impl Default for PeerDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerDiscovery {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a known peer address (e.g. from manual `--peers` CLI flag)
    pub async fn add_peer(&self, peer_id: &str, address: &str) {
        let mut list = self.peers.write().await;
        if !list.iter().any(|p| p.address == address) {
            list.push(SwarmPeer {
                peer_id: peer_id.to_string(),
                address: address.to_string(),
            });
        }
    }

    /// Parse comma-separated list of peer addresses: "127.0.0.1:6881,192.168.1.10:6881"
    pub async fn parse_and_add_peers(&self, peer_list: &str) {
        for (i, addr_str) in peer_list.split(',').enumerate() {
            let addr_str = addr_str.trim();
            if !addr_str.is_empty() {
                self.add_peer(&format!("seed_{}", i), addr_str).await;
            }
        }
    }

    pub async fn get_peers(&self) -> Vec<SwarmPeer> {
        self.peers.read().await.clone()
    }

    /// Formulates a local discovery broadcast payload
    pub fn create_announcement_packet(peer_id: &str, listen_port: u16) -> Vec<u8> {
        format!("LZSW_PEER:{}:{}", peer_id, listen_port).into_bytes()
    }

    /// Parses broadcast announcement packet and extracts peer address
    pub fn parse_announcement_packet(packet: &[u8], sender: SocketAddr) -> Option<SwarmPeer> {
        let text = std::str::from_utf8(packet).ok()?;
        if text.starts_with("LZSW_PEER:") {
            let parts: Vec<&str> = text.split(':').collect();
            if parts.len() == 3 {
                let peer_id = parts[1];
                let port: u16 = parts[2].parse().ok()?;
                let mut peer_addr = sender;
                peer_addr.set_port(port);
                return Some(SwarmPeer {
                    peer_id: peer_id.to_string(),
                    address: peer_addr.to_string(),
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discovery_add_and_parse() {
        let disc = PeerDiscovery::new();
        disc.parse_and_add_peers("127.0.0.1:6881, 10.0.0.2:6882").await;
        let peers = disc.get_peers().await;
        assert_eq!(peers.len(), 2);
        assert_eq!(peers[0].address, "127.0.0.1:6881");
        assert_eq!(peers[1].address, "10.0.0.2:6882");
    }
}
