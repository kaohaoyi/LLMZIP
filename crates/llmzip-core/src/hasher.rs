/// FIPS 180-4 SHA-256 initial hash constants $H_0^{(0)} \dots H_7^{(0)}$.
const SHA256_INIT_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// FIPS 180-4 SHA-256 round constants $K_0 \dots K_{63}$.
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Exact 112-Byte serializable SHA-256 Mid-State snapshot for Zero-Re-Download crash recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sha256MidState {
    /// 32 bytes: 8 x 32-bit internal hash state words $H_0 \dots H_7$.
    pub state: [u32; 8],
    /// 8 bytes: Total bytes processed so far (including bytes currently in `tail_buf`).
    pub total_bytes_fed: u64,
    /// 8 bytes (u32 + 4B reserved alignment): Number of unaligned trailing bytes in `tail_buf` (`0..64`).
    pub tail_len: u32,
    pub reserved: u32,
    /// 64 bytes: Uncompressed partial 512-bit block buffer.
    pub tail_buf: [u8; 64],
}

impl Sha256MidState {
    pub const BYTE_SIZE: usize = 112;

    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut out = [0u8; Self::BYTE_SIZE];
        for (i, &word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        out[32..40].copy_from_slice(&self.total_bytes_fed.to_le_bytes());
        out[40..44].copy_from_slice(&self.tail_len.to_le_bytes());
        out[44..48].copy_from_slice(&self.reserved.to_le_bytes());
        out[48..112].copy_from_slice(&self.tail_buf);
        out
    }

    pub fn from_bytes(bytes: &[u8; Self::BYTE_SIZE]) -> Option<Self> {
        let mut state = [0u32; 8];
        for i in 0..8 {
            let mut w = [0u8; 4];
            w.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
            state[i] = u32::from_le_bytes(w);
        }
        let mut tb = [0u8; 8];
        tb.copy_from_slice(&bytes[32..40]);
        let total_bytes_fed = u64::from_le_bytes(tb);

        let mut tl = [0u8; 4];
        tl.copy_from_slice(&bytes[40..44]);
        let tail_len = u32::from_le_bytes(tl);
        if tail_len >= 64 {
            return None;
        }

        let mut rv = [0u8; 4];
        rv.copy_from_slice(&bytes[44..48]);
        let reserved = u32::from_le_bytes(rv);

        let mut tail_buf = [0u8; 64];
        tail_buf.copy_from_slice(&bytes[48..112]);

        Some(Self {
            state,
            total_bytes_fed,
            tail_len,
            reserved,
            tail_buf,
        })
    }
}

/// Streaming FIPS 180-4 SHA-256 engine with Mid-State snapshot & restore.
#[derive(Debug, Clone)]
pub struct StreamingSha256 {
    state: [u32; 8],
    total_bytes_fed: u64,
    tail_len: usize,
    tail_buf: [u8; 64],
}

impl Default for StreamingSha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingSha256 {
    pub fn new() -> Self {
        Self {
            state: SHA256_INIT_STATE,
            total_bytes_fed: 0,
            tail_len: 0,
            tail_buf: [0u8; 64],
        }
    }

    /// Restores the hasher from a 112-Byte `Sha256MidState` snapshot.
    pub fn from_midstate(mid: Sha256MidState) -> Self {
        Self {
            state: mid.state,
            total_bytes_fed: mid.total_bytes_fed,
            tail_len: mid.tail_len as usize,
            tail_buf: mid.tail_buf,
        }
    }

    /// Exports the current 112-Byte `Sha256MidState` snapshot without consuming the hasher.
    pub fn export_midstate(&self) -> Sha256MidState {
        Sha256MidState {
            state: self.state,
            total_bytes_fed: self.total_bytes_fed,
            tail_len: self.tail_len as u32,
            reserved: 0,
            tail_buf: self.tail_buf,
        }
    }

    pub fn total_bytes_fed(&self) -> u64 {
        self.total_bytes_fed
    }

    #[inline]
    fn compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&block[i * 4..(i + 1) * 4]);
            w[i] = u32::from_be_bytes(b);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.total_bytes_fed = self.total_bytes_fed.wrapping_add(data.len() as u64);

        if self.tail_len > 0 {
            let needed = 64 - self.tail_len;
            if data.len() < needed {
                self.tail_buf[self.tail_len..self.tail_len + data.len()].copy_from_slice(data);
                self.tail_len += data.len();
                return;
            } else {
                self.tail_buf[self.tail_len..64].copy_from_slice(&data[..needed]);
                Self::compress_block(&mut self.state, &self.tail_buf);
                self.tail_len = 0;
                self.tail_buf = [0u8; 64];
                data = &data[needed..];
            }
        }

        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress_block(&mut self.state, &block);
            data = &data[64..];
        }

        if !data.is_empty() {
            self.tail_buf[..data.len()].copy_from_slice(data);
            self.tail_len = data.len();
        }
    }

    /// Finalizes the SHA-256 digest and returns the 32-byte digest.
    pub fn finalize(&self) -> [u8; 32] {
        let mut state = self.state;
        let mut block = self.tail_buf;
        let mut pos = self.tail_len;

        // Append the '1' bit (0x80 byte)
        block[pos] = 0x80;
        pos += 1;

        if pos > 56 {
            for b in &mut block[pos..64] {
                *b = 0;
            }
            Self::compress_block(&mut state, &block);
            block = [0u8; 64];
        } else {
            for b in &mut block[pos..56] {
                *b = 0;
            }
        }

        let bit_len = self.total_bytes_fed.wrapping_mul(8);
        block[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress_block(&mut state, &block);

        let mut out = [0u8; 32];
        for (i, &w) in state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    /// Finalizes and formats as lowercase 64-character hexadecimal string.
    pub fn finalize_hex(&self) -> String {
        let digest = self.finalize();
        let mut s = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write;
            let _ = write!(&mut s, "{:02x}", b);
        }
        s
    }
}

/// One-shot convenience SHA-256 hex digest helper.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = StreamingSha256::new();
    h.update(data);
    h.finalize_hex()
}

/// IEEE 802.3 CRC32 checksum for Micro-WAL integrity verification.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}
