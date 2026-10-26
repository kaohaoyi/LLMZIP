# LLMZIP

> **Stream & Quantize Large Language Models with Zero Temp Files.**  
> Peak disk footprint = **1.0x final output size + 4 KB `.wal`**. No intermediate 140 GB weight dumps.

[![License: AGPL-3.0 / Commercial](https://img.shields.io/badge/License-AGPL%203.0%20%2F%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](crates/llmzip-core/tests)

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

---

## The Problem: The "Disk Amplification Tax"

Downloading and quantizing an open-weights 70B FP16 model (such as Llama 3 or Qwen 2.5) on developer hardware traditionally requires:

1. **140 GB** for raw upstream `.safetensors` shards.
2. **140 GB** in RAM or uncompressed intermediate files during layer extraction.
3. **40 GB** for the final quantized `.gguf` file.

**Peak disk space required: 180 GB to 320 GB.**  
On laptops or cloud spot instances with 256 GB or 512 GB SSDs, you simply hit `ENOSPC` and get aborted. Furthermore, if a 5 GB shard drops at 99%, naive streaming forces a complete redownload because shard-level Git-LFS SHA-256 cannot resume mid-stream.

## The Solution

LLMZIP inspects only the 8-byte header prefix and JSON metadata of each `.safetensors` shard over HTTP Range requests. It constructs a **Virtual Tensor Address Space (VTAS)**, pre-allocates the exact byte layout of the target `.gguf` or AWQ container, streams weights directly through bounded SIMD buffers in RAM, writes quantized blocks into their deterministic file offsets, and unlocks the file atomically.

```
[ HuggingFace / HTTP Mirror ]
              │  HTTP Range Stream (SafeTensors)
              ▼
   ┌───────────────────────┐
   │ Streaming SHA-256     │ ──► Tracks 112-byte mid-state in <4KB .wal
   └───────────────────────┘
              │  Raw Weight Bytes (Zero Disk Dumps)
              ▼
   ┌───────────────────────┐
   │ Rayon SIMD Quantizer  │ ──► Multi-threaded Q4_K_M, Q8_0, AWQ, NVFP4, MXFP4
   └───────────────────────┘
              │  Quantized Blocks
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ Pre-allocated Output File (fallocate / SetEndOfFile)   │
   │ Header: 'UNVF' ──(LFS SHA-256 Verified)──► 'GGUF'     │
   └────────────────────────────────────────────────────────┘
```

- **0 Temp Files**: The raw uncompressed tensors never touch local storage.
- **0-Byte Redownload Resumption**: FIPS 180-4 SHA-256 registers (`H[0..7]`, byte count, tail buffer) are journaled in a `< 4KB` Micro-WAL. If your connection drops, resume with HTTP `206 Partial Content`.
- **Cryptographic Commit Lock (`UNVF` → `GGUF`)**: Files remain locked with magic header `UNVF`. Only when every shard matches upstream LFS SHA-256 does the header flip atomically to `GGUF`. No corrupted or poisoned weights ever reach your inference runtime.
- **Line-Adaptive Batch Pipeline**: Dynamically tunes batch chunk sizes and disk flush frequencies based on network bandwidth (`fast`, `standard`, `slow`).

---

## Real Model Benchmark (Qwen 2.5-0.5B Full Shard)

Tested on local machine with full 290 tensors from official Hugging Face snapshots:

| Metric | Traditional Workflow | LLMZIP v0.1.0 (`--line fast`) | Advantage |
| :--- | :--- | :--- | :--- |
| **Peak Disk Footprint** | 1.56 GB (Raw + GGUF) | **614.59 MiB** (1.0x Output) | **-61% disk saved** |
| **Intermediate Temp Files** | 942.32 MiB | **0 Bytes** | **Zero disk pollution** |
| **End-to-End Elapsed Time** | ~75s (Download + Write + Read + Quant) | **27.94s** | **~2.7x faster turnaround** |
| **Quantization Throughput** | Single-threaded (~18 MiB/s) | **33.72 MiB/s** (~282.8 Mbps) | **Rayon parallel batch SIMD** |
| **Interrupted Resume Overhead** | Full shard restart (942 MiB) | **0 Bytes** (Micro-WAL HTTP 206) | **Zero wasted bandwidth** |
| **Integrity Guarantee** | Manual hash check post-quant | **Cryptographic Lock (`UNVF` → `GGUF`)** | **Hardware-safe, zero corrupt files** |

---

## System Requirements (Minimum vs Recommended)

Thanks to the **Layer Ring Buffer** (`LayerRingBuffer`) and **Zero-Temp Streaming** architecture, LLMZIP's memory and disk requirements are strictly bounded by a single Transformer layer rather than total model parameter size:

| Hardware Component | **Minimum Requirements** | **Recommended Specs** | Why So Low? (Architectural Basis) |
| :--- | :--- | :--- | :--- |
| **CPU** | **Dual-core x86_64 or ARM64**<br>• AVX2 or ARM NEON (Intel Core 2014+, AMD Ryzen, Apple M1+) | **4+ core modern CPU**<br>• Apple M1–M4, Intel 12th+, AMD Zen3+ | SIMD vectorization saturates standard bandwidth on CPU. **No GPU required.** |
| **RAM** | **4 GB RAM minimum**<br>• Quantizing 70B models requires only **4–8 GB RAM** | **8 GB – 16 GB RAM** | **Single-layer bounded**: RAM only holds 1 Transformer layer at any time, freeing weights as layers complete. |
| **Disk Space** | **Target Output File Size + 100 MB buffer**<br>• 0.5B model: **~700 MB free**<br>• 8B model: **~5.5 GB free**<br>• 70B model: **~45 GB free** | **NVMe SSD matching output size** | **0 Temp Files**: Raw 140 GB weights never touch disk, breaking the 300 GB disk requirement forever. |
| **Network** | **10 Mbps+** (HTTP Range `206` support) | **100 Mbps – 1 Gbps Fiber** | `< 4KB` Micro-WAL resumes on drops with 0 bytes re-downloaded. |
| **Operating System** | • Windows 10/11 x64<br>• macOS 12+ (Apple Silicon / Intel)<br>• Linux (Ubuntu 18.04+, Debian, Arch, Alpine) | Same as minimum | Standalone binary & clean C ABI. Zero Python or CUDA dependency burdens. |

---

## Installation

### From Source

```bash
git clone https://github.com/your-org/llmzip.git
cd llmzip
cargo build --release
```

Binary will be located at `target/release/llmzip` (or `llmzip.exe` on Windows).

---

## Quickstart

### 1. Inspect Remote or Local Shard Layout
Inspect the SafeTensors header without allocating model weights, previewing the planned deterministic byte offsets:

```bash
llmzip inspect model-00001-of-00004.safetensors
```

### 2. Stream & Quantize (Zero-Temp Disk Footprint)
Stream from local or mapped network pipes directly into target GGUF with your chosen preset:

```bash
# Recommended for standard consumer inference (llama.cpp / Ollama)
llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m --line fast

# High-precision 8-bit quantization
llmzip quantize model.safetensors model-q8_0.gguf --preset q8_0

# Next-gen NVIDIA Blackwell FP4 two-level microscaling
llmzip quantize model.safetensors model-nvfp4.gguf --preset nvfp4 --sanitize
```

### 3. Verify Output Integrity
Check whether an output file was committed with cryptographically verified weights:

```bash
llmzip verify model-q4km.gguf
# ✓ VALID: 'model-q4km.gguf' has passed cryptographic commit verification (Magic: 'GGUF').
```

### 4. Inspect Crash Recovery Journal
If a job was interrupted, inspect the Micro-WAL journal:

```bash
llmzip wal-info model-q4km.gguf.wal
# Progress: [====================          ] 66.7%
# SHA-256 Mid-State Fed: 628.12 MiB
```

Re-running the same `quantize` command resumes instantly from the exact confirmed byte boundary.

---

## Line-Adaptive Optimization (`--line`)

Streaming speed depends on balancing CPU SIMD execution with network stream arrival:

| Profile | Target Network | Chunk Buffer | WAL Flush Interval | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **`fast`** | Fiber / Gigabit / Local (>500 Mbps) | **1 MiB** | Batched (16 tensors) | Max throughput; minimizes sync pauses |
| **`standard`** | Home Broadband (100–300 Mbps) | **256 KiB** | Balanced (4 tensors) | Default balanced mode |
| **`slow`** | Mobile / Edge / Flaky (<50 Mbps) | **64 KiB** | Immediate (Every tensor) | Maximizes resume safety; throttles CPU |
| **`auto`** | Dynamic detection | Auto-tuned | Auto-tuned | Automatically selects optimal profile |

---

## Supported Quantization Presets

| Preset | Target Format | Elements / Block | Description |
| :--- | :--- | :--- | :--- |
| **`q4_k_m`** | GGUF Q4_K_M | 256 super-block | Standard llama.cpp 4-bit with Q6_K attention layers |
| **`q8_0`** | GGUF Q8_0 | 32 block | 8-bit integer quantization; virtually lossless |
| **`awq`** | AWQ W4G128 | 128 group | Activation-aware linear quantization with local sliding calibration |
| **`nvfp4`** | NVFP4 (Blackwell) | 256 super-block | 4-bit E2M1 elements + FP8 E4M3 micro-scales + FP32 global scale |
| **`mxfp4`** | OCP MXFP4 | 32 micro-block | OCP standard 4-bit E2M1 paired with 1-byte E8M0 exponent |
| **`fp8`** | FP8 E4M3FN | 32 block | FP8 with FP16 per-block scales; maximum dynamic range |

---

## C / C++ SDK Integration

LLMZIP provides a zero-dependency C ABI header [`include/llmzip.h`](include/llmzip.h) and shared library (`libllmzip_ffi.so` / `llmzip_ffi.dll`) for embedding into inference backends, local frontends (LM Studio, Jan), or custom edge loaders.

```c
#include "llmzip.h"

// Check file verification status
int status = llmzip_verify_file("model-q4km.gguf");
if (status == LLMZIP_OK) {
    // File verified and ready for mmap inference
}
```

---

## Roadmap

- [x] v0.1.0: Zero-Temp SafeTensors streaming engine with GGUF & NVFP4 support.
- [x] Multi-threaded Rayon SIMD batch pipeline with line-adaptive tuning.
- [x] Cryptographic Magic Commit Lock & `< 4KB` Micro-WAL crash recovery.
- [ ] v0.2.0: Direct native Hugging Face client integration with parallel Range connections.
- [ ] v0.3.0: Direct plugin integration for Ollama and `llama.cpp` (`/models/pull` on-the-fly streaming quantizer).

---

## License

This project is dual-licensed:

- **Open Source / Community Edition**: Licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE)**. Free for personal, academic, and community use. If you modify or integrate LLMZIP into network-accessible services (SaaS, cloud APIs, hosted inference platforms), you must provide your complete source code under the AGPLv3.
- **Commercial & Enterprise License**: For organizations seeking to embed LLMZIP into proprietary, closed-source software, cloud infrastructure, or commercial SaaS products without AGPLv3 copyleft obligations.

For commercial licensing, custom integration, or enterprise SLA inquiries, please contact the maintainer or open a commercial inquiry via [GitHub Discussions](https://github.com/kaohaoyi/LLMZIP/discussions).
