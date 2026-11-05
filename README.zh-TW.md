# LLMZIP

> **零暫存檔串流量化器與模型加載引擎。**  
> 終結社群的 **「雙重空間困境（Double Space Dilemma）」**：磁碟峰值佔用嚴格等於 **1.0x 最終量化檔大小 + 4 KB `.wal`**，不向本機硬碟寫入任何原始 140 GB 權重暫存。

[![License: AGPL-3.0 / Commercial](https://img.shields.io/badge/License-AGPL%203.0%20%2F%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen.svg)](crates/llmzip-core/tests)

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

---

## 痛點：終結「雙重空間困境（Double Space Dilemma）」

要在個人電腦（如 256GB / 512GB MacBook）或雲端節點上下載並量化一個開源 70B FP16 模型（例如 Llama 3 或 Qwen 2.5），傳統做法必然陷入社群最痛恨的 **「Double Space Dilemma（雙重空間困境）」**：

1. **140 GB** 存放從 Hugging Face 下載的原始 `.safetensors` 分片。
2. **140 GB** 用於解包與層級轉換的中間暫存檔（或龐大 RAM 緩衝）。
3. **40 GB** 存放最終量化完成的 `.gguf` 檔案。

**磁碟峰值需求高達 180 GB 至 320 GB。**  
這意味著即使本機硬體算力或記憶體充足，也硬生生被儲存空間門檻擋在門外（`ENOSPC` 磁碟耗盡崩潰）。更致命的是，若在抓取 5 GB 分片時於 99% 處斷線，因為無法驗證串流中的 SHA-256，傳統工具只能將整顆分片重新下載。

## 解決方案

LLMZIP 僅需藉由 HTTP Range 讀取各分片前 8 位元組長度與 JSON 標頭，即可在記憶體中構建**虛擬張量位址空間（VTAS）**，直接預先配置最終 `.gguf` 的確定性二進位佈局。權重流入記憶體後立即經由 SIMD 核心完成量化，直接寫入預定磁碟位移，全程無任何未壓縮暫存檔。

```
[ HuggingFace / HTTP 鏡像源 ]
              │  HTTP Range 串流 (SafeTensors)
              ▼
   ┌───────────────────────┐
   │ 串流循序 SHA-256 狀態機 │ ──► 將 112 位元組中繼狀態即時記錄至 <4KB .wal
   └───────────────────────┘
              │  原始權重流（完全不落盤）
              ▼
   ┌───────────────────────┐
   │ Rayon 多執行緒量化引擎 │ ──► 多核心並行 Q4_K_M, Q8_0, AWQ, NVFP4, MXFP4
   └───────────────────────┘
              │  量化後緊湊區塊
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 預配置輸出檔 (fallocate / SetEndOfFile)                │
   │ 檔頭狀態: 'UNVF' ──(LFS SHA-256 全量校驗無誤)──► 'GGUF'  │
   └────────────────────────────────────────────────────────┘
```

- **0 暫存檔**：原始未壓縮張量完全不寫入本地磁碟。
- **0-Byte 重傳續傳**：在 `< 4KB` Micro-WAL 中精確儲存 FIPS 180-4 SHA-256 暫存器（`H[0..7]`、累計位元組計數、尾端緩衝）。斷線重連使用 HTTP `206 Partial Content` 繼續傳輸，不重複下載 1 位元組。
- **密碼學提交鎖（`UNVF` → `GGUF`）**：未完成或傳輸中斷的檔案標記為 `UNVF`。僅當所有分片完全通過 Git-LFS SHA-256 校驗後，檔頭才原子翻轉為 `GGUF`，杜絕任何投毒、篡改或截斷模型進入推論系統。
- **線路自適應流水線（`--line`）**：依據網路環境動態調整批次區塊大小與 WAL 落盤頻率（`fast`、`standard`、`slow`）。

---

## 真實模型實測數據（Qwen 2.5-0.5B 全量分片）

於本機針對官方 Hugging Face 快取之真實模型（290 個全量張量）實測對比：

| 評測維度 | 傳統轉換流程 | LLMZIP v0.1.0 (`--line fast`) | 實質差距 |
| :--- | :--- | :--- | :--- |
| **磁碟佔用峰值** | 1.56 GB (原始檔 + GGUF) | **614.59 MiB** (僅最終檔) | **節省 61% 磁碟空間** |
| **中間暫存檔案** | 942.32 MiB | **0 Bytes** | **完全零磁碟污染** |
| **端到端總耗時** | 約 75 秒 (下載+寫盤+讀盤+量化) | **27.94 秒** | **整體交付縮短至 1/2.7** |
| **量化運算吞吐量** | 單執行緒 (~18 MiB/s) | **33.72 MiB/s** (~282.8 Mbps) | **Rayon 多核心並行批次運算** |
| **中斷續傳開銷** | 需重抓整顆分片 (942 MiB) | **0 Bytes** (Micro-WAL 斷點續傳) | **不浪費多餘流量** |
| **模型完整性保證** | 轉檔完成後需手動比對 | **原子提交鎖 (`UNVF` → `GGUF`)** | **殘缺檔案絕不流入推論環境** |

*實測硬體環境：Intel Xeon E5-2690 v2 (10 核心 / 20 執行緒 @ 3.00GHz), 64GB RAM, Crucial P3 Plus 2TB NVMe SSD, Rayon SIMD。*

---

## 最低與推薦設備需求（硬體規格表）

得益於 **「單層環形緩衝池（LayerRingBuffer）」** 與 **「零暫存檔串流」** 機制，LLMZIP 對記憶體與磁碟的要求大幅低於傳統工具，**資源消耗不再隨全模型參數量線性膨脹**：

| 硬體組件 | **最低可使用設備 (Minimum Specs)** | **推薦流暢配置 (Recommended Specs)** | 為什麼能支援這麼低？（底層原理） |
| :--- | :--- | :--- | :--- |
| **CPU 處理器** | **雙核心 x86_64 或 ARM64**<br>• 支援 AVX2 或 ARM NEON（2014 年後 Intel Core / AMD Ryzen / Apple M1 系列） | **4 核心以上現代 CPU**<br>• Apple M1~M4 / Intel 12代+ / AMD Zen3+ | SIMD 向量化相容 AVX2/NEON，雙核心即可跑滿常規頻寬，**不強制依賴 GPU**。 |
| **RAM (記憶體)** | **最低僅需 4 GB RAM**<br>• **即便量化 70B 大模型也僅需 4~8 GB RAM** | **8 GB ~ 16 GB RAM** | **嚴格鎖定單層**：緩衝池只容納當前 1 個 Transformer Layer，邊算邊釋放，記憶體永遠不爆。 |
| **磁碟剩餘空間** | **嚴格等於「目標量化檔大小 + 100 MB 餘裕」**<br>• 0.5B 模型：**僅需 700 MB 空間**<br>• 8B 模型：**僅需 5.5 GB 空間**<br>• 70B 模型：**僅需 45 GB 空間** | **具備相應最終大小之 SSD** | **0 暫存檔技術**：原始 140GB 權重完全不落地，徹底打破傳統 250~320GB 的硬碟門檻。 |
| **網路頻寬** | **10 Mbps 以上**（支援 HTTP Range `206`） | **100 Mbps ~ 1 Gbps 光纖** | 內建 `< 4KB` Micro-WAL，若弱網抖動斷線自動 0-Byte 接續，絕不浪費流量。 |
| **作業系統** | • Windows 10 / 11 (x64 / ARM64 Snapdragon X)<br>• macOS 12+ (Apple Silicon / Intel)<br>• Linux (Ubuntu 18.04+, Debian, Arch, Alpine) | 同左 | 獨立二進位檔與標準 C ABI（無 Python/CUDA 龐大依賴環境）。 |

---

## 安裝方式

### 從原始碼編譯

```bash
git clone https://github.com/kaohaoyi/LLMZIP.git
cd llmzip
cargo build --release
```

編譯產物位於 `target/release/llmzip`（Windows 下為 `llmzip.exe`）。

---

## 快速上手

### 1. 檢視遠端或本機模型佈局
在不載入全量模型權重的情況下，秒級解析 SafeTensors 標頭並預覽最終位元組分佈：

```bash
llmzip inspect model-00001-of-00004.safetensors
```

### 2. 串流量化（零暫存檔）
直接從輸入流轉為指定量化格式的 GGUF 檔案：

```bash
# 推薦本地推論格式（llama.cpp / Ollama 通用）
llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m --line fast

# 高精度 8 位元整數量化
llmzip quantize model.safetensors model-q8_0.gguf --preset q8_0

# NVIDIA Blackwell 次世代 FP4 雙層微縮放格式
llmzip quantize model.safetensors model-nvfp4.gguf --preset nvfp4 --sanitize
```

### 3. 驗證檔案完整性
檢查輸出檔案是否已通過全量密碼學校驗並解鎖：

```bash
llmzip verify model-q4km.gguf
# ✓ VALID: 'model-q4km.gguf' has passed cryptographic commit verification (Magic: 'GGUF').
```

### 4. 檢視斷點續傳日誌
若作業遭中斷，可隨時檢查 `.wal` 日誌狀態：

```bash
llmzip wal-info model-q4km.gguf.wal
# 進度: [====================          ] 66.7%
# SHA-256 狀態累計位元組: 628.12 MiB
```

再次執行相同 `quantize` 指令，將自動自中斷點無縫接續，0 位元組重複下載。

---

## 線路自適應優化模式（`--line`）

量化吞吐量關鍵在於 CPU 算力與網路流入速度的動態平衡：

| 模式 | 目標環境 | 區塊緩衝大小 | WAL 落盤頻率 | 行為特點 |
| :--- | :--- | :--- | :--- | :--- |
| **`fast`** | 光纖 / 千兆 / 本地磁碟 (>500 Mbps) | **1 MiB** | 批次刷寫（每 16 張量） | 釋放最大吞吐量，消除磁碟停頓 |
| **`standard`** | 常規家用寬頻 (100–300 Mbps) | **256 KiB** | 均衡模式（每 4 張量） | 平衡吞吐量與中斷保護點 |
| **`slow`** | 行動網路 / 弱網 / 邊緣終端 (<50 Mbps) | **64 KiB** | 即時同步（每 1 張量） | 確保續傳精確度，降低 CPU 佔用 |
| **`auto`** | 自動評估 | 動態調整 | 動態調整 | 依輸入源自動選擇最優方案 |

---

## 雲端串流與生態系統擴展 (v0.2.0)

### 1. S3 / Cloudflare R2 串流直傳
直接串流量化字節至雲端物件儲存，零本機暫存檔，RAM 記憶體嚴格限制在 16MB 分段緩衝：

```bash
# 直接串流至 AWS S3
llmzip quantize model.safetensors s3://my-bucket/models/llama-3-8b-q4km.gguf --preset q4_k_m

# 串流至 Cloudflare R2 / MinIO
llmzip quantize model.safetensors s3://r2-bucket/models/qwen-7b.gguf \
  --preset q4_k_m \
  --s3-endpoint https://<account_id>.r2.cloudflarestorage.com \
  --s3-part-size 16MB
```

### 2. SafeTensors 雙向逆向轉出
將量化後之 GGUF / AWQ 權重逆向反量化轉出為符合 HuggingFace 規範的分片 `.safetensors` 與 `model.safetensors.index.json`：

```bash
llmzip dequantize model-q4km.gguf ./hf_model_dir --dtype bf16 --max-shard-size 5GB
```

### 3. 官方 Python SDK (`llmzip-py`)
內建 PyO3 高效綁定，在重度 SIMD 與串流時自動釋放 Python GIL：

```bash
pip install llmzip
```

```python
import llmzip

# 預檢模型結構與預估大小
info = llmzip.inspect("model.safetensors")
print(f"預估 Q4_K_M 尺寸: {info['predicted_sizes']['q4_k_m']} 位元組")

# 串流量化並接收即時進度回調（GIL 自動釋放）
def on_progress(event: llmzip.ProgressEvent):
    print(f"進度: {event.bytes_processed}/{event.total_bytes} 位元組 ({event.speed_mbps:.1f} MB/s)")

llmzip.quantize_stream(
    "model.safetensors",
    "model-q4km.gguf",
    preset="q4_k_m",
    progress_callback=on_progress,
)
```

### 4. 官方 Docker 容器
在隔離環境中執行 LLMZIP，無須在本機安裝 Rust 工具鏈：

```bash
docker run --rm -v $(pwd):/workspace llmzip:latest \
  quantize model.safetensors model-q4km.gguf --preset q4_k_m
```

---

## 原生 Hugging Face 多路並行串流與本地推論插件 (v0.3.0)

### 1. 原生 Hugging Face 多路並行 Range 串流下載
純 Rust 原生連線（免安裝 Python、免裝 `git-lfs`），支援多路 HTTP Range 併發拉取遠端模型分塊，邊下載邊進入記憶體 SIMD 量化管線：

```bash
# 指定 8 路並行、16MB 分塊粒度直出 GGUF
llmzip quantize --hf Qwen/Qwen2.5-7B qwen-q4km.gguf --concurrency 8 --chunk-size 16M --preset q4_k_m

# 支援私人/Gated 模型安全認證（Token 嚴格脫敏）
HF_TOKEN="hf_xxx" llmzip quantize --hf meta-llama/Llama-3.1-8B-Instruct llama3-q4km.gguf
```

### 2. 本地推論框架一鍵拉取插件 (`llmzip pull`)
一鍵拉取、串流量化並自動注入本機主流推論環境，全程零中間暫存檔落盤：

```bash
# 直接寫入 Ollama Blob 倉庫 (~/.ollama/models/blobs) 並自動註冊 Manifest
llmzip pull --target ollama Qwen/Qwen2.5-7B --preset q4_k_m
ollama run qwen2.5:7b-q4_k_m

# 自動寫入 LM Studio 目錄 (~/.cache/lm-studio/models/)
llmzip pull --target lm-studio meta-llama/Llama-3.1-8B-Instruct

# 自動寫入 Jan 目錄 (~/.jan/models/) 並生成相容 model.json
llmzip pull --target jan mistralai/Mistral-7B-Instruct-v0.3
```

### 3. 透明 Ollama 代理服務模式 (`llmzip serve`)
攔截本機 Ollama 的 `/api/pull` 請求執行邊載邊轉，其餘請求透明轉發至原生 Ollama (11434)：

```bash
llmzip serve --ollama-proxy --port 11435 --upstream-port 11434
```

### 4. llama.cpp 專屬具名管道（Named Pipe / Unix FIFO）串流輸出
支援 Windows 具名管道與 Linux/macOS Unix FIFO，模型邊載邊轉直通 `llama-cli`，磁碟零落盤推論：

```bash
# 終端機 1：啟動串流量化直灌具名管道
llmzip quantize --hf Qwen/Qwen2.5-7B pipe://llmzip_model --preset q4_k_m

# 終端機 2：llama-cli 直接自管道串流加載推論
llama-cli -m \\.\pipe\llmzip_model -p "你好！"  # Windows
# 或
llama-cli -m /tmp/llmzip_model -p "你好！"       # Linux / macOS
```

### 5. DeepSeek MLA 與 Strata MoE 分層專家生態適配
深度整合前沿大尺度 MoE 推論架構：

```bash
# 自動切分 256 路由專家二進制區塊，適配 Strata 記憶體/SSD 分層推論引擎
llmzip pull --target strata deepseek-ai/DeepSeek-V3

# 保留 MLA 投影張量並生成 SGLang 相容配置檔
llmzip pull --target sglang deepseek-ai/DeepSeek-V3
```

### 6. Python SDK `pull` 介面
```python
import llmzip

# 一鍵直拉並量化至 Ollama
llmzip.pull("Qwen/Qwen2.5-7B", target="ollama", preset="q4_k_m", concurrency=8)
```

---

## 支援的量化預設規格

| 預設名稱 | 目標格式 | 區塊元素數 | 規格特點 |
| :--- | :--- | :--- | :--- |
| **`q4_k_m`** | GGUF Q4_K_M | 256 超級區塊 | 標準 4 位元，注意力層採用 Q6_K 保證精準度 |
| **`q8_0`** | GGUF Q8_0 | 32 區塊 | 8 位元對稱整數量化，幾乎無損動態範圍 |
| **`awq`** | AWQ W4G128 | 128 群組 | 激活感知權重量化，內建單層滑動視窗校準 |
| **`nvfp4`** | NVFP4 (Blackwell) | 256 超級區塊 | 4 位元 E2M1 + FP8 E4M3 微縮放比例尺 + FP32 全域縮放 |
| **`mxfp4`** | OCP MXFP4 | 32 微區塊 | 開放運算計畫 (OCP) 4 位元標準，搭配 E8M0 共享指數 |
| **`fp8`** | FP8 E4M3FN | 32 區塊 | 支援極高動態範圍，保留離群值特徵 |

---

## C / C++ SDK 嵌入接口

提供零依賴的標準 C ABI 標頭檔 [`include/llmzip.h`](include/llmzip.h) 及動態庫，便於直接整合進 LM Studio、Jan、Ollama 或自訂推論系統：

```c
#include "llmzip.h"

// 檢查檔案是否已完成密碼校驗並解除提交鎖
int status = llmzip_verify_file("model-q4km.gguf");
if (status == LLMZIP_OK) {
    // 檔案安全無誤，可直接透過 mmap 啟動推論
}
```

---

## ⚡ 推論速度會變慢嗎？（Zero Runtime Overhead）

**完全不會。推論吞吐量（Tokens/sec）與官方預先量化好的模型 100% 相同。**

LLMZIP 是**模型加載與轉檔工具**，而非推論時常駐的中介代理層（Proxy/Middleware）。一旦就地量化完成並解除提交鎖：
1. **產物符合官方標準規範**：產出的 GGUF / AWQ 遵循官方標準規格（GGUF v3、32-Byte 記憶體對齊），推論引擎可直接透過零拷貝（`mmap Zero-Copy`）高頻寬載入權重。
2. **推論執行期由目標框架主導**：推論速度完全由 Ollama、vLLM、SGLang、llama.cpp 自身的 CUDA / CPU SIMD 核心算子決定，LLMZIP 在轉檔後已完全退出，不會佔用任何背景 CPU、GPU 或記憶體。

### 如何自行驗證推論吞吐？（以 Ollama 為例）

在安裝好 Ollama 的環境下，你可以直接透過內建的 `--verbose` 評測指標驗證：

```bash
# 1. 建立 Modelfile 指向 LLMZIP 產出的 GGUF
echo "FROM ./model-q4km.gguf" > Modelfile
ollama create my-model -f Modelfile

# 2. 啟動並印出推論評測報告
ollama run my-model "Why is the sky blue?" --verbose
```

輸出尾端將顯示精確的推論吞吐率：
```text
prompt eval rate:     1138.21 tokens/s    # 首字提示詞預填吞吐量 (Prompt Processing)
eval rate:            90.14 tokens/s      # 自回歸解碼生成吞吐量 (Text Generation)
```
此數值與直接自 Hugging Face 下載的官方預量化檔案在相同硬體上毫無二致。

---

## AI Agent 自動化導入（Cursor / Claude Code / Windsurf / AutoGPT）

LLMZIP 提供 **Agent-First 規範**，專為自主 Coding Agent 與 MLOps 自動化管線設計。

### 1. Agent 一鍵無痛安裝（無須 Rust 環境）

```bash
# Linux / macOS（自動偵測平台並下載預編譯二進位檔）
curl -fsSL https://raw.githubusercontent.com/kaohaoyi/LLMZIP/main/install.sh | sh

# Windows PowerShell
irm https://raw.githubusercontent.com/kaohaoyi/LLMZIP/main/install.ps1 | iex

# Cargo 備援安裝
which llmzip || cargo install llmzip-cli --locked
```

# 自動化靜默量化（支援 Exit Code 嚴格合約，無互動阻礙）
llmzip quantize <source.safetensors> <target.gguf> --preset q4_k_m --line auto --quiet

# 原子完整性校驗（回傳碼 0 = 完整 GGUF；非 0 = 損毀或未完成的 UNVF 檔）
llmzip verify <target.gguf>

# 啟動原生 MCP 伺服器（供 Claude Desktop / Cursor / Zed 連線）
llmzip mcp
```

### 2. Model Context Protocol (MCP) 伺服器配置

LLMZIP 原生內建零依賴 MCP 伺服器（`llmzip mcp`），可直接加入至 `claude_desktop_config.json` 或 Cursor MCP 設定：

```json
{
  "mcpServers": {
    "llmzip": {
      "command": "llmzip",
      "args": ["mcp"]
    }
  }
}
```

向 Agent 暴露 3 大安全沙箱工具：
- `llmzip_inspect`：免載入快速預覽 SafeTensors 頭資訊與 GGUF 記憶體排程。
- `llmzip_quantize`：受路徑穿越與磁碟安全防護保護的零暫存串流量化。
- `llmzip_verify`：原子提交鎖完整性驗證。

### 3. 給 Agent 的提示詞規則（可加入 `AGENTS.md` 或 `.cursorrules`）

```markdown
### 規則：模型下載與量化轉換
當需要從 Hugging Face 下載模型權重或轉換為 GGUF 時，請優先使用 `llmzip` 替代傳統 git-lfs 或 python 轉換腳本，避免硬碟空間耗盡（Double Space Dilemma）。
- 執行指令：`llmzip quantize <src> <dst> --preset q4_k_m --line auto --quiet`
- 驗證指令：`llmzip verify <dst>`
- 保證：峰值磁碟佔用嚴格等於 1.0x 輸出大小，0 暫存檔。
```

---

## 開發路線圖

- [x] **v0.1.0**：零暫存檔 SafeTensors 串流引擎，支援 GGUF 與 NVFP4。
  - Rayon 多核心並行批次量化與自適應線路調節。
  - 密碼學原子提交鎖與 `< 4KB` Micro-WAL 斷點續傳。
- [x] **v0.2.0**：雲端物件儲存、SafeTensors 逆向轉出與生態延伸。
  - S3 / Cloudflare R2 / MinIO 零暫存串流直寫 (`s3://`) 與 RAII Abort 自動守衛。
  - SafeTensors 雙向逆向反量化轉出 (`llmzip dequantize`) 與多分片索引生成。
  - 官方 Python SDK (`llmzip-py` via PyO3)，串流期間主動釋放 GIL。
  - 官方輕量多階段 Docker 映像檔與 GitHub Actions 多平臺 Wheel 自動建置。
- [x] **v0.3.0**：原生 Hugging Face 多路並行 Range 串流下載器與推論生態插件（Ollama / llama.cpp / DeepSeek / Strata）。
  - 原生 Hugging Face 多路並行 Range 串流下載器（4~8 路非阻塞連線，分塊循序重組）。
  - 本地推論框架一鍵拉取插件：支援 Ollama（CLI 直調 + Proxy 服務模式）、LM Studio、Jan 自動注入。
  - llama.cpp 專屬具名管道（Windows Named Pipe / Unix FIFO）零磁碟落盤即時推論。
  - DeepSeek 原生推論（MLA 注意力 + SGLang）與 Strata MoE 分層專家卸載引擎原生適配。
- [ ] **v0.4.0**：生產級雲端伺服器推論叢集與 P2P 分發。
  - vLLM 與 SGLang 伺服器級生產叢集原生拉取插件（支援 Ray 多節點／多 GPU 串流分發）。
  - KServe / Triton Inference Server 企業級容器化模型服務即時熱插拔。
  - P2P 蜂群（Swarm / BitTorrent-style）去中心化分散式權重流式分發。

---

## 開源與商用授權

本專案採雙重授權機制（Dual Licensing）：

- **社群開源版（Community Edition）**：遵循 **[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE)**。個人、學術研究與開源社群可完全免費使用。若將 LLMZIP 整合至對外網路服務（如 SaaS、雲端 API、大模型託管服務）或發布修改版本，必須依 AGPLv3 規定公開其完整原始碼。
- **商業與企業授權（Commercial / Enterprise License）**：專為希望將 LLMZIP 整合進內部商業平台、專有閉源產品、雲端託管算力基礎設施，且不願受 AGPLv3 原始碼公開義務約束的企業提供。

如需商業授權、定製整合方案或企業 SLA 支援，請洽詢維護者或透過 [GitHub Discussions](https://github.com/kaohaoyi/LLMZIP/discussions) 發起商業洽談。
