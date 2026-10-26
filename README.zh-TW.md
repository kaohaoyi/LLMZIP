# LLMZIP

> **零暫存檔串流量化器與模型加載引擎。**  
> 磁碟峰值佔用嚴格等於 **1.0x 最終量化檔大小 + 4 KB `.wal`**，不向本機硬碟寫入任何原始 140 GB 權重暫存。

[![License: AGPL-3.0 / Commercial](https://img.shields.io/badge/License-AGPL%203.0%20%2F%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](crates/llmzip-core/tests)

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

---

## 痛點：AI 邊緣部署的「磁碟膨脹稅」

要在個人電腦或雲端節點上下載並量化一個開源 70B FP16 模型（例如 Llama 3 或 Qwen 2.5），傳統做法需要：

1. **140 GB** 存放從 Hugging Face 下載的原始 `.safetensors` 分片。
2. **140 GB** 用於解包與層級轉換的中間暫存檔（或龐大 RAM 緩衝）。
3. **40 GB** 存放最終量化完成的 `.gguf` 檔案。

**磁碟峰值需求高達 180 GB 至 320 GB。**  
對於配備 256 GB 或 512 GB SSD 的 MacBook、邊緣設備或雲端搶佔式實例，硬碟直接被塞滿報錯（`ENOSPC`）。更致命的是，若在抓取 5 GB 分片時於 99% 處斷線，因為無法驗證串流中的 SHA-256，傳統工具只能將整顆分片重新下載。

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

---

## 最低與推薦設備需求（硬體規格表）

得益於 **「單層環形緩衝池（LayerRingBuffer）」** 與 **「零暫存檔串流」** 機制，LLMZIP 對記憶體與磁碟的要求大幅低於傳統工具，**資源消耗不再隨全模型參數量線性膨脹**：

| 硬體組件 | **最低可使用設備 (Minimum Specs)** | **推薦流暢配置 (Recommended Specs)** | 為什麼能支援這麼低？（底層原理） |
| :--- | :--- | :--- | :--- |
| **CPU 處理器** | **雙核心 x86_64 或 ARM64**<br>• 支援 AVX2 或 ARM NEON（2014 年後 Intel Core / AMD Ryzen / Apple M1 系列） | **4 核心以上現代 CPU**<br>• Apple M1~M4 / Intel 12代+ / AMD Zen3+ | SIMD 向量化相容 AVX2/NEON，雙核心即可跑滿常規頻寬，**不強制依賴 GPU**。 |
| **RAM (記憶體)** | **最低僅需 4 GB RAM**<br>• **即便量化 70B 大模型也僅需 4~8 GB RAM** | **8 GB ~ 16 GB RAM** | **嚴格鎖定單層**：緩衝池只容納當前 1 個 Transformer Layer，邊算邊釋放，記憶體永遠不爆。 |
| **磁碟剩餘空間** | **嚴格等於「目標量化檔大小 + 100 MB 餘裕」**<br>• 0.5B 模型：**僅需 700 MB 空間**<br>• 8B 模型：**僅需 5.5 GB 空間**<br>• 70B 模型：**僅需 45 GB 空間** | **具備相應最終大小之 SSD** | **0 暫存檔技術**：原始 140GB 權重完全不落地，徹底打破傳統 250~320GB 的硬碟門檻。 |
| **網路頻寬** | **10 Mbps 以上**（支援 HTTP Range `206`） | **100 Mbps ~ 1 Gbps 光纖** | 內建 `< 4KB` Micro-WAL，若弱網抖動斷線自動 0-Byte 接續，絕不浪費流量。 |
| **作業系統** | • Windows 10 / 11 x64<br>• macOS 12+ (Apple Silicon / Intel)<br>• Linux (Ubuntu 18.04+, Debian, Arch, Alpine) | 同左 | 獨立二進位檔與標準 C ABI（無 Python/CUDA 龐大依賴環境）。 |

---

## 安裝方式

### 從原始碼編譯

```bash
git clone https://github.com/your-org/llmzip.git
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

## 開發路線圖

- [x] v0.1.0：零暫存檔 SafeTensors 串流引擎，支援 GGUF 與 NVFP4。
- [x] Rayon 多核心並行批次量化與自適應線路調節。
- [x] 密碼學原子提交鎖與 `< 4KB` Micro-WAL 斷點續傳。
- [ ] v0.2.0：原生內建 Hugging Face 多路並行 Range 串流下載器。
- [ ] v0.3.0：提供 Ollama 與 `llama.cpp` 專屬拉取插件（`/models/pull` 邊載邊轉）。

---

## 開源與商用授權

本專案採雙重授權機制（Dual Licensing）：

- **社群開源版（Community Edition）**：遵循 **[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE)**。個人、學術研究與開源社群可完全免費使用。若將 LLMZIP 整合至對外網路服務（如 SaaS、雲端 API、大模型託管服務）或發布修改版本，必須依 AGPLv3 規定公開其完整原始碼。
- **商業與企業授權（Commercial / Enterprise License）**：專為希望將 LLMZIP 整合進內部商業平台、專有閉源產品、雲端託管算力基礎設施，且不願受 AGPLv3 原始碼公開義務約束的企業提供。

如需商業授權、定製整合方案或企業 SLA 支援，請洽詢維護者或透過 [GitHub Discussions](https://github.com/kaohaoyi/LLMZIP/discussions) 發起商業洽談。
