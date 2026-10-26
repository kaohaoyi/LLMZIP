# LLMZIP

> **零暂存文件流式量化器与模型加载引擎。**  
> 磁盘峰值占用严格等于 **1.0x 最终量化文件大小 + 4 KB `.wal`**，不向本地硬盘写入任何原始 140 GB 权重暂存。

[![License: AGPL-3.0 / Commercial](https://img.shields.io/badge/License-AGPL%203.0%20%2F%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](crates/llmzip-core/tests)

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

---

## 痛点：AI 本地部署的“磁盘膨胀税”

在个人电脑或云端实例上下载并量化一个开源 70B FP16 模型（如 Llama 3 或 Qwen 2.5），传统流程需要：

1. **140 GB** 存放从 Hugging Face 下载的原始 `.safetensors` 分片。
2. **140 GB** 用于解包与转换的中间文件（或庞大内存占用）。
3. **40 GB** 存放最终量化好的 `.gguf` 文件。

**磁盘峰值占用高达 180 GB 至 320 GB。**  
对于配有 256 GB 或 512 GB 固态硬盘的笔记本或云端按量付费实例，直接报错磁盘写满（`ENOSPC`）。更致命的是，若在下载 5 GB 分片时在 99% 断线，由于无法校验流式传输中的 SHA-256，传统工具只能将整颗分片重新下载。

## 解决方案

LLMZIP 仅通过 HTTP Range 请求读取各分片前 8 字节长度与 JSON 头部，便在内存中构建**虚拟张量地址空间（VTAS）**，直接预分配目标 `.gguf` 的确定性二进制布局。权重流入内存后立即通过 SIMD 核心并行量化，直接写入预定磁盘偏移，全程无任何未压缩暂存文件。

```
[ HuggingFace / HTTP 镜像源 ]
              │  HTTP Range 流 (SafeTensors)
              ▼
   ┌───────────────────────┐
   │ 流式顺序 SHA-256 状态机│ ──► 将 112 字节中继状态实时记录至 <4KB .wal
   └───────────────────────┘
              │  原始权重流（完全不落盘）
              ▼
   ┌───────────────────────┐
   │ Rayon 多线程量化引擎  │ ──► 多核并行 Q4_K_M, Q8_0, AWQ, NVFP4, MXFP4
   └───────────────────────┘
              │  量化后紧凑块
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 预分配输出文件 (fallocate / SetEndOfFile)              │
   │ 文件头状态: 'UNVF' ──(LFS SHA-256 全量校验无误)──► 'GGUF'│
   └────────────────────────────────────────────────────────┘
```

- **0 暂存文件**：原始未压缩张量完全不写入本地磁盘。
- **0-Byte 重传续传**：在 `< 4KB` Micro-WAL 中精确保存 FIPS 180-4 SHA-256 寄存器（`H[0..7]`、累计字节数、尾部缓冲）。网络断连使用 HTTP `206 Partial Content` 恢复，不重复下载 1 个字节。
- **密码学提交锁（`UNVF` → `GGUF`）**：未完成或传输中断的文件标记为 `UNVF`。仅当所有分片完全通过 Git-LFS SHA-256 校验后，文件头才原子翻转为 `GGUF`，杜绝任何投毒、篡改或截断模型进入推理引擎。
- **线路自适应流水线（`--line`）**：依据网络吞吐动态调节批处理块大小与 WAL 刷盘频率（`fast`、`standard`、`slow`）。

---

## 真实模型实测数据（Qwen 2.5-0.5B 全量分片）

基于本地 Hugging Face 官方 Snapshot 真实模型（290 个全量张量）实测对比：

| 评测维度 | 传统转换流程 | LLMZIP v0.1.0 (`--line fast`) | 实质优势 |
| :--- | :--- | :--- | :--- |
| **磁盘占用峰值** | 1.56 GB (原始文件 + GGUF) | **614.59 MiB** (仅最终文件) | **节省 61% 磁盘空间** |
| **中间暂存文件** | 942.32 MiB | **0 Bytes** | **完全零磁盘污染** |
| **端到端总耗时** | 约 75 秒 (下载+写盘+读盘+量化) | **27.94 秒** | **交付耗时缩短至 1/2.7** |
| **量化运算吞吐率** | 单线程 (~18 MiB/s) | **33.72 MiB/s** (~282.8 Mbps) | **Rayon 多核并行批处理加速** |
| **中断续传开销** | 需重下整颗分片 (942 MiB) | **0 Bytes** (Micro-WAL 断点续传) | **不浪费多余流量** |
| **模型完整性保证** | 转换完成后需手动比对 | **原子提交锁 (`UNVF` → `GGUF`)** | **残缺文件绝不流入推理环境** |

---

## 最低与推荐设备需求（硬件规格表）

得益于 **“单层环形缓冲池（LayerRingBuffer）”** 与 **“零暂存文件流式”** 架构，LLMZIP 对内存与磁盘的要求大幅低于传统工具，**资源消耗不再随全模型参数量线性膨胀**：

| 硬件组件 | **最低可使用设备 (Minimum Specs)** | **推荐流畅配置 (Recommended Specs)** | 为什么能支持这么低？（底层原理） |
| :--- | :--- | :--- | :--- |
| **CPU 处理器** | **双核心 x86_64 或 ARM64**<br>• 支持 AVX2 或 ARM NEON（2014 年后 Intel Core / AMD Ryzen / Apple M1 系列） | **4 核心以上现代 CPU**<br>• Apple M1~M4 / Intel 12代+ / AMD Zen3+ | SIMD 向量化兼容 AVX2/NEON，双核心即可跑满常规带宽，**不强制依赖 GPU**。 |
| **RAM (内存)** | **最低仅需 4 GB RAM**<br>• **即便量化 70B 大模型也仅需 4~8 GB RAM** | **8 GB ~ 16 GB RAM** | **严格锁定单层**：缓冲池只容纳当前 1 个 Transformer Layer，边算边释放，内存永远不爆。 |
| **磁盘剩余空间** | **严格等于“目标量化文件大小 + 100 MB 余量”**<br>• 0.5B 模型：**仅需 700 MB 空间**<br>• 8B 模型：**仅需 5.5 GB 空间**<br>• 70B 模型：**仅需 45 GB 空间** | **具备相应最终大小之 SSD** | **0 暂存文件技术**：原始 140GB 权重完全不落地，彻底打破传统 250~320GB 的硬盘门槛。 |
| **网络带宽** | **10 Mbps 以上**（支持 HTTP Range `206`） | **100 Mbps ~ 1 Gbps 光纤** | 内置 `< 4KB` Micro-WAL，若弱网抖动断线自动 0-Byte 接续，绝不浪费流量。 |
| **操作系统** | • Windows 10 / 11 x64<br>• macOS 12+ (Apple Silicon / Intel)<br>• Linux (Ubuntu 18.04+, Debian, Arch, Alpine) | 同左 | 独立二进制文件与标准 C ABI（无 Python/CUDA 庞大依赖环境）。 |

---

## 安装方式

### 源码编译

```bash
git clone https://github.com/your-org/llmzip.git
cd llmzip
cargo build --release
```

编译生成文件位于 `target/release/llmzip`（Windows 下为 `llmzip.exe`）。

---

## 快速上手

### 1. 检查远程或本地模型布局
在不加载全量模型权重的前提下，毫秒级解析 SafeTensors 头部并预览字节偏移：

```bash
llmzip inspect model-00001-of-00004.safetensors
```

### 2. 流式量化（零暂存文件）
直接从输入流转换为指定量化格式的 GGUF 文件：

```bash
# 推荐本地推理格式（llama.cpp / Ollama 通用）
llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m --line fast

# 高精度 8 位整数量化
llmzip quantize model.safetensors model-q8_0.gguf --preset q8_0

# NVIDIA Blackwell 次世代 FP4 双层微缩放格式
llmzip quantize model.safetensors model-nvfp4.gguf --preset nvfp4 --sanitize
```

### 3. 验证文件完整性
校验输出文件是否已通过全量密码学校验并解锁：

```bash
llmzip verify model-q4km.gguf
# ✓ VALID: 'model-q4km.gguf' has passed cryptographic commit verification (Magic: 'GGUF').
```

### 4. 查看断点续传日志
若任务被中断，可随时查看 `.wal` 日志状态：

```bash
llmzip wal-info model-q4km.gguf.wal
# 进度: [====================          ] 66.7%
# SHA-256 状态累计字节: 628.12 MiB
```

再次执行相同 `quantize` 命令，将自动自中断点无缝接续，0 字节重复下载。

---

## 线路自适应优化模式（`--line`）

流式吞吐量取决于 CPU 算力与网络输入速度的动态匹配：

| 模式 | 目标环境 | 块缓冲大小 | WAL 落盘频率 | 行为特点 |
| :--- | :--- | :--- | :--- | :--- |
| **`fast`** | 光纤 / 千兆 / 本地磁盘 (>500 Mbps) | **1 MiB** | 批量刷写（每 16 张量） | 释放最大吞吐量，消除磁盘停顿 |
| **`standard`** | 常规家用宽带 (100–300 Mbps) | **256 KiB** | 均衡模式（每 4 张量） | 平衡吞吐量与中断恢复点 |
| **`slow`** | 移动网络 / 弱网 / 边缘终端 (<50 Mbps) | **64 KiB** | 实时同步（每 1 张量） | 确保续传精度，降低 CPU 占用 |
| **`auto`** | 自动探测 | 动态调整 | 动态调整 | 依据输入源自动选取最优策略 |

---

## 支持的量化预设规格

| 预设名称 | 目标格式 | 块内元素数 | 规格特点 |
| :--- | :--- | :--- | :--- |
| **`q4_k_m`** | GGUF Q4_K_M | 256 超级块 | 标准 4 位，注意力层采用 Q6_K 保证精度 |
| **`q8_0`** | GGUF Q8_0 | 32 块 | 8 位对称整数量化，几乎无损动态范围 |
| **`awq`** | AWQ W4G128 | 128 分组 | 激活感知权重量化，内置单层滑动窗口校准 |
| **`nvfp4`** | NVFP4 (Blackwell) | 256 超级块 | 4 位 E2M1 + FP8 E4M3 微缩放比例尺 + FP32 全局缩放 |
| **`mxfp4`** | OCP MXFP4 | 32 微块 | 开放计算项目 (OCP) 4 位标准，搭配 E8M0 共享指数 |
| **`fp8`** | FP8 E4M3FN | 32 块 | 支持极高动态范围，保留离群值特征 |

---

## C / C++ SDK 嵌入接口

提供零依赖的标准 C ABI 头文件 [`include/llmzip.h`](include/llmzip.h) 与动态库，便于直接嵌入 LM Studio、Jan、Ollama 或自定义推理后端：

```c
#include "llmzip.h"

// 检查文件是否已完成密码校验并解除提交锁
int status = llmzip_verify_file("model-q4km.gguf");
if (status == LLMZIP_OK) {
    // 文件安全无误，可直接通过 mmap 启动推理
}
```

---

## 开发路线图

- [x] v0.1.0：零暂存文件 SafeTensors 流式引擎，支持 GGUF 与 NVFP4。
- [x] Rayon 多核并行批处理量化与自适应线路调节。
- [x] 密码学原子提交锁与 `< 4KB` Micro-WAL 断点续传。
- [ ] v0.2.0：原生内置 Hugging Face 多路并行 Range 流式下载器。
- [ ] v0.3.0：提供 Ollama 与 `llama.cpp` 专属拉取插件（`/models/pull` 边下边转）。

---

## 开源与商用授权

本项目采用双重授权机制（Dual Licensing）：

- **社区开源版（Community Edition）**：遵循 **[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE)**。个人、学术研究与开源社区可完全免费使用。若将 LLMZIP 集成至对外网络服务（如 SaaS、云端 API、大模型托管服务）或发布修改版本，必须依 AGPLv3 规定公开其完整源代码。
- **商业与企业授权（Commercial / Enterprise License）**：专为希望将 LLMZIP 集成进内部商业平台、专有闭源产品、云端托管算力基础设施，且不愿受 AGPLv3 源代码公开义务约束的企业提供。

如需商业授权、定制集成方案或企业 SLA 支持，请洽询维护者或通过 [GitHub Discussions](https://github.com/kaohaoyi/LLMZIP/discussions) 发起商业洽谈。
