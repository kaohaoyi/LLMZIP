# LLMZIP

> **임시 파일 생성 없는 스트리밍 LLM 양자화 & 모델 로더.**  
> 커뮤니티의 오랜 골칫거리 **"Double Space Dilemma(이중 디스크 용량 딜레마)"** 종결: 디스크 피크 점유율은 엄격하게 **1.0x 최종 양자화 파일 크기 + 4 KB `.wal`**. 140 GB에 달하는 중간 가중치 덤프가 발생하지 않습니다.

[![License: AGPL-3.0 / Commercial](https://img.shields.io/badge/License-AGPL%203.0%20%2F%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.4.0-green.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](crates/llmzip-core/tests)

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

---

## 해결하고자 하는 문제: "Double Space Dilemma(이중 디스크 용량 딜레마)"

개발자 PC(256GB / 512GB 맥북 등)나 클라우드 인스턴스에서 오픈소스 70B FP16 모델(Llama 3, Qwen 2.5 등)을 직접 다운로드하고 양자화하려면 커뮤니티가 항상 맞닥뜨리는 **"Double Space Dilemma"**에 빠지게 됩니다:

1. Hugging Face에서 내려받는 원본 `.safetensors` 샤드 (**140 GB**)
2. 레이어 추출 및 변환을 위한 압축 해制 임시 파일 또는 거대 RAM 버퍼 (**140 GB**)
3. 최종 양자화 완료된 `.gguf` 파일 (**40 GB**)

**요구되는 디스크 피크 용량: 180 GB ~ 320 GB.**  
하드웨어 연산 능력이나 메모리가 충분하더라도 스토리지 용량 한계 때문에 작업이 강제 중단(`ENOSPC` 디스크 고갈 에러)됩니다. 더욱이 5 GB 크기의 샤드가 99%에서 끊길 경우, 기존 도구는 스트리밍 중 SHA-256을 검증하지 못해 샤드 전체를 처음부터 다시 받아야 했습니다.

## LLMZIP의 해결 방식

LLMZIP은 HTTP Range 요청을 통해 각 `.safetensors` 샤드의 앞부분 8바이트 헤더 크기와 JSON 메타데이터만 먼저 읽어 들입니다. 메모리에 **가상 텐서 주소 공간(VTAS)**을 구축하고, 타겟 `.gguf` 컨테이너의 바이트 레이아웃을 디스크에 사전 할당(fallocate)합니다. 이후 유입되는 가중치는 RAM의 바인딩된 SIMD 버퍼를 통해 즉시 병렬 양자화되며, 결정론적 오프셋 위치로 직접 기록됩니다.

```
[ HuggingFace / HTTP 미러 ]
              │  HTTP Range 스트림 (SafeTensors)
              ▼
   ┌───────────────────────┐
   │ 스트리밍 SHA-256 검증 │ ──► 112바이트 중간 상태를 <4KB .wal 에 실시간 기록
   └───────────────────────┘
              │  원본 가중치 바이트 스트림 (디스크 임시 파일 0)
              ▼
   ┌───────────────────────┐
   │ Rayon SIMD 양자화기   │ ──► 멀티스레드 병렬 Q4_K_M, Q8_0, AWQ, NVFP4, MXFP4
   └───────────────────────┘
              │  양자화된 블록
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 사전 할당된 출력 파일 (fallocate / SetEndOfFile)        │
   │ 헤더 상태: 'UNVF' ──(LFS SHA-256 전량 검증 통과)──► 'GGUF'│
   └────────────────────────────────────────────────────────┘
```

- **임시 파일 0개 (Zero Temp Files)**: 압축 해제된 원본 텐서가 로컬 스토리지에 기록되지 않습니다.
- **0바이트 재다운로드 재개**: FIPS 180-4 SHA-256 레지스터(`H[0..7]`, 누적 바이트 수, 후미 버퍼)를 `< 4KB` 크기의 Micro-WAL에 안전하게 기록합니다. 네트워크가 끊겨도 HTTP `206 Partial Content`로 중단된 지점부터 즉시 이어받습니다.
- **암호학적 커밋 락 (`UNVF` → `GGUF`)**: 전송 중인 파일은 매직 헤더 `UNVF`로 잠겨 있습니다. 모든 샤드가 Git-LFS SHA-256과 정확히 일치할 때만 원자적으로 `GGUF`로 전환됩니다. 손상되거나 변조된 모델이 추론 런타임에 유입되는 위험을 근본적으로 차단합니다.
- **네트워크 회선 적응형 파이프라인 (`--line`)**: 네트워크 속도에 따라 배치 청크 크기와 디스크 플러시 주기를 동적으로 조절합니다 (`fast`, `standard`, `slow`).

---

## 실제 모델 벤치마크 (Qwen 2.5-0.5B 전체 샤드 실측)

로컬 머신에서 공식 Hugging Face 스냅샷 모델(전체 290개 텐서)을 대상으로 실측한 결과입니다:

| 측정 지표 | 기존 변환 방식 | LLMZIP v0.1.0 (`--line fast`) | 실질적 차이 |
| :--- | :--- | :--- | :--- |
| **디스크 피크 점유율** | 1.56 GB (원본 + GGUF) | **614.59 MiB** (최종 파일만) | **디스크 61% 절약** |
| **중간 임시 파일** | 942.32 MiB | **0 Bytes** | **스토리지 오염 없음** |
| **전체 소요 시간 (E2E)** | 약 75초 (다운로드+쓰기+읽기+변환) | **27.94초** | **소요 시간 1/2.7로 단축** |
| **양자화 처리량** | 단일 스레드 (~18 MiB/s) | **33.72 MiB/s** (~282.8 Mbps) | **Rayon 멀티스레드 배치 SIMD** |
| **연결 끊김 복구 비용** | 전체 샤드 재다운로드 (942 MiB) | **0 Bytes** (Micro-WAL 이어받기) | **대역폭 낭비 방지** |
| **무결성 보증** | 변환 후 수동 해시 검증 필요 | **암호학적 락 (`UNVF` → `GGUF`)** | **손상된 파일 유입 완전 차단** |

*실측 하드웨어 환경: Intel Xeon E5-2690 v2 (10코어 / 20스레드 @ 3.00GHz), 64GB RAM, Crucial P3 Plus 2TB NVMe SSD, Rayon SIMD.*

---

## 최소 및 권장 시스템 요구사양 (하드웨어 규격표)

**「단일 레이어 링 버퍼(LayerRingBuffer)」** 및 **「임시 파일 0개 스트리밍」** 아키텍처 덕분에, 메모리 및 디스크 소모량은 전체 모델 파라미터 수에 비례하여 증가하지 않고 레이어 1개 크기로 엄격히 제한됩니다:

| 하드웨어 구성요소 | **최소 동작 요구사양 (Minimum Specs)** | **권장 원활 사양 (Recommended Specs)** | 리소스를 적게 소모하는 이유 (기술적 근거) |
| :--- | :--- | :--- | :--- |
| **CPU 프로세서** | **듀얼코어 x86_64 또는 ARM64**<br>• AVX2 또는 ARM NEON 지원 (2014년 이후 Intel Core / AMD Ryzen / Apple M1 계열) | **4코어 이상 최신 CPU**<br>• Apple M1~M4 / Intel 12세대+ / AMD Zen3+ | SIMD 벡터화 최적화로 CPU만으로 일반 대역폭을 포화 처리합니다. **GPU 불필요.** |
| **RAM (메모리)** | **최소 4 GB RAM**<br>• **70B 대형 모델 양자화 시에도 4~8 GB RAM으로 충분** | **8 GB ~ 16 GB RAM** | **단일 레이어 엄격 제한**: 버퍼는 항상 Transformer 1개 레이어 가중치만 유지하고 즉시 해제하므로 메모리가 고갈되지 않습니다. |
| **디스크 여유 공간** | **「최종 출력 파일 크기 + 100 MB 여유」**<br>• 0.5B 모델: **약 700 MB 필요**<br>• 8B 모델: **약 5.5 GB 필요**<br>• 70B 모델: **약 45 GB 필요** | **최종 파일이 수용되는 NVMe SSD** | **임시 파일 0개 기술**: 원본 140GB 가중치를 디스크에 쓰지 않으므로 기존 250~320GB 공간 요구를 완전히 없앴습니다. |
| **네트워크 대역폭** | **10 Mbps 이상** (HTTP Range `206` 지원 회선) | **100 Mbps ~ 1 Gbps 광랜** | `< 4KB` Micro-WAL을 통해 회선이 끊겨도 0바이트 손실 없이 이어받습니다. |
| **운영체제** | • Windows 10 / 11 x64<br>• macOS 12+ (Apple Silicon / Intel)<br>• Linux (Ubuntu 18.04+, Debian, Arch, Alpine) | 최소 사양과 동일 | 단일 바이너리 및 C ABI 제공 (Python/CUDA 거대 의존성 없음). |

---

## 설치 방법

### 소스코드 빌드

```bash
git clone https://github.com/kaohaoyi/LLMZIP.git
cd llmzip
cargo build --release
```

실행 파일은 `target/release/llmzip` (Windows 환경은 `llmzip.exe`)에 생성됩니다.

---

## 빠른 시작

### 0. Hugging Face에서 로컬 추론 환경으로 원클릭 Pull (v0.3.0)
수십 GB의 원본 파일을 다운로드하지 않고, 다운로드·양자화·추론 등록을 한 번에 실행:

```bash
# Ollama에 직접 풀, 양자화 및 Manifest 등록 (중간 디스크 낭비 0)
llmzip pull --target ollama Qwen/Qwen2.5-7B --preset q4_k_m
ollama run qwen2.5:7b-q4_k_m

# Hugging Face에서 8개 병렬 연결로 직접 GGUF 스트리밍 양자화
llmzip quantize --hf Qwen/Qwen2.5-7B qwen-q4km.gguf --concurrency 8 --preset q4_k_m
```

### 1. 원격 또는 로컬 샤드 구조 확인
가중치 전체를 로드하지 않고 SafeTensors 헤더를 수 밀리초 만에 파싱하여 바이트 배치 계획을 미리 확인합니다:

```bash
llmzip inspect model-00001-of-00004.safetensors
```

### 2. 스트리밍 양자화 실행 (임시 파일 0)
입력 스트림에서 원하는 프리셋의 GGUF 파일로 직접 변환합니다:

```bash
# llama.cpp / Ollama 권장 표준 4비트 포맷
llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m --line fast

# 고정밀 8비트 정수 양자화
llmzip quantize model.safetensors model-q8_0.gguf --preset q8_0

# 엔비디아 블랙웰 차세대 NVFP4 (2단계 마이크로스케일)
llmzip quantize model.safetensors model-nvfp4.gguf --preset nvfp4 --sanitize
```

### 3. 파일 무결성 검증
출력 파일이 전량 암호학적 검증을 통과하여 정상 커밋되었는지 확인합니다:

```bash
llmzip verify model-q4km.gguf
# ✓ VALID: 'model-q4km.gguf' has passed cryptographic commit verification (Magic: 'GGUF').
```

### 4. 복구 저널 상태 확인
작업이 예기치 않게 중단되었을 경우 `.wal` 파일의 진행도를 확인합니다:

```bash
llmzip wal-info model-q4km.gguf.wal
# 진행률: [====================          ] 66.7%
# SHA-256 상태 누적 바이트: 628.12 MiB
```

동일한 `quantize` 명령을 다시 실행하면 중단된 바이트 위치부터 0바이트 낭비 없이 이어받아 완료합니다.

---

## 회선 적응형 프로파일 (`--line`)

스트리밍 처리량은 CPU의 SIMD 연산 속도와 네트워크 유입 속도의 균형에 좌우됩니다:

| 프로파일 | 대상 네트워크 환경 | 청크 버퍼 크기 | WAL 디스크 기록 주기 | 동작 특징 |
| :--- | :--- | :--- | :--- | :--- |
| **`fast`** | 기가비트 / 광랜 / 로컬 (>500 Mbps) | **1 MiB** | 배치 기록 (16개 텐서마다) | 최대 처리량 확보, 디스크 병목 배제 |
| **`standard`** | 일반 가정용 브로드밴드 (100–300 Mbps) | **256 KiB** | 밸런스 모드 (4개 텐서마다) | 처리량과 복구 지점의 균형 |
| **`slow`** | 모바일 / 엣지 / 불안정한 회선 (<50 Mbps) | **64 KiB** | 즉시 동기화 (텐서마다) | 복구 정확도 최우선, CPU 점유 완화 |
| **`auto`** | 자동 감지 | 동적 설정 | 동적 설정 | 환경에 맞춰 최적 설정 자동 선택 |

---

## 클라우드 스트리밍 및 생태계 확장 (v0.2.0)

### 1. S3 / Cloudflare R2 스트리밍 직접 쓰기
로컬 임시 파일을 전혀 생성하지 않고, 클라우드 객체 스토리지로 직접 양자화 바이트 스트리밍 (RAM 버퍼는 16MB로 고정):

```bash
# AWS S3 직접 스트리밍
llmzip quantize model.safetensors s3://my-bucket/models/llama-3-8b-q4km.gguf --preset q4_k_m

# Cloudflare R2 / MinIO 직접 스트리밍
llmzip quantize model.safetensors s3://r2-bucket/models/qwen-7b.gguf \
  --preset q4_k_m \
  --s3-endpoint https://<account_id>.r2.cloudflarestorage.com \
  --s3-part-size 16MB
```

### 2. SafeTensors 역양자화 익스포트
양자화된 GGUF / AWQ 가중치를 HuggingFace 표준 샤딩 `.safetensors` 및 `model.safetensors.index.json`으로 역변환:

```bash
llmzip dequantize model-q4km.gguf ./hf_model_dir --dtype bf16 --max-shard-size 5GB
```

### 3. 공식 Python SDK (`llmzip-py`)
고부하 SIMD 연산 및 네트워크 스트리밍 중 Python GIL을 자동 해제하는 고성능 PyO3 바인딩:

```bash
pip install llmzip
```

```python
import llmzip

# 모델 구조 및 예상 크기 사전 점검
info = llmzip.inspect("model.safetensors")
print(f"예상 Q4_K_M 크기: {info['predicted_sizes']['q4_k_m']} 바이트")

# 스트리밍 양자화 및 실시간 진행률 콜백 (GIL 자동 해제)
def on_progress(event: llmzip.ProgressEvent):
    print(f"진행률: {event.bytes_processed}/{event.total_bytes} 바이트 ({event.speed_mbps:.1f} MB/s)")

llmzip.quantize_stream(
    "model.safetensors",
    "model-q4km.gguf",
    preset="q4_k_m",
    progress_callback=on_progress,
)
```

### 4. 공식 Docker 컨테이너
로컬에 Rust를 설치하지 않고 격리된 환경에서 LLMZIP 실행:

```bash
docker run --rm -v $(pwd):/workspace llmzip:latest \
  quantize model.safetensors model-q4km.gguf --preset q4_k_m
```

---

## 네이티브 Hugging Face 병렬 스트리밍 & 로컬 추론 플러그인 (v0.3.0)

### 1. 네이티브 Hugging Face 병렬 Range 스트리밍 다운로드
순수 Rust 네이티브 연결(Python 및 `git-lfs` 불필요). 다중 HTTP Range 연결을 통해 원격 모델 청크를 병렬 수신하며, 메모리 내에서 직접 SIMD 양자화 수행:

```bash
# 8개 병렬 연결, 16MB 청크 크기로 직접 GGUF 생성
llmzip quantize --hf Qwen/Qwen2.5-7B qwen-q4km.gguf --concurrency 8 --chunk-size 16M --preset q4_k_m

# Gated / 프라이빗 모델용 토큰 인증 지원 (토큰은 로그 및 디버그에서 안전하게 마스킹)
HF_TOKEN="hf_xxx" llmzip quantize --hf meta-llama/Llama-3.1-8B-Instruct llama3-q4km.gguf
```

### 2. 로컬 추론 런타임 원클릭 Pull 플러그인 (`llmzip pull`)
단일 명령으로 다운로드, 스트리밍 양자화, 로컬 추론 환경 자동 등록 완결 (중간 임시 파일 0바이트):

```bash
# Ollama Blob 저장소 (~/.ollama/models/blobs) 에 직접 작성 및 Manifest 자동 등록
llmzip pull --target ollama Qwen/Qwen2.5-7B --preset q4_k_m
ollama run qwen2.5:7b-q4_k_m

# LM Studio 디렉토리 (~/.cache/lm-studio/models/) 로 직접 저장
llmzip pull --target lm-studio meta-llama/Llama-3.1-8B-Instruct

# Jan 디렉토리 (~/.jan/models/) 에 저장하고 호환 model.json 자동 생성
llmzip pull --target jan mistralai/Mistral-7B-Instruct-v0.3
```

### 3. 투명한 Ollama 프록시 서버 모드 (`llmzip serve`)
Ollama의 `/api/pull` 요청을 가로채 즉시 스트리밍 양자화를 진행하고, 나머지 요청은 네이티브 Ollama (11434) 로 투명하게 프록시:

```bash
llmzip serve --ollama-proxy --port 11435 --upstream-port 11434
```

### 4. llama.cpp 전용 명명된 파이프(Named Pipe / Unix FIFO) 스트리밍 출력
Windows Named Pipe 및 Linux/macOS Unix FIFO 지원. 가중치를 다운로드 및 양자화하면서 디스크 쓰기 없이 `llama-cli` 로 직접 파이프 스트리밍:

```bash
# 터미널 1: 스트리밍 양자화 시작 후 파이프로 데이터 공급
llmzip quantize --hf Qwen/Qwen2.5-7B pipe://llmzip_model --preset q4_k_m

# 터미널 2: llama-cli 에서 파이프를 통해 직접 스트리밍 로드 및 추론
llama-cli -m \\.\pipe\llmzip_model -p "안녕하세요!"  # Windows
# 또는
llama-cli -m /tmp/llmzip_model -p "안녕하세요!"       # Linux / macOS
```

### 5. DeepSeek MLA & Strata MoE 계층형 오프로드 연동
최첨단 대규모 MoE 추론 프레임워크와의 네이티브 연동:

```bash
# 256개 라우팅 전문가 바이너리 청크 분할, Strata 계층형 추론 엔진에 즉시 호환
llmzip pull --target strata deepseek-ai/DeepSeek-V3

# MLA 투영 텐서를 보존하고 SGLang 호환 구성 파일 자동 생성
llmzip pull --target sglang deepseek-ai/DeepSeek-V3
```

### 6. Python SDK `pull` 인터페이스
```python
import llmzip

# 단일 명령으로 Ollama에 직접 pull 및 양자화
llmzip.pull("Qwen/Qwen2.5-7B", target="ollama", preset="q4_k_m", concurrency=8)
```

---

## 오픈 클러스터 직접 주입 & 탈중앙화 P2P 스웜 (v0.4.0)

### 1. vLLM & SGLang 제로 디스크 공유 메모리 직접 주입
IPC 및 공유 메모리를 통해 디코딩된 가중치를 vLLM / SGLang 워커 메모리로 직접 주입하여 중간 파일 디스크 저장을 완전히 제거합니다:

```bash
# 텐서 병렬화(TP=4)를 지원하며 vLLM으로 직접 스트리밍 주입
llmzip cluster pull --target vllm Qwen/Qwen2.5-7B --tp 4

# SGLang 추론 엔드포인트로 직접 스트리밍 주입
llmzip cluster pull --target sglang deepseek-ai/DeepSeek-V3 --tp 8
```

### 2. 탈중앙화 플랫 P2P 스웜 분산 (Flat P2P Swarm)
노드 간에 청크를 P2P 방식으로 상호 전송하며, 8MB 실시간 Merkle Root 오염 방지 검증을 내장하여 순수 메모리 스트리밍 및 제로 임시 파일을 달성합니다:

```bash
# 메모리 시딩 노드 실행 (Seeder)
llmzip swarm seed Qwen/Qwen2.5-7B --listen 0.0.0.0:6881 --chunk-size 8M

# 피어 노드들로부터 모델 병렬 수신 (Leecher)
llmzip swarm pull Qwen/Qwen2.5-7B --peers 192.168.1.10:6881,192.168.1.11:6881 --output ./qwen.gguf
```

### 3. Python SDK P2P 스웜 연동
```python
import llmzip

# 메모리 청킹 및 Merkle Tree 루트 해시 계산
manifest_json = llmzip.swarm_seed("Qwen/Qwen2.5-7B", raw_weights_bytes)

# 피어 분할 청크 병렬 수신
status = llmzip.swarm_pull("Qwen/Qwen2.5-7B", ["192.168.1.10:6881", "192.168.1.11:6881"])
```

---

## 지원 양자화 포맷 프리셋

| 프리셋 이름 | 대상 포맷 | 블록 구성 | 특징 |
| :--- | :--- | :--- | :--- |
| **`q4_k_m`** | GGUF Q4_K_M | 256 슈퍼블록 | llama.cpp 표준 4비트, Attention 레이어는 Q6_K 적용으로 정확도 유지 |
| **`q8_0`** | GGUF Q8_0 | 32 블록 | 8비트 대칭 정수 양자화, 원본 대비 손실 거의 없음 |
| **`awq`** | AWQ W4G128 | 128 그룹 | 활성화 인식 선형 양자화, 단일 레이어 슬라이딩 윈도우 보정 내장 |
| **`nvfp4`** | NVFP4 (Blackwell) | 256 슈퍼블록 | 4비트 E2M1 요소 + FP8 E4M3 마이크로스케일 + FP32 글로벌 스케일 |
| **`mxfp4`** | OCP MXFP4 | 32 마이크로블록 | OCP 표준 규격 4비트 E2M1 및 E8M0 공유 지수 |
| **`fp8`** | FP8 E4M3FN | 32 블록 | 넓은 동적 범위 유지, 이상치(outlier) 가중치 보존 |

---

## C / C++ SDK 연동

의존성 없는 표준 C ABI 헤더 [`include/llmzip.h`](include/llmzip.h) 및 라이브러리를 제공하여 LM Studio, Jan, Ollama 또는 자체 엣지 로더에 직접 임베딩할 수 있습니다.

```c
#include "llmzip.h"

// 파일 검증 상태 확인
int status = llmzip_verify_file("model-q4km.gguf");
if (status == LLMZIP_OK) {
    // 검증 완료. mmap을 통한 즉각적인 추론 가능
}
```

---

## ⚡ 추론 속도가 느려지나요? (Zero Runtime Overhead)

**전혀 느려지지 않습니다. 추론 처리량(Tokens/sec)은 공식 사전 양자화 모델과 100% 동일합니다.**

LLMZIP은 **모델 수신 및 변환 도구**이며, 추론 시 상주하는 중간 프록시(Middleware)가 아닙니다. 변환이 완료되고 커밋 락이 해제된 이후에는:
1. **공식 표준 규격 100% 준수**: 생성된 GGUF / AWQ 파일은 공식 표준 사양(GGUF v3, 32바이트 메모리 정렬)을 엄격히 준수하므로, 추론 엔진이 제로카피(`mmap Zero-Copy`)를 통해 최대 대역폭으로 가중치를 로드합니다.
2. **추론 프레임워크 자체 연산 엔진 구동**: 추론 성능은 Ollama, vLLM, SGLang, llama.cpp 자체의 CUDA / CPU SIMD 커널 성능에 의해서만 결정됩니다. LLMZIP은 변환 후 완전히 프로세스가 종료되므로 배경 CPU, GPU 또는 메모리를 전혀 점유하지 않습니다.

### 추론 성능 자체 검증 방법 (Ollama 기준)

Ollama가 설치된 환경에서는 `--verbose` 플래그를 통해 즉시 처리량을 검증할 수 있습니다:

```bash
# 1. LLMZIP으로 생성된 GGUF를 지정하여 Modelfile 생성
echo "FROM ./model-q4km.gguf" > Modelfile
ollama create my-model -f Modelfile

# 2. 벤치마크 프롬프트 실행
ollama run my-model "Why is the sky blue?" --verbose
```

출력 하단에서 정확한 처리량 지표를 확인할 수 있습니다:
```text
prompt eval rate:     1138.21 tokens/s    # 프롬프트 사전 처리 처리량 (Prefill)
eval rate:            90.14 tokens/s      # 자동회귀 토큰 생성 처리량 (Generation)
```
이 수치는 동일한 하드웨어에서 Hugging Face로부터 직접 다운로드한 공식 사전 양자화 모델의 성능과 통계적 오차 범위 내에서 완전히 일치합니다.

---

## AI 에이전트 자동화 연동 가이드 (Cursor / Claude Code / Windsurf / AutoGPT)

LLMZIP은 코딩 에이전트 및 MLOps 자동화 파이프라인을 위해 **Agent-First 계약**을 지원합니다.

### 1. Agent 원라인 자동 설치 (Rust 환경 불필요)

```bash
# Linux / macOS (플랫폼 자동 감지 및 사전 빌드 바이너리 설치)
curl -fsSL https://raw.githubusercontent.com/kaohaoyi/LLMZIP/main/install.sh | sh

# Windows PowerShell
irm https://raw.githubusercontent.com/kaohaoyi/LLMZIP/main/install.ps1 | iex

# Cargo 대체 설치
which llmzip || cargo install llmzip-cli --locked
```

# 비대화형 자동 스트리밍 양자화 (종료 코드 계약 지원)
llmzip quantize <source.safetensors> <target.gguf> --preset q4_k_m --line auto --quiet

# 원자적 무결성 검증 (종료 코드 0 = 검증 완료된 GGUF, 0이 아님 = 손상 또는 미완료 UNVF 파일)
llmzip verify <target.gguf>

# 네이티브 MCP 서버 실행 (Claude Desktop / Cursor / Zed 연동)
llmzip mcp
```

### 2. Model Context Protocol (MCP) 서버 설정

LLMZIP은 외부 의존성이 전혀 없는 네이티브 MCP 서버(`llmzip mcp`)를 기본 탑재하고 있습니다. `claude_desktop_config.json` 또는 Cursor MCP 설정에 아래 내용을 추가하여 즉시 연동할 수 있습니다:

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

에이전트에 노출되는 3대 샌드박스 보안 도구:
- `llmzip_inspect`: SafeTensors 헤더 및 목표 GGUF 레이아웃 사전 분석.
- `llmzip_quantize`: 경로 탐색(Path Traversal) 공격 방어 가드레일이 적용된 무임시파일 스트리밍 양자화.
- `llmzip_verify`: 암호학적 커밋 락 검증.

### 3. 에이전트 지침 규칙 (`AGENTS.md` / `.cursorrules` 권장)

```markdown
### 규칙: 모델 다운로드 및 양자화 변환
Hugging Face에서 모델 가중치를 다운로드하거나 GGUF로 변환할 때는 디스크 고갈(Double Space Dilemma)을 방지하기 위해 기존 git-lfs나 python 스크립트 대신 반드시 `llmzip`을 사용하십시오.
- 실행 명령어: `llmzip quantize <src> <dst> --preset q4_k_m --line auto --quiet`
- 검증 명령어: `llmzip verify <dst>`
- 보장 사항: 피크 디스크 사용량은 엄격히 최종 출력 크기의 1.0x, 임시 파일 0 바이트.
```

---

## 로드맵

- [x] **v0.1.0**: 임시 파일 없는 SafeTensors 스트리밍 코어 엔진 (GGUF 및 NVFP4 지원).
  - Rayon 멀티스레드 SIMD 배치 파이프라인 및 회선 적응형 튜닝.
  - 암호학적 원자적 커밋 락 및 `< 4KB` Micro-WAL 복구 시스템.
- [x] **v0.2.0**: 클라우드 객체 스토리지, SafeTensors 역양자화 익스포트 및 생태계 확장.
  - S3 / Cloudflare R2 / MinIO 직접 스트리밍 쓰기 (`s3://`) 및 RAII Abort 안전 가드.
  - SafeTensors 양방향 역양자화 익스포트 (`llmzip dequantize`) 및 멀티 샤드 인덱싱.
  - 공식 Python SDK (`llmzip-py` via PyO3), 처리 중 GIL 자동 해제 지원.
  - 공식 멀티스테이지 Docker 컨테이너 및 GitHub Actions 크로스 플랫폼 Wheel CI.
- [x] **v0.3.0**: Hugging Face 직접 연결 멀티파트 Range 스트리밍 클라이언트 및 추론 생태계 플러그인 (Ollama / llama.cpp / DeepSeek / Strata).
  - Hugging Face 다중 연결 Range 스트리밍 엔진 (4~8개 비차단 연결 및 순차 재배열 버퍼).
  - 로컬 추론 원클릭 Pull 플러그인: Ollama (CLI 직접 호출 + Proxy 모드), LM Studio, Jan 자동 연동.
  - llama.cpp 전용 명명된 파이프 (Windows Named Pipe / Unix FIFO) 무디스크 실시간 스트리밍 추론.
  - DeepSeek 네이티브 추론 (MLA Attention + SGLang) 및 Strata MoE 계층형 전문가 오프로딩 엔진 연동.
- [x] **v0.4.0**: 오픈 클러스터 직접 주입 & 탈중앙화 P2P 스웜 분산.
  - vLLM 및 SGLang 제로 디스크 공유 메모리 / IPC 스트리밍 주입 (TP=2/4/8 지원).
  - Tokio TCP 바이너리 프레이밍 및 8MB Merkle Root 변조 방지 검증 기능의 플랫 P2P 스웜 프로토콜.
  - Enterprise Edition (EE)과 코어 동기화를 유지하면서 100% AGPL-3.0 경계 엄수.
- [ ] **v0.5.0**: 익스트림 엣지 동적 In-VRAM 스트리밍 추론 (8GB VRAM에서 100B+ 모델 구동).
  - 비동기 듀얼 버퍼 핑퐁(Ping-Pong) VRAM 파이프라이닝 (레이어별 순방향 연산과 가중치 사전 로드 병렬화).
  - 동적 MoE Top-2 활성 전문가 VRAM 캐시 교체 및 RAII VRAM 해제 가드.

---

## 오픈소스 및 상용 라이선스

본 프로젝트는 듀얼 라이선스(Dual Licensing) 방식을 채택합니다:

- **커뮤니티 오픈소스 버전 (Community Edition)**: **[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE)** 하에 배포됩니다. 개인 학습, 학술 연구 및 오픈소스 커뮤니티에서는 완전 무료로 사용할 수 있습니다. 단, LLMZIP을 네트워크 서비스(SaaS, 클라우드 API, 모델 호스팅 등)에 통합하거나 수정본을 배포하는 경우 AGPLv3에 따라 전체 소스 코드를 공개해야 합니다.
- **상용 및 엔터프라이즈 라이선스 (Commercial / Enterprise License)**: 내부 상용 플랫폼, 독점 비공개 제품, 클라우드 인프라에 통합하여 AGPLv3의 소스 코드 공개 의무를 면제받고자 하는 기업을 위해 제공됩니다.

상용 라이선스 구매, 맞춤형 통합 솔루션 또는 SLA 지원이 필요한 경우 메인테이너에게 문의하거나 [GitHub Discussions](https://github.com/kaohaoyi/LLMZIP/discussions)를 통해 문의해 주시기 바랍니다.
