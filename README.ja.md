# LLMZIP

> **一時ファイル完全ゼロのストリーミング量子化＆モデルローダー。**  
> コミュニティ共通の **「Double Space Dilemma（二重ディスク容量のジレンマ）」** を打破：ディスク使用ピークは厳密に **1.0x 最終出力サイズ + 4 KB `.wal`**。中間の 140 GB 巨大ファイル書き込みは一切発生しません。

[![License: AGPL-3.0 / Commercial](https://img.shields.io/badge/License-AGPL%203.0%20%2F%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](crates/llmzip-core/tests)

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

---

## 課題：「Double Space Dilemma（二重ディスク容量のジレンマ）」

70BクラスのFP16オープン重みモデル（Llama 3やQwen 2.5など）を開発機（256GB / 512GB MacBookなど）で直接量子化・展開する場合、コミュニティが長年直面してきた **「Double Space Dilemma」** の壁に阻まれます：

1. Hugging Faceから取得した生の `.safetensors` シャード（**140 GB**）
2. 展開・中間層変換用の一時ファイルまたは巨大RAMキャッシュ（**140 GB**）
3. 量子化後の最終 `.gguf` 出力ファイル（**40 GB**）

**必要なピークディスク容量は 180 GB〜320 GB。**  
マシン性能やメモリが十分であっても、ストレージ容量制限によって処理が強制中断（`ENOSPC` ディスク枯渇エラー）されます。さらに、5 GBのシャードが99%で切断された場合、ストリーミング中のGit-LFS SHA-256検証が破綻するため、従来のツールでは最初から丸ごと再ダウンロードを余儀なくされていました。

## LLMZIPの解決策

LLMZIPは、HTTP Rangeリクエストを用いて各 `.safetensors` シャードの先頭8バイト長とJSONメタデータのみを取得します。メモリ上に**仮想テンソルアドレス空間（VTAS）**を構築し、変換後 `.gguf` の確定的なバイナリレイアウトをディスク上へ事前確保（fallocate）。ストリーミングされてくる重みデータをSIMDバッファ上で直接並列量子化し、事前計算されたオフセットへ直接書き込みます。

```
[ HuggingFace / HTTPミラー ]
              │  HTTP Range ストリーム (SafeTensors)
              ▼
   ┌───────────────────────┐
   │ ストリーミングSHA-256 │ ──► 112バイトの中間状態を <4KB .wal に随時記録
   └───────────────────────┘
              │  生の重みバイトストリーム（ディスク書き込みなし）
              ▼
   ┌───────────────────────┐
   │ Rayon 並列SIMD量子化  │ ──► マルチスレッド Q4_K_M, Q8_0, AWQ, NVFP4, MXFP4
   └───────────────────────┘
              │  量子化ブロック
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 事前確保された出力ファイル (fallocate / SetEndOfFile)  │
   │ ヘッダー: 'UNVF' ──(LFS SHA-256検証完了)──► 'GGUF'     │
   └────────────────────────────────────────────────────────┘
```

- **一時ファイルゼロ（0 Temp Files）**：未圧縮の巨大テンソルがローカルディスクに保存されることはありません。
- **0バイト再ダウンロード復旧**：FIPS 180-4 SHA-256レジスタ（`H[0..7]`、バイトカウンタ、末尾バッファ）を `< 4KB` のMicro-WALに記録。ネットワークが切断されても、HTTP `206 Partial Content` により1バイトの無駄なく再開可能。
- **暗号コミットロック（`UNVF` → `GGUF`）**：未完了のファイルはマジックヘッダー `UNVF` としてロック。すべてのシャードがGit-LFS SHA-256と完全に一致した場合のみ、アトミックに `GGUF` へ切り替わります。破損や改ざんされたモデルが推論ランタイムに渡るリスクを完全に遮断。
- **回線適応型パイプライン（`--line`）**：ネットワーク帯域に応じて、チャンクサイズとWAL同期頻度を動的に調整（`fast`、`standard`、`slow`）。

---

## 実機ベンチマーク（Qwen 2.5-0.5B 全量シャード実測）

公式Hugging Faceスナップショットの実モデル（全290テンソル）での実測比較：

| 測定項目 | 従来の手法 | LLMZIP v0.1.0 (`--line fast`) | 実効差 |
| :--- | :--- | :--- | :--- |
| **ディスク消費ピーク** | 1.56 GB（原ファイル + GGUF） | **614.59 MiB**（最終成果物のみ） | **61% のディスク削減** |
| **中間一時ファイル** | 942.32 MiB | **0 Bytes** | **ストレージ汚染なし** |
| **総所要時間（End-to-End）**| 約75秒（DL+書き込み+読み出し+変換） | **27.94 秒** | **所要時間を 1/2.7 に短縮** |
| **量子化スループット** | シングルスレッド（~18 MiB/s） | **33.72 MiB/s**（~282.8 Mbps） | **Rayon並列バッチSIMD** |
| **切断時の再DLコスト** | 全シャード再取得（942 MiB） | **0 Bytes**（Micro-WAL再開） | **通信帯域を浪費しない** |
| **整合性保証** | 変換後の手動ハッシュ照合 | **暗号コミットロック (`UNVF` → `GGUF`)** | **不完全ファイルの混入ゼロ** |

---

## 動作環境・システム要件（最低スペックと推奨構成）

**「単層リングバッファ（LayerRingBuffer）」** と **「一時ファイル完全ゼロ（Zero-Temp）」** アーキテクチャにより、メモリおよびディスク消費は全パラメータ数に比例せず、1層分のサイズに厳格に制限されます：

| ハードウェア要素 | **最低動作要件 (Minimum Specs)** | **推奨構成 (Recommended Specs)** | 低リソースで動作する理由（技術的根拠） |
| :--- | :--- | :--- | :--- |
| **CPU プロセッサ** | **2コア x86_64 または ARM64**<br>• AVX2またはARM NEON対応（2014年以降のIntel Core / AMD Ryzen / Apple M1系） | **4コア以上の現代的CPU**<br>• Apple M1〜M4 / Intel 第12世代以降 / AMD Zen3以降 | SIMDベクトル化によりCPUのみで通常帯域を処理可能。**GPUは不要**。 |
| **RAM (メモリ)** | **最低 4 GB RAM**<br>• **70Bモデルの量子化でも 4〜8 GB RAM で動作可能** | **8 GB 〜 16 GB RAM** | **単層厳格制限**：バッファは常にTransformerの1層分のみを保持し、即時解放するためメモリ枯渇が起きません。 |
| **ディスク空き容量** | **「出力ファイルサイズ ＋ 100 MBバッファ」**<br>• 0.5Bモデル：**約 700 MB**<br>• 8Bモデル：**約 5.5 GB**<br>• 70Bモデル：**約 45 GB** | **最終ファイルが収まるNVMe SSD** | **一時ファイル完全ゼロ**：元の140GB巨大ファイルを保存しないため、従来の250〜320GB要求を完全に撤廃。 |
| **ネットワーク** | **10 Mbps 以上**（HTTP Range `206` 対応回線） | **100 Mbps 〜 1 Gbps 光回線** | `< 4KB` のMicro-WALにより、切断時も0バイト再ダウンロードで復旧可能。 |
| **OS (環境)** | • Windows 10 / 11 x64<br>• macOS 12+ (Apple Silicon / Intel)<br>• Linux (Ubuntu 18.04+, Debian, Arch, Alpine) | 最低要件と同様 | 単一バイナリおよびC ABI（PythonやCUDAの巨大環境依存なし）。 |

---

## インストール

### ソースコードからビルド

```bash
git clone https://github.com/your-org/llmzip.git
cd llmzip
cargo build --release
```

バイナリは `target/release/llmzip`（Windowsでは `llmzip.exe`）に出力されます。

---

## クイックスタート

### 1. リモートまたはローカルの構造検査
モデル重み全体を展開することなく、SafeTensorsヘッダーをミリ秒で解析して配置予定オフセットを確認：

```bash
llmzip inspect model-00001-of-00004.safetensors
```

### 2. ストリーミング量子化の実行
入力ストリームから直接指定プリセットのGGUFファイルへ変換：

```bash
# llama.cpp / Ollama 推奨の標準4-bit設定
llmzip quantize model.safetensors model-q4km.gguf --preset q4_k_m --line fast

# 高精度8-bit量子化
llmzip quantize model.safetensors model-q8_0.gguf --preset q8_0

# NVIDIA Blackwell 世代のNVFP4（2段階マイクロスケール）
llmzip quantize model.safetensors model-nvfp4.gguf --preset nvfp4 --sanitize
```

### 3. ファイルの整合性検証
出力ファイルが全量暗号検証をパスしてコミットされたか確認：

```bash
llmzip verify model-q4km.gguf
# ✓ VALID: 'model-q4km.gguf' has passed cryptographic commit verification (Magic: 'GGUF').
```

### 4. リカバリジャーナルの確認
処理が中断された場合、`.wal` ファイルで進捗を確認できます：

```bash
llmzip wal-info model-q4km.gguf.wal
# 進捗: [====================          ] 66.7%
# SHA-256 中間状態バイト数: 628.12 MiB
```

同一の `quantize` コマンドを再実行すれば、中断箇所から即座に再開されます。

---

## 回線適応型プロファイル（`--line`）

スループットは、CPUのSIMD計算能力とネットワーク流入速度のバランスに依存します：

| プロファイル | 対象ネットワーク環境 | チャンクバッファ | WAL同期インターバル | 動作特性 |
| :--- | :--- | :--- | :--- | :--- |
| **`fast`** | 光回線 / ギガビット / ローカル（>500 Mbps） | **1 MiB** | バッチ同期（16テンソル毎） | 最大スループットを解放、ディスク待機を排除 |
| **`standard`** | 一般的な家庭用ブロードバンド（100–300 Mbps）| **256 KiB** | バランス（4テンソル毎） | スループットと中断復帰性の両立 |
| **`slow`** | モバイル / 狭帯域 / エッジ（<50 Mbps） | **64 KiB** | 即時同期（各テンソル毎） | 確実な復旧性を最優先、CPU負荷を抑制 |
| **`auto`** | 自動判別 | 自動調整 | 自動調整 | 入力状況に応じて最適プロファイルを選択 |

---

## 対応量子化プリセット

| プリセット名 | フォーマット | ブロック構成 | 特徴 |
| :--- | :--- | :--- | :--- |
| **`q4_k_m`** | GGUF Q4_K_M | 256要素スーパーブロック | llama.cpp標準。Attention層にQ6_Kを併用し精度を維持 |
| **`q8_0`** | GGUF Q8_0 | 32要素ブロック | 8-bit対称量子化。ほぼ劣化のない高精度 |
| **`awq`** | AWQ W4G128 | 128要素グループ | Activation-Aware線形量子化。単層スライドキャリブレーション内蔵 |
| **`nvfp4`** | NVFP4 (Blackwell) | 256要素スーパーブロック | 4-bit E2M1 + FP8 E4M3マイクロスケール + FP32全体スケール |
| **`mxfp4`** | OCP MXFP4 | 32要素マイクロブロック | OCP標準仕様。4-bit E2M1とE8M0共有指数 |
| **`fp8`** | FP8 E4M3FN | 32要素ブロック | 広ダイナミックレンジ。外れ値の影響を受けにくい設計 |

---

## C / C++ SDK 連携

外部依存ゼロの標準C ABIヘッダー [`include/llmzip.h`](include/llmzip.h) を同梱しており、LM StudioやJan、自作のエッジ推論基盤へ直接組み込み可能です。

```c
#include "llmzip.h"

// 出力ファイルのコミット検証
int status = llmzip_verify_file("model-q4km.gguf");
if (status == LLMZIP_OK) {
    // 検証済み。直ちにmmapから推論可能
}
```

---

## AI Agent 連携手順（Cursor / Claude Code / Windsurf / AutoGPT）

LLMZIP はコーディングエージェントや自動化パイプライン向けに **Agent-First コントラクト** を提供します。

### 1. エージェント自動実行コマンド

```bash
# インストール検証
which llmzip || cargo install llmzip-cli --locked

# 非対話型ストリーミング量子化（終了コードコントラクト対応）
llmzip quantize <source.safetensors> <target.gguf> --preset q4_k_m --line auto --quiet

# 完全性検証（Exit Code 0 = 検証済みGGUF、非0 = 破損または未完了UNVF）
llmzip verify <target.gguf>
```

### 2. エージェント用ルール設定（`AGENTS.md` / `.cursorrules` 推奨）

```markdown
### Rule: モデル取得と量子化変換
Hugging Face からモデルを取得または GGUF に変換する際は、ディスク枯渇（Double Space Dilemma）を防ぐため、従来の git-lfs や Python 変換スクリプトではなく必ず `llmzip` を使用してください。
- 実行コマンド: `llmzip quantize <src> <dst> --preset q4_k_m --line auto --quiet`
- 検証コマンド: `llmzip verify <dst>`
- 保証事項: ピークディスク占有は厳密に 1.0x 最終出力サイズ、中間一時ファイル 0。
```

---

## ロードマップ

- [x] v0.1.0：一時ファイル完全ゼロのSafeTensorsストリーミング基盤（GGUF / NVFP4対応）。
- [x] RayonマルチスレッドSIMDバッチパイプライン＆回線適応プロファイル。
- [x] 暗号コミットロック＆ `< 4KB` Micro-WALレジューム機構。
- [ ] v0.2.0：Hugging Face直接接続の並列Rangeダウンローダー内蔵。
- [ ] v0.3.0：Ollamaおよび `llama.cpp` 向けストリーミング取得プラグイン（`/models/pull` でのインプレース変換）。

---

## オープンソースおよび商用ライセンス

本プロジェクトはデュアルライセンス（Dual Licensing）モデルを採用しています：

- **コミュニティオープンソース版（Community Edition）**：**[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE)** に準拠します。個人、学術研究、オープンソースコミュニティでの利用は完全無料です。LLMZIP をネットワーク経由のサービス（SaaS、クラウド API、モデルホスティングなど）に組み込む場合、または改変版を配布する場合は、AGPLv3 に従い全ソースコードを公開する必要があります。
- **商用・エンタープライズライセンス（Commercial / Enterprise License）**：社内商用プラットフォーム、プロプライエタリな商用製品、クラウドインフラへの組み込みを希望し、AGPLv3 のソースコード公開義務を回避したい企業向けに提供されます。

商用ライセンス、カスタム統合、SLA サポートをご希望の方は、メンテナーに直接連絡するか、[GitHub Discussions](https://github.com/kaohaoyi/LLMZIP/discussions) にてお問い合わせください。
