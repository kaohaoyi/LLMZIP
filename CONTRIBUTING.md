# Contributing to LLMZIP

## Repository Layout

- `crates/llmzip-core`: Streaming SafeTensors reader, VTAS layout planner, SIMD quantizers, WAL/SHA-256 commit lock.
- `crates/llmzip-cli`: CLI binary (`llmzip`) and MCP server.
- `crates/llmzip-ffi`: C ABI bindings (`include/`).
- `crates/llmzip-py`: Python bindings (`PyO3` / `maturin`).

## Local Setup & Checks

Requirements: Rust 1.80+ (`cargo`, `rustfmt`, `clippy`).

```bash
cargo fmt --check
cargo clippy -p llmzip-core -p llmzip-cli -p llmzip-ffi --all-targets
cargo test -p llmzip-core -p llmzip-cli -p llmzip-ffi
```

Note: `crates/llmzip-py` uses PyO3 `extension-module`, so build and test it separately with `maturin`:

```bash
maturin develop --manifest-path crates/llmzip-py/Cargo.toml
pytest crates/llmzip-py/tests
```

For release binaries (CLI + C ABI):

```bash
cargo build -p llmzip-cli -p llmzip-ffi --release
```

## Pull Requests

1. Open an issue before starting large architectural changes or adding new container/quant formats.
2. Keep each PR focused on one change. Do not mix unrelated formatting diffs with logic edits.
3. Add unit or integration tests for bug fixes, quantization math, WAL recovery, or offset calculation changes.
4. Confirm `cargo fmt --check`, `clippy`, and `cargo test` pass before opening the PR.

## Reporting Bugs

Include:
- OS and architecture (e.g. Linux x86_64, Windows ARM64, macOS arm64)
- Output of `llmzip --version`
- Exact command run and terminal output
- Model repo / shard size (note whether output stayed locked as `UNVF` or committed as `GGUF`)

## Licensing

LLMZIP is dual-licensed under `AGPL-3.0-or-later` and a commercial license. Contributions submitted to this repository are licensed under `AGPL-3.0-or-later`.
