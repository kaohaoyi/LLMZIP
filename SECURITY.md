# Security Policy

## Supported Versions

Security fixes apply to `main` and the latest tagged release (`v0.1.0`+).

## Reporting a Vulnerability

Do not open a public GitHub issue for security bugs.

Report privately via:
- [GitHub Private Vulnerability Reporting](https://github.com/kaohaoyi/LLMZIP/security/advisories/new) (preferred)
- Email: **kaohaoyi@gmail.com**

Include steps to reproduce, OS/architecture, and any minimal input files or stack traces.

## Scope

Issues of highest concern:
- Path traversal or arbitrary file overwrite in `llmzip mcp` or CLI path handling
- Bypassing the `UNVF` to `GGUF` SHA-256 verification commit lock
- Memory safety bugs or UB in `crates/llmzip-ffi` or `unsafe` SIMD paths
- `.wal` journal or HTTP Range bugs that silently corrupt quantized output
