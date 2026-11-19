# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Workspace crates `forge-core`, `forge-device`, `forge-runtime`, `forge-benchmark`, and CLI binary `forge` (`forge-cli`).
- Declarative `gfx1030` hardware target profile (`profiles/gfx1030/rx6900xt.toml`).
- Multi-source Windows GPU discovery correlating `RtlGetVersion`, Windows Display Registry (`qwMemorySize`, Adrenalin version), Vulkan (`ash`), HIP (`amdhip64_*.dll` `R0600` ABI), and dynamic ADLX (`amdadlx64.dll`).
- Pure-Rust WGSL-to-SPIR-V compute pipeline (`naga` + `ash`) implementing tiled FP32 GEMM, RMSNorm, and RoPE with GPU timestamp profiling and FP64 CPU reference checks.
- Live HIP memory round-trip and event timing verifier (`amdhip64_*.dll`).
- `forge doctor [--json] [--verify] [--python]` and `forge bench {vulkan,llama,triton}` CLI commands.
- Reproducible PyTorch ROCm + `triton-windows` kernel suite (`modules/ai/triton/bench_suite.py`).
- Hardware-independent integration tests (`tests/integration/`) and ignored physical GPU correctness tests (`tests/gpu/`).
- Windows GitHub Actions CI workflow and issue/PR templates.
- Verified upstream feasibility audit (`COMPATIBILITY.md`), baseline benchmark report (`BENCHMARKS.md`), `ARCHITECTURE.md`, `DEVELOPMENT.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `README.md`.
