# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Milestone 3A (Forge Live Telemetry Monitor & ADLX Tuning Inspector)
- Extended `forge-profiler` with pure-Rust ADLX `V1.4` tuning COM vtable bindings (`IADLXGPUTuningServices`, `IADLXGPUPresetTuning`, `IADLXManualPowerTuning(1)`, `IADLXManualGraphicsTuning2(_1)`, `IADLXManualVRAMTuning2(_1)`, `IADLXManualFanTuning(1)`) without vendoring proprietary SDK headers.
- Added domain tuning types and hardware `IntRange` validation (`min..=max` and `step` alignment), `MemoryTimingMode` mapping, read-only `read_all_gpu_tuning()`, and dry-run-safe `set_gpu_power_limit()` / `reset_gpu_tuning_to_factory()`.
- Added `ForgeError::InvalidParameter` in `forge-core` for out-of-range or ambiguous tuning parameters.
- Added `forge monitor` CLI command supporting live ADLX polling (`--interval-ms`, `--count`, `--duration-secs`, `--device-id`, `--format table|jsonl|csv`, `--output`), Win32 `SetConsoleCtrlHandler` graceful shutdown, WDDM VRAM warnings, and end-of-run summary statistics.
- Added `forge tune` CLI commands (`show [--device-id] [--json]`, `power-limit <PCT> [--device-id] [--apply]`, `reset [--device-id] [--apply]`) with safe dry-run execution by default.
- Added unit, CLI parsing, integration, and strictly read-only live RX 6900 XT GPU tuning/dry-run validation tests.

### Added — Milestone 2 (ADLX Telemetry, Optimized `gfx1030` Kernels & WDDM Root-Cause Analysis)
- New crate `forge-profiler` implementing pure-Rust ADLX `V1.4` COM vtable telemetry (`amdadlx64.dll`, zero vendored headers): live core/VRAM clocks, core/hotspot temperatures, GPU power, fan RPM, voltage, and Windows WDDM VRAM occupancy / competing-workload warnings across `forge doctor` and benchmark reports.
- Vulkan compute optimizations in `forge-runtime`:
  - Register-blocked `128x64x16` (`8x4` `vec4` FMA, bank-padded LDS) FP32 GEMM (`gemm_f32_tiled_vec4.wgsl`) achieving **13.61 TFLOP/s** (**9.50x speedup** vs baseline `16x16` at `2048^3` on RX 6900 XT).
  - Packed INT8 DP4A GEMM (`gemm_i8_dp4a.wgsl`, `requires packed_4x8_integer_dot_product;`) compiled via `naga` to native SPIR-V 1.6 `OpSDot` (`V_DOT4_I32_I8`), achieving **30.29 TOPS** bit-exact integer matmul (`max_abs_err=0`) gated on `PhysicalDeviceVulkan13Features::shader_integer_dot_product`.
- CLI flags `--kernel {all,gemm,gemm-opt,gemm-i8,rmsnorm,rope}` for `forge bench vulkan`, `--flash-attn 0|1` and `--hip-disable-graphs` for `forge bench llama`, plus `--smoke` and opt-in `--telemetry` for `forge bench triton`.
- Additional `gfx1030` Triton kernels in `modules/ai/triton/bench_suite.py` (smoke-verified `passed=true` at `256x256`): L2-swizzled FP16 GEMM (`GROUP_SIZE_M=8`), per-channel `W8A16` dequant GEMM (filling the `hipBLASLt` gap), group-128 `W4A16` packed dequant GEMM, fused single-pass `RMSNorm+RoPE`, and causal tiled online-softmax FlashAttention.
- Root-cause documentation in `BENCHMARKS.md` and `COMPATIBILITY.md` explaining the Milestone 1 `llama.cpp` `ROCm0` `53.71 tok/s` WDDM VRAM contention trap vs uncontended `ROCm0` (`7369.46 tok/s` `pp512`, `233.50 tok/s` `tg128`), and documenting full-size Triton stress power limits.
- Offline (`tests/integration/`) and serialized live-GPU (`tests/gpu/`) correctness tests covering ADLX telemetry, `OpSDot` SPIR-V emission, tiled `vec4` FP32 GEMM, and bit-exact INT8 DP4A GEMM.

### Added — Milestone 1 (Foundation & Feasibility Baseline)
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
