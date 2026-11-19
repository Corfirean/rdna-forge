# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Milestone 5 (ADLX Fan Curve Diagnostics + GGUF Inspector, Anti-WDDM-Spill Memory Planner & RDNA2 LLM Autotuner)
- Upgraded `crates/forge-profiler/src/adlx_tuning.rs` and `tuning.rs` (`FanTuningInfo`):
  - Added `IADLXManualFanTuning1::GetDefaultFanTuningStates` (vtable slot `[23]`) to read the GPU VBIOS factory 5-point fan curve (`default_fan_curve`, verified on RX 6900 XT as `30C->20%, 50C->30%, 61C->40%, 72C->50%, 80C->70%`) alongside active ADLX states (`GetFanTuningStates`).
  - Added flat PWM override detection (`flat_pwm_override_pct`) and Win32 `CreateToolhelp32Snapshot` process detection (`external_controller`, e.g. `FanControl.exe`) explaining why external fan controllers set all 5 RDNA2 ADLX curve points to a single instant PWM percentage (clamped to the `20%` hardware floor when Zero-RPM is disabled).
  - Updated `forge tune show` and `Forge Desktop` (**Tuning** tab & `drawFanCurveChart`) to plot both active ADLX and dashed factory VBIOS curves.
- Added `crates/forge-benchmark/src/llm_optimizer.rs` (`inspect_gguf_model`, `estimate_kv_and_state_bytes`, `discover_local_llm_assets`, `plan_llm_configuration`, `autotune_llm_model`):
  - Pure-Rust GGUF v2/v3 header parser supporting standard and hybrid SSM/Attention architectures (`qwen35` `full_attention_interval`, `nextn_predict_layers`, `ssm.*`), computing exact per-layer weight sizes and `f16`/`q8_0`/`q4_0` KV+SSM memory budgets against live ADLX free physical VRAM.
  - Hardware-verified AMD RDNA2 (`gfx1030`) tweaks (`GGML_VK_ENABLE_MEMORY_PRIORITY=1` for Windows WDDM `VK_EXT_memory_priority` residency, `GGML_VK_VISIBLE_DEVICES=0` & `HIP_VISIBLE_DEVICES=0` to isolate from the Ryzen 7000 iGPU `12.4 GiB` shared-RAM trap, `-sm none`, `-ctk/-ctv q4_0`, and physical core thread count `-t 8`).
  - Live-verified on RX 6900 XT + `Qwen3.8-27B` `UD-IQ4_XS` (`13.41 GiB`, 65 layers): `GGML_VK_ENABLE_MEMORY_PRIORITY=1` + `-fa 0` boosts `d=2048` prefill by **+108%** (`165.12 -> 343.20 tok/s`) and token generation by **+63%** (`16.42 -> 26.81 tok/s`); at `d=4096`, `-ctk q4_0 -ctv q4_0` sustains **`25.63 tok/s`** decode on Vulkan (**+61%**) and **`246.22 tok/s`** prefill on ROCm (**20.3x** vs Vulkan `-fa 1`).
- Added `forge llm {discover,inspect,plan,autotune}` CLI commands and the **LLM Optimizer** tab in `Forge Desktop`.

### Added — Milestone 4C (AMD AMF VCN 3.0 Hardware Video Encode Benchmark)
- Extended `crates/forge-runtime/src/amf.rs` with pure-Rust dynamic `amfrt64.dll` COM vtable bindings (`AMFVariantStruct` 24-byte x64 ABI, `AMFContext::AllocSurface`, `AMFComponent::{SetProperty,Init,Drain,SubmitInput,QueryOutput}`, `AMFData::{QueryInterface,Convert,SetPts,SetDuration}`, `AMFSurface::{GetPlanesCount,GetPlaneAt}`, `AMFPlane::{GetNative,GetWidth,GetHeight,GetHPitch}`, `AMFBuffer::{GetSize,GetNative}` with `AMFBuffer` IID `b04b7248-b6f0-4321-b691-baa4740f9fcb`) and `run_amf_encode_benchmark`:
  - Pre-allocates a 16-surface ring in `AMF_MEMORY_HOST`, fills moving synthetic `NV12` (8-bit) or `P010` (10-bit MSB-aligned `u16`) Y/UV planes, and converts all 16 surfaces to `AMF_MEMORY_DX11` VRAM textures via `AMFData::Convert(AMF_MEMORY_DX11)` prior to timing so CPU `memcpy` is excluded from VCN hardware encode measurements.
  - Verifies output bitstream packets (`120/120`) for valid Annex-B NAL start codes (`00 00 00 01` / `00 00 01`).
- Added `crates/forge-benchmark/src/video.rs` (`VideoBenchSuiteConfig`, `VideoBenchReport`, `run_video_benchmark_suite`) wrapping `TelemetryGuard` ADLX sampling and `peak_process_rss_bytes()` around multi-codec VCN encode runs.
- Added `forge video bench` CLI subcommand (`--codec all|h264|hevc|hevc10`, `--width`, `--height`, `--fps`, `--bitrate-mbps`, `--preset speed|balanced|quality`, `--warmup`, `--frames`, `--json`, `--output`) and integrated an interactive **VCN Hardware Encode Benchmark Runner** into the **Video (AMF)** tab of `Forge Desktop` (`run_video_benchmark` Tauri command synchronized by `BusyGuard`).
- Live-verified on RX 6900 XT (VCN 3.0, AMF `1.4.37.0`, Adrenalin `26.11.2`):
  - `1920x1080` (`1080p`, `speed`, `12 Mbps`): **`413.13 FPS`** (`856.66 MPix/s`) on `H.264/AVC NV12`, **`417.74 FPS`** (`866.22 MPix/s`) on `HEVC/H.265 NV12`, and **`412.85 FPS`** (`856.09 MPix/s`) on `HEVC/H.265 10-bit P010` at **`37.0 W`** GPU power (`68 MHz` GFX clock).
  - `3840x2160` (`4K UHD`, `speed`, `35 Mbps`): **`120.30 FPS`** (`997.79 MPix/s`) on `H.264/AVC NV12`, **`117.94 FPS`** (`978.23 MPix/s`) on `HEVC/H.265 NV12`, and **`116.46 FPS`** (`965.94 MPix/s`) on `HEVC/H.265 10-bit P010` at **`44.0 W`** GPU power (`73 MHz` GFX clock).

### Added — Milestone 4B (RX 6900 XT Vulkan Compute Practice: GGUF `Q4_K` GEMV & Causal FlashAttention)
- Implemented two new RDNA2 Wave32 Vulkan 1.3 WGSL->SPIR-V 1.3 compute kernels in `forge-runtime`:
  - `gemv_q4_k_wave32.wgsl`: zero-LDS, zero-barrier GGUF `Q4_K` super-block-256 (`144 B` / `7.11x` compression vs FP32) GEMV with IEEE-754 half `(d, dmin)` unpacking and 6-bit sub-block `(scale, min)` decoding, achieving **`192.73 GB/s`** (`0.0491 ms`) at `4096x4096` (**4.08x** vs `batch=1`) and **`239.14 GB/s`** (`0.1063 ms`, `IQR=0.0006 ms`, `CV=0.51%`, `max_abs=1.60e-4`) at `11008x4096` with `batch=16`.
  - `flash_attention_f32_wave32.wgsl`: `16 KiB` LDS tiled (`BC = 16`, `head_dim <= 128`) online-softmax FlashAttention with workgroup-uniform upper-triangle causal skipping, achieving **`904.26 GFLOP/s`** (`2.3772 ms`, `max_abs=1.27e-7`) at `16x1024x64` and **`1125.66 GFLOP/s`** (`1.9096 ms`, `max_abs=1.42e-7`) at `8x1024x128` while saving `32–64 MiB` of naive `S^2` score HBM traffic.
- Added pure-Rust `f16_bits_to_f32`, `f32_to_f16_bits`, `pack_q4_k_block`, `unpack_q4_k_scale_min`, `deterministic_q4_k_weights`, `ref_gemv_q4_k_f32`, and `ref_flash_attention_f32` CPU references in `crates/forge-runtime/src/reference.rs`, plus `.emit_to_string(source)` WGSL validation error reporting in `shaders.rs`.
- Integrated `--kernel gemv-q4-k` and `--kernel flash-attn` across `forge doctor --verify` (12/12 capabilities verified), `forge-benchmark`, `forge-cli`, and `Forge Desktop`, with offline unit/integration tests and live RX 6900 XT edge-case/tail-length GPU tests (`40/40` offline, `17/17` live GPU passing).

### Added — Milestone 4A (Forge Desktop — Tauri v2 Windows UI)
- Added `apps/forge-desktop` (Tauri v2 + Edge WebView2, zero `node_modules` build dependencies) providing an engineering-focused desktop UI over the shared Rust crates:
  - **Overview**: Hardware adapter specifications, detected compute/video/telemetry backends, live HIP+Vulkan verification trigger (`Verify Live`), and JSON export.
  - **Monitor**: Live ADLX polling (`500 ms` / `1000 ms` / `2000 ms`, pause/resume, automatic pause during exclusive GPU operations), 8 KPI tiles, WDDM VRAM pressure warnings, 4 high-DPI Canvas 2D rolling strip charts (`Clocks`, `Temperatures`, `Power`, `Load & VRAM %`), and CSV/JSONL export.
  - **Tuning**: Read-only ADLX capabilities/preset/GFX/VRAM state, 5-point fan curve plot, hardware-range-bounded Power Limit dry-run validator, and explicit `Unlock Hardware Writes` confirmation gate for `--apply` / factory reset.
  - **Video (AMF)**: DX11 COM VCN hardware encoder/decoder capability table (`H.264`, `HEVC`, `VP9`, `AV1`, `MJPEG`) and JSON export.
  - **Benchmarks**: Parameterized Vulkan compute runner (`gemv_w8`, `gemv_q4`, `fused_rms_rope`, `rmsnorm`, `rope`, `gemm_opt`, `gemm_i8`, `gemm`, `all`) with configurable batch dispatch count, full statistical dispersion & accuracy table, and ADLX telemetry summary.
- Added thread-safe `AppState` / `BusyGuard` mutual exclusion in `apps/forge-desktop/src/commands.rs` with unit tests for lock exclusion, benchmark parameter bounds, and safe `dry_run = true` defaults.

### Changed
- Extracted shared `VULKAN_VERIFY_CAPABILITIES`, `default_untested_capabilities`, and `run_live_capability_checks` from `apps/forge-cli/src/main.rs` into `crates/forge-runtime/src/verify.rs` so CLI diagnostics and Forge Desktop share one verified implementation.

### Added — Milestone 3C (Batched Vulkan Dispatch & AMD AMF Video Capability Probing)
- Added multi-dispatch command buffer batching (`dispatches_per_submit` `1..=256`, default `16` for sub-millisecond decode/elementwise kernels) with inter-dispatch `SHADER_WRITE -> SHADER_WRITE | SHADER_READ` `vkCmdPipelineBarrier` synchronization in `forge-runtime`, `forge-benchmark`, and `forge bench vulkan --batch-dispatches N`:
  - Eliminated single-dispatch host-fence overhead on RX 6900 XT (`gfx1030`), delivering **`512.41–835.67 GB/s`** on Wave32 `W8A32` GEMV (**3.77x–4.03x** speedup vs `batch=1`), **`353.30–438.74 GB/s`** on `Q4_0` GEMV (**4.16x–4.67x** speedup; `11008x4096` IQR collapsed from `0.209 ms` to `0.0003 ms`), **`936.98 GB/s`** on fused Wave32 `RMSNorm+RoPE` (**3.34x**), and **`1317.99 GB/s`** on `rope_f32` (**3.48x**), all passing full FP64 CPU reference checks.
- Implemented pure-Rust dynamic AMD AMF (`amfrt64.dll`) DX11 COM vtable probing (`crates/forge-runtime/src/amf.rs`, zero vendored headers):
  - Upgraded `BackendKind::Amf` diagnostics in `forge doctor` and added `forge video caps [--json]`.
  - Live-verified AMF `1.4.37.0` on RX 6900 XT (VCN 3.0): 2 HW encoders (`H.264/AVC`, `HEVC/H.265`), 5 HW decoders (`H.264/AVC`, `HEVC/H.265`, `VP9`, `AV1`, `MJPEG`), 10-bit/interlaced flags, native/supported surface formats, and clean `AV1 Encode` unsupported detection (`error 30`).
- Added unit tests, CLI parsing tests, integration tests, and live GPU tests (`vulkan_batched_decode_dispatch_matches_single_dispatch` and `amf_live_video_capabilities_probe_verifies_vcn_codecs`).

### Added — Milestone 3B (Vulkan Compute LLM Decode Kernels for RDNA2)
- Enabled Vulkan 1.3 `subgroup_size_control` and `compute_full_subgroups` (`requiredSubgroupSize = 32`, `REQUIRE_FULL_SUBGROUPS`) plus `naga::valid::Capabilities::SUBGROUP` SPIR-V 1.3 codegen (`OpGroupNonUniformFAdd`) in `forge-runtime`.
- Implemented three pure-Rust WGSL->SPIR-V RDNA2 LLM decode kernels:
  - `gemv_w8a32_wave32.wgsl`: zero-LDS, zero-barrier Wave32 `extractBits` (`OpBitFieldSExtract`) + `subgroupAdd` INT8 weight-only GEMV achieving **207.35 GB/s** (`0.218 ms`, `413.73 GFLOP/s`) at `11008x4096` (`max_abs=9.77e-4`).
  - `gemv_q4_0_wave32.wgsl`: zero-LDS, zero-barrier Wave32 block-32 `Q4_0` (`6.40x` compression vs FP32) GEMV reaching `0.089 ms` min (`~317.8 GB/s` peak) / `0.301 ms` median (`93.97 GB/s`) at `11008x4096` (`max_abs=1.14e-5`).
  - `rmsnorm_rope_f32_subgroup.wgsl`: two-barrier, 64-byte LDS single-pass subgroup `RMSNorm + RoPE` with KV-cache `pos_offset` support (`0.120 ms` / `280.55 GB/s` at `1024x32x128`, eliminating intermediate HBM traffic).
- Added exact FP64 CPU reference evaluators (`ref_gemv_w8a32_f32`, `ref_gemv_q4_0_f32`, `ref_rmsnorm_rope_f32`, `deterministic_q4_0_weights`), `forge bench vulkan --kernel {gemv-w8,gemv-q4,fused-rms-rope}` integration, `forge doctor --verify` checks (8/8 Vulkan + HIP + discovery = 10/10 verified), and offline/live-GPU tests (`13/13` passing on RX 6900 XT).

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
