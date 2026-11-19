# RDNA Forge

**Windows-native AMD Radeon diagnostics, capability verification, and reproducible GPU compute benchmarking toolkit.**

RDNA Forge helps AMD Radeon developers, local AI engineers, and GPU kernel contributors on Windows 11 accurately detect `gfx` targets, audit broken vs working compute backends, and measure kernels and LLM inference against strict numerical correctness and statistical standards—starting with RDNA2 (`gfx1030`, RX 6900 XT).

### What Works Today
- **Multi-source Windows GPU, runtime & live ADLX telemetry (`forge doctor`, `forge monitor`)**: Correlates `RtlGetVersion`, Windows Display Registry (`qwMemorySize`, Adrenalin version), Vulkan (`ash`), AMD HIP (`amdhip64_*.dll` `R0600` ABI), Python ROCm/`triton-windows`, and pure-Rust ADLX `V1.4` live streaming (`table`, `jsonl`, `csv`) with WDDM VRAM pressure alerts.
- **Read-only ADLX tuning inspection & safe power-limit validation (`forge tune`)**: Inspects hardware tuning capabilities, presets, power limits (`%`), GFX clocks/voltage, VRAM clock/memory timing modes, fan curves, and Zero-RPM state (`tune show`); validates `power-limit <PCT>` and `reset` in safe dry-run mode by default (explicit `--apply` required to commit).
- **Live GPU capability verification (`forge doctor --verify`)**: Executes bit-exact HIP 4 MiB H2D/D2H transfers and pure-Rust WGSL->SPIR-V (`naga`) Vulkan compute kernels (baseline FP32 GEMM, register-blocked `128x64 vec4` FP32 GEMM, `OpSDot` INT8 DP4A GEMM, Wave32 `W8A32` / `Q4_0` GEMV, RMSNorm, RoPE, fused subgroup `RMSNorm+RoPE`) validated against exact FP64/`i64` CPU references.
- **Reproducible benchmark harnesses (`forge bench`)**: Statistical median/IQR/MAD/p95 timing + numerical accuracy + peak RSS + ADLX telemetry across Vulkan compute (`--kernel all|gemm|gemm-opt|gemm-i8|gemv-w8|gemv-q4|rmsnorm|rope|fused-rms-rope`), PyTorch ROCm + `triton-windows` (`--smoke`, opt-in `--telemetry`), and `llama.cpp` (`--flash-attn`, `--hip-disable-graphs`).

### Currently Verified Hardware
- **Primary verified target**: AMD Radeon RX 6900 XT (`gfx1030`, RDNA2, 16 GiB VRAM) on Windows 11 Build `26300` with Adrenalin `26.11.2`.
- **Detected secondary adapter**: AMD Ryzen 7 7700X integrated Radeon Graphics (`gfx1036`, RDNA2).
- Other RDNA2, RDNA3, and RDNA4 targets are modeled in capability tables but **not yet tested on physical hardware**.

---

## 1. Purpose & Status

Consumer AMD GPUs on Windows—especially RDNA2 (`gfx1030`)—face fragmented tooling: upstream Triton removed RDNA1/RDNA2 support (`PR #12042`), `hipBLASLt` excludes `gfx10XX`, `rocm_smi_lib` and `rocprofiler-sdk` are unsupported on Windows, and prebuilt Windows ROCm packages frequently omit runtime dependencies or fail silently.

RDNA Forge provides an evidence-driven engineering foundation before building custom optimizations: every backend is probed honestly, gated by actual hardware ISA capabilities, and measured with reproducible baselines.

**Current Project Status:** Milestone 5 complete (ADLX active vs factory VBIOS fan curve diagnostics + external `FanControl.exe` override detection, plus pure-Rust GGUF v2/v3 Model Inspector, Anti-WDDM-Spill VRAM Planner & RDNA2 `llama.cpp` Autotuner `forge llm {discover,inspect,plan,autotune}` and `Forge Desktop` **LLM Optimizer** tab — verified on `Qwen3.8-27B` `UD-IQ4_XS` `13.41 GiB` with **`+108%` prefill (`343.20 tok/s`)** and **`+63%` token generation (`26.81 tok/s`)** at 2K context via `GGML_VK_ENABLE_MEMORY_PRIORITY=1`), building on Milestone 4C (pure-Rust AMF VCN 3.0 DX11 video encode benchmark at **`412.85–417.74 FPS`** `1080p` and **`116.46–120.30 FPS`** `4K`), Milestone 4B (Wave32 GGUF `Q4_K` GEMV at **`239.14 GB/s`** and `16 KiB` LDS tiled causal FlashAttention at **`1125.66 GFLOP/s`**), Milestone 4A (**Forge Desktop**), Milestone 3C, Milestone 3B, Milestone 3A, and Milestone 2.

---

## 2. Supported Hardware & Operating Systems

| Target | Architecture | OS | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 (`26300`) | **Verified** | Full discovery, Vulkan compute, HIP runtime, PyTorch ROCm 10.2, `triton-windows` 3.9, `llama.cpp` `b12532` |
| AMD Radeon Graphics (`gfx1036`, Raphael iGPU) | RDNA2 | Windows 11 (`26300`) | **Partially verified** | Enumerated across Vulkan, HIP, and Registry; compute not separately benchmarked |
| Other RDNA2 / RDNA3 / RDNA3.5 / RDNA4 | RDNA1–4 | Windows 11 | **Not tested** | Profile/PCI tables present; no hardcoded RX 6900 XT assumptions |
| Linux | — | Linux | **Deferred** | Future milestone; Windows-native is the Milestone 1 priority |

---

## 3. Verified Runtime Compatibility

| Runtime / SDK | Verified Version | Status on `gfx1030` (Win11) | Key Notes |
| :--- | :--- | :--- | :--- |
| **Vulkan Compute** (`ash` + `naga`) | API `1.4.315` / Loader `1.4.309` | **Verified** | Zero external SDK needed (`naga` compiles WGSL->SPIR-V in Rust) |
| **AMD HIP Driver DLL** (`amdhip64_6.dll`) | HIP `6.4.50101` (`10.0.3652.0`) | **Partially verified** | Ships with Adrenalin `26.11.2`; `hipMalloc`/`hipMemcpy`/events work; SDK compiler absent unless TheRock installed |
| **ROCm / TheRock + PyTorch** | `rocm==10.2.0`, `torch==2.15.0+rocm10.2.0` | **Verified** (FP32/FP16) / **Unsupported** (INT8, Flash SDPA) | `torch._int_mm` fails (`hipBLASLt` excludes `gfx10XX`); Flash/Efficient SDPA unavailable |
| **Triton (`triton-windows`)** | `3.9.0.post30` | **Partially verified** | Community RDNA2 support (`triton-windows#41`); M1 suite verified; M2 W8A16/W4A16/swizzled/fused/causal kernels verified in `--smoke` mode (full-size stress omitted due to host `Kernel-Power 41` resets) |
| **llama.cpp (`Vulkan` & `ROCm`)** | `b12532` (`rocm-10.0`) | **Verified** (Vulkan) / **Partially verified** (ROCm) | Uncontended `ROCm0`: `7369.46` `pp512` / `233.50` `tg128` (`Vulkan0`: `5227.71` / `240.21` default, `6211.14` `pp512` with `-fa 0`); ROCm drops 4.3x to `53.71` under WDDM VRAM contention |
| **ADLX** (`amdadlx64.dll`) | `1.4.1.122` (`V1.4` COM) | **Verified** (telemetry + read-only tuning) | Pure-Rust COM vtables for live monitoring (`monitor`), tuning inspection (`tune show`), and range-validated dry-run power control without vendoring proprietary SDK headers |
| **DirectML** | `DirectML.dll` | **Partially verified** | Present in Windows `System32`; upstream is in maintenance mode |
| **AMF** (`amfrt64.dll`) | `1.4.37.0` (COM DX11 probe & VCN encode) | **Verified** (`2 HW enc / 5 HW dec`) | Pure-Rust COM vtable DX11 probe (`forge video caps`) & hardware encode benchmark (`forge video bench`: `412.85–417.74 FPS` 1080p, `116.46–120.30 FPS` 4K across H.264/HEVC/HEVC 10-bit P010) |

See [COMPATIBILITY.md](COMPATIBILITY.md) for the complete 11-repository upstream feasibility audit, licenses, and technical blockers.

---

## 4. Installation & Build Instructions

### Prerequisites
- Windows 11 (x64) with AMD Adrenalin driver (`26.11.2` tested)
- Rust stable toolchain (`1.87+` with `rustfmt` and `clippy`, matching [rust-toolchain.toml](rust-toolchain.toml))
- Visual Studio Build Tools (MSVC linker)
- Optional: Python `3.11` with `torch[device-gfx1030]==2.15.0` and `triton-windows<3.10` (see [DEVELOPMENT.md](DEVELOPMENT.md))
- Optional: official `llama.cpp` Windows Vulkan/ROCm binaries

### Build & Verify

```powershell
cargo build --release -p forge-cli
cargo test --workspace
```

Run live GPU correctness tests serially on physical AMD hardware:

```powershell
cargo test --workspace -- --ignored --test-threads=1 --nocapture
```

---

## 5. Usage Examples

### Diagnostics (`forge doctor`)

```powershell
.\target\release\forge.exe doctor --verify --python <venv>\Scripts\python.exe
```

Observed output on RX 6900 XT (Windows 11 Build `26300`, Adrenalin `26.11.2`):

```text
RDNA FORGE DIAGNOSTICS

GPU
  AMD Radeon RX 6900 XT
  Architecture: RDNA2
  Target: gfx1030
  VRAM: 15.98 GiB (17163091968 bytes)
  PCI: 0x1002:0x73AF (0000:03:00.0)
  Compute Units: 80 CUs (40 WGPs)
  Resizable BAR: DISABLED (256 MiB BAR)

SECONDARY GPU [1]
  AMD Radeon(TM) Graphics
  Architecture: RDNA2
  Target: gfx1036
  VRAM: 0.50 GiB (536870912 bytes)

SYSTEM
  OS: Windows 11 (10.0.26300, Build 26300)
  CPU: AMD Ryzen 7 7700X 8-Core Processor (16 logical cores)
  RAM: 31.10 GiB
  Driver: Adrenalin 26.11.2 (32.0.21045.29003)

BACKENDS
  Vulkan: [VERIFIED] Detected (Vulkan 1.4.309, compute queue ready)
    - Driver: 26.11.2 (AMD proprietary shader compiler)
    - FP16=true, INT8=true, CoopMatrix=true, ReBAR=disabled (256 MiB BAR)
  HIP: [PARTIALLY VERIFIED] Driver runtime detected (HIP 6.4.50101; SDK toolchain absent)
    - Library: amdhip64_6.dll
    - Driver runtime DLL present; full ROCm SDK (rocm-sdk / hipcc) not in PATH
  Triton: [PARTIALLY VERIFIED] Detected (triton-windows 3.9.0.post30, PyTorch HIP 7.17.26434; community RDNA2 support)
    - Python 3.11.9 (<venv>\Scripts\python.exe)
    - PyTorch 2.15.0+rocm10.2.0, HIP=7.17.26434
    - Torch GPU[0]: AMD Radeon RX 6900 XT
  ADLX: [VERIFIED] Detected (ADLX 1.4.1.122)
    - amdadlx64.dll dynamically loaded
  DirectML: [PARTIALLY VERIFIED] Present (DirectML.dll; upstream in maintenance mode)
  AMF: [VERIFIED] Detected (AMF 1.4.37.0, 2 HW encoders / 5 HW decoders)
    - amfrt64.dll DX11 probe OK; HW Encoders: [H.264/AVC, HEVC/H.265]
    - HW Decoders: [H.264/AVC, HEVC/H.265, VP9, AV1, MJPEG]
    - Unsupported: [AV1 Encode]
  llama.cpp: [NOT TESTED] Not in PATH (external binary)

CAPABILITIES
  Device discovery: VERIFIED (2 GPU(s) enumerated)
  HIP transfer: VERIFIED (AMD Radeon RX 6900 XT (gfx1030) 4 MiB H2D+D2H bit-exact in 2.718 ms (3.09 GB/s))
  GEMM (Vulkan FP32): VERIFIED (64x64x64 passed (max_abs_err=2.86e-6, 0.010 ms))
  RMSNorm (Vulkan FP32): VERIFIED (128x256 passed (max_abs_err=2.38e-7, 0.005 ms))
  RoPE (Vulkan FP32): VERIFIED (64x8x64 passed (max_abs_err=6.35e-6, 0.010 ms))
```

Machine-readable output is supported via `forge doctor --json`, `forge video caps --json`, `forge tune show --json`, `forge monitor --format jsonl|csv --output <path>`, and `--output <path>` on benchmark commands.

### Live Monitor, Tuning & Video Capabilities (`forge monitor`, `forge tune`, `forge video`)

```powershell
# Stream real-time telemetry (table, jsonl, or csv) with Ctrl+C graceful exit
.\target\release\forge.exe monitor --count 3 --interval-ms 200

# Read-only ADLX tuning inspection and safe dry-run power limit check
.\target\release\forge.exe tune show
.\target\release\forge.exe tune power-limit 5
# Pass --apply only when intentionally modifying GPU tuning:
# .\target\release\forge.exe tune power-limit 5 --apply

# Probe AMD AMF hardware video encode/decode codec support & run VCN 3.0 hardware encode benchmark
.\target\release\forge.exe video caps
.\target\release\forge.exe video bench --codec all --width 1920 --height 1080 --fps 60 --bitrate-mbps 12 --preset speed --warmup 15 --frames 120
```

---

## 6. Verified Benchmark Summary (`gfx1030`, RX 6900 XT)

All benchmarks use warmups, repeated runs, median and dispersion (`min`, `p95`, `IQR`, `MAD`, `stddev`), and CPU FP64/FP32/`i64` verification.

| Workload | Backend | Shape / Model | Median Latency | Median Throughput | Notes / Status |
| :--- | :--- | :--- | ---: | ---: | :--- |
| GEMM FP32 (baseline vs `tiled_vec4`) | Vulkan (`ash` + `naga`) | `2048x2048x2048` | 11.990 ms vs **1.262 ms** | 1432.90 vs **13610.84 GFLOP/s** | Verified (**9.50x speedup**, `max_abs=9.54e-5`) |
| GEMM INT8 (`dot4I8Packed` -> `OpSDot`) | Vulkan (`ash` + `naga`) | `2048x2048x2048` | **0.567 ms** | **30.29 TOPS** | Verified (**bit-exact**, `max_abs=0`) |
| GEMV `W8A32` (Wave32 subgroup, `batch=16`) | Vulkan 1.3 Wave32 | `11008x4096` (`4096^2`) | **0.054 ms** (`0.033 ms`) | **835.67 GB/s** (`512.41 GB/s`) | Verified (**4.03x** vs single `0.218 ms`, `4.00x` wt comp, `max_abs=9.77e-4`) |
| GEMV `Q4_0` (Wave32 subgroup, `batch=16`) | Vulkan 1.3 Wave32 | `11008x4096` (`4096^2`) | **0.064 ms** (`0.030 ms`) | **438.74 GB/s** (`353.30 GB/s`) | Verified (**4.67x** vs single `0.301 ms`, `IQR=0.0003 ms`, `6.40x` wt comp) |
| GEMV `Q4_K` (Wave32 GGUF, `batch=16`) | Vulkan 1.3 Wave32 | `11008x4096` (`4096^2`) | **0.106 ms** (`0.049 ms`) | **239.14 GB/s** (`192.73 GB/s`) | Verified (`7.11x` wt comp, `IQR=0.0006 ms`, `max_abs=1.60e-4`) |
| Fused `RMSNorm+RoPE` (`batch=16`) | Vulkan subgroup | `1024x32x128` | **0.036 ms** | **936.98 GB/s** | Verified (**3.34x** vs single `0.120 ms`, `IQR=0.0002 ms`) |
| Causal FlashAttention (16 KiB LDS) | Vulkan 1.3 Wave32 | `8x1024x128` (`16x1024x64`) | **1.910 ms** (`2.377 ms`) | **1125.66 GFLOP/s** (`904.26 GFLOP/s`) | Verified (FP32 `max_abs=1.42e-7`, saves `32–64 MiB` `S^2` score HBM) |
| GEMM FP32 (`rocBLAS`) | PyTorch ROCm 10.2 | `2048x2048x2048` | 0.828 ms | 20.75 TFLOP/s | Verified |
| GEMM FP16 (`rocBLAS`) | PyTorch ROCm 10.2 | `4096x4096x4096` | 3.975 ms | 34.58 TFLOP/s | Verified |
| GEMM FP16 (Triton) | `triton-windows` 3.9 | `2048x2048x2048` | 0.884 ms | 19.43 TFLOP/s | Verified |
| GEMM INT8 (`torch._int_mm`) vs Triton `W8A16`/`W4A16` | `hipBLASLt` vs `triton-windows` | `256^3` smoke (`W8/W4`) | — | — | `_int_mm` **Unsupported** (`gfx1030`); Triton `W8A16`/`W4A16` **Smoke-verified** (`passed=true`) |
| RMSNorm FP16 | PyTorch eager vs Triton | `4096x4096` | 1.038 ms vs **0.093 ms** | 64.66 vs **723.24 GB/s** | Verified (11.2x Triton speedup) |
| RoPE FP16 | PyTorch eager vs Triton | `1x2048x32x128` | 0.589 ms vs **0.082 ms** | 56.94 vs **411.21 GB/s** | Verified (7.2x Triton speedup) |
| Attention FP16 (Tiled Triton) | `triton-windows` 3.9 | `1x16x1024x64` | 0.657 ms | 6.54 TFLOP/s | Verified (Flash/Efficient SDPA unavailable); causal smoke uses 70x less VRAM than Math SDPA |
| LLM Prefill (`pp512`, uncontended) | `llama.cpp` `b12532` ROCm vs Vulkan | Qwen2.5-1.5B Q4_K_M | **69.48 ms** vs 97.94 ms | **7369.46** vs 5227.71 tok/s | Verified (**ROCm +41.0% faster** prefill; Vulkan `-fa 0`: `6211.14 tok/s`) |
| LLM Decode (`tg128`, uncontended) | `llama.cpp` `b12532` Vulkan vs ROCm | Qwen2.5-1.5B Q4_K_M | **532.87 ms** vs 548.17 ms | **240.21** vs 233.50 tok/s | Verified (note: ROCm drops to `53.71 tok/s` under WDDM VRAM contention) |
| 27B LLM @ 2K–4K Context (`forge llm`) | `llama.cpp` `b12532` Tuned Vulkan & ROCm | Qwen3.8-27B `UD-IQ4_XS` (`13.41 GiB`) | — | **`343.20` pp / `26.81` tg tok/s** (`d=2048`) | Verified (`GGML_VK_ENABLE_MEMORY_PRIORITY=1`: **+108% pp / +63% tg**; `q4_0` KV @ `d=4096`: **`25.63` tg** Vulkan / **`246.22` pp** ROCm) |
| VCN 3.0 Hardware Encode (`1080p` / `4K`) | AMF `amfrt64.dll` `1.4.37.0` DX11 | `H.264 NV12` / `HEVC NV12` / `HEVC 10-bit P010` | **2.39–2.42 ms** (`1080p`) / **8.31–8.59 ms** (`4K`) | **412.85–417.74 FPS** (`1080p`) / **116.46–120.30 FPS** (`4K`) | Verified (`856–998 MPix/s`, `120/120` Annex-B packets, `37–44 W` GPU power) |

Full dispersion metrics, WDDM VRAM diagnosis, ADLX telemetry, model SHA-256, and reproduction steps are in [BENCHMARKS.md](BENCHMARKS.md).

---

## 7. Architecture Overview

RDNA Forge separates Rust orchestration and safety policy from GPU compute backends:

- [`crates/forge-core`](crates/forge-core): Target parser (`GfxTarget`), PCI ID table, capability policy, hardware profile schema ([rx6900xt.toml](profiles/gfx1030/rx6900xt.toml)).
- [`crates/forge-device`](crates/forge-device): Windows OS, Registry, Vulkan (`ash`), HIP `R0600` ABI, and dynamic ADLX device correlation.
- [`crates/forge-profiler`](crates/forge-profiler): Dynamic ADLX `V1.4` COM telemetry & tuning (`IADLXGPUTuningServices`, clocks, temps, power, fan curves, WDDM VRAM contention checks).
- [`crates/forge-runtime`](crates/forge-runtime): Backend probing, pure-Rust dynamic AMF DX11 COM video capability probing (`amfrt64.dll`), pure-Rust WGSL->SPIR-V (`naga`), RAII Vulkan compute engine with multi-dispatch batching (`gemm_f32_tiled_vec4`, `gemm_i8_dp4a`, Wave32 `gemv_w8a32` / `gemv_q4_0` / `gemv_q4_k`, `rmsnorm_rope_f32_subgroup`, `flash_attention_f32_wave32`), FP64/`i64` CPU references, HIP transfer verifier, and shared live capability verification.
- [`crates/forge-benchmark`](crates/forge-benchmark): Statistical harness, process RSS tracking, ADLX telemetry integration, and `llama-bench` / Triton adapters.
- [`apps/forge-cli`](apps/forge-cli): `forge` binary (`doctor`, `video`, `monitor`, `tune`, `bench`, `profile`).
- [`apps/forge-desktop`](apps/forge-desktop): Optional Tauri v2 + WebView2 Windows GUI (`Overview`, live `Monitor`, `Tuning`, `Video (AMF)`, and Vulkan `Benchmarks` with shared `BusyGuard` mutual exclusion).
- [`modules/ai/triton`](modules/ai/triton): Reproducible PyTorch ROCm + Triton kernel suite.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module contracts and extension points.

---

## 8. Project Roadmap

- **Milestone 1 (Complete):** Feasibility audit, clean repository foundation, multi-source Windows discovery, `forge doctor [--verify]`, Vulkan/PyTorch/Triton/`llama.cpp` baseline harness, and RX 6900 XT verification.
- **Milestone 2 (Complete):**
  - Diagnosed `llama.cpp` `ROCm0` decode gap (`hipMemGetInfo` blindness to non-HIP Windows WDDM VRAM pressure vs `7369` `pp512` / `233.5` `tg128` uncontended, HIP graphs confirmed active).
  - Implemented `forge-profiler` dynamic ADLX `V1.4` COM telemetry with automated WDDM VRAM pressure alerts.
  - Added register-blocked `gemm_f32_tiled_vec4` (**9.50x** Vulkan speedup to `13.61 TFLOP/s`) and native `OpSDot` INT8 DP4A (`30.29 TOPS`, bit-exact) compute shaders.
  - Added smoke-verified `gfx1030` Triton `W8A16`, `W4A16`, L2-swizzled FP16 GEMM, fused `RMSNorm+RoPE`, and causal FlashAttention kernels.
- **Milestone 3A (Complete):**
  - Added `forge monitor` real-time ADLX telemetry streaming (`table`, `jsonl`, `csv`, `--output`, Win32 `SetConsoleCtrlHandler` graceful exit, summary stats).
  - Added `forge tune` pure-Rust ADLX `V1.4` tuning inspection (`show`) and hardware range-validated `power-limit` / `reset` commands with safe dry-run default (`--apply` opt-in).
- **Milestone 3B (Complete):**
  - Added Vulkan 1.3 `subgroup_size_control` (`required_subgroup_size = 32`, `REQUIRE_FULL_SUBGROUPS`) and SPIR-V `OpGroupNonUniformFAdd` subgroup compilation in `forge-runtime`.
  - Implemented and live-verified Wave32 `W8A32` GEMV (`207.35 GB/s` at `11008x4096`), `Q4_0` block-32 GEMV (`6.40x` compression, `0.089 ms` min / `93.97 GB/s` median), and single-pass fused `RMSNorm+RoPE` with KV-cache `pos_offset` (`0.120 ms` / `280.55 GB/s`).
- **Milestone 3C (Complete):**
  - Added multi-dispatch Vulkan command buffer batching (`--batch-dispatches 1..=256`, default `16`) with inter-dispatch `vkCmdPipelineBarrier` synchronization, boosting RX 6900 XT decode/elementwise kernels by **3.34x–4.67x** (`W8A32` GEMV `835.67 GB/s`, `Q4_0` GEMV `438.74 GB/s` with `IQR=0.0003 ms`, fused `RMSNorm+RoPE` `936.98 GB/s`).
  - Implemented pure-Rust dynamic AMD AMF (`amfrt64.dll`) DX11 COM capability probing via `forge video caps [--json]` and upgraded `forge doctor` (`1.4.37.0`, `2 HW enc / 5 HW dec` verified on VCN 3.0).
- **Milestone 4A (Complete):**
  - Added `Forge Desktop` (`apps/forge-desktop`, Tauri v2 + Edge WebView2, zero `node_modules`) with real-time Canvas 2D ADLX strip charts, hardware-locked tuning validation, AMF VCN viewer, and batched Vulkan compute benchmark runner synchronized by RAII `BusyGuard`.
- **Milestone 4B (Complete):**
  - Added pure-Rust WGSL->SPIR-V 1.3 Wave32 GGUF `Q4_K` super-block GEMV (`7.11x` compression, **`239.14 GB/s`** / `0.106 ms` at `11008x4096`, `batch=16`) and `16 KiB` LDS tiled causal FlashAttention (**`1125.66 GFLOP/s`** at `8x1024x128`, `904.26 GFLOP/s` at `16x1024x64`, `max_abs=1.42e-7` vs FP64 CPU reference), integrated into `forge doctor --verify`, `forge bench`, and `Forge Desktop`.
- **Milestone 4C (Complete):**
  - Added pure-Rust dynamic AMD AMF (`amfrt64.dll` `1.4.37.0`) VCN 3.0 DX11 hardware video encode benchmarking (`forge video bench` + `Forge Desktop` **Video (AMF)** runner) across `H.264/AVC NV12`, `HEVC/H.265 NV12`, and `HEVC/H.265 10-bit P010` using a pre-uploaded 16-surface `AMF_MEMORY_DX11` VRAM ring, Annex-B NAL bitstream verification, and live ADLX telemetry (**`412.85–417.74 FPS`** at `1080p` / `37 W`; **`116.46–120.30 FPS`** / **`965.94–997.79 MPix/s`** at `4K UHD` / `44 W`).
- **Milestone 5 (Complete):**
  - Upgraded `forge-profiler` ADLX fan tuning inspection to query both active (`GetFanTuningStates`) and VBIOS factory default (`GetDefaultFanTuningStates`) 5-point curves and automatically detect external fan controllers (`FanControl.exe`) driving RDNA2 in flat-PWM override mode.
  - Added pure-Rust GGUF v2/v3 Model Inspector, Anti-WDDM-Spill Memory Planner, and RDNA2 `llama.cpp` Empirical Autotuner (`forge llm {discover,inspect,plan,autotune}` + `Forge Desktop` **LLM Optimizer** tab), verified on `Qwen3.8-27B` `UD-IQ4_XS` (`13.41 GiB`, 65 layers) with **`+108%` prefill (`343.20 tok/s`)** and **`+63%` token generation (`26.81 tok/s`)** at 2K context (`GGML_VK_ENABLE_MEMORY_PRIORITY=1`, `GGML_VK_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`, and `-ctk q4_0 -ctv q4_0`).
- **Future:** AMF VCN 3.0 hardware decode pipeline benchmarking, FidelityFX SDK evaluation, RDNA3/RDNA4 verification, Linux support.

---

## 9. Contributing, License, and Security

- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).
- **License**: Licensed under the [MIT License](LICENSE).
- **Security**: Report vulnerabilities privately per [SECURITY.md](SECURITY.md).
- **Changelog**: Tracked in [CHANGELOG.md](CHANGELOG.md).
