# RDNA Forge

**Windows-native AMD Radeon diagnostics, capability verification, and reproducible GPU compute benchmarking toolkit.**

RDNA Forge helps AMD Radeon developers, local AI engineers, and GPU kernel contributors on Windows 11 accurately detect `gfx` targets, audit broken vs working compute backends, and measure kernels and LLM inference against strict numerical correctness and statistical standards—starting with RDNA2 (`gfx1030`, RX 6900 XT).

### What Works Today
- **Multi-source Windows GPU & runtime diagnostics (`forge doctor`)**: Correlates `RtlGetVersion`, Windows Display Registry (`qwMemorySize`, Adrenalin version), Vulkan (`ash`), AMD HIP (`amdhip64_*.dll` `R0600` ABI), ADLX (`amdadlx64.dll`), and Python ROCm/`triton-windows` without relying on broken 32-bit WMI VRAM fields.
- **Live GPU capability verification (`forge doctor --verify`)**: Executes bit-exact HIP 4 MiB H2D/D2H transfers and pure-Rust WGSL->SPIR-V (`naga`) Vulkan compute kernels (`GEMM`, `RMSNorm`, `RoPE`) validated against FP64 CPU references.
- **Reproducible benchmark harnesses (`forge bench`)**: Statistical median/IQR/MAD/p95 timing + numerical accuracy + peak memory across Vulkan compute, PyTorch ROCm + `triton-windows`, and `llama.cpp` (`Vulkan` vs `ROCm/HIP`).

### Currently Verified Hardware
- **Primary verified target**: AMD Radeon RX 6900 XT (`gfx1030`, RDNA2, 16 GiB VRAM) on Windows 11 Build `26300` with Adrenalin `26.11.2`.
- **Detected secondary adapter**: AMD Ryzen 7 7700X integrated Radeon Graphics (`gfx1036`, RDNA2).
- Other RDNA2, RDNA3, and RDNA4 targets are modeled in capability tables but **not yet tested on physical hardware**.

---

## 1. Purpose & Status

Consumer AMD GPUs on Windows—especially RDNA2 (`gfx1030`)—face fragmented tooling: upstream Triton removed RDNA1/RDNA2 support (`PR #12042`), `hipBLASLt` excludes `gfx10XX`, `rocm_smi_lib` and `rocprofiler-sdk` are unsupported on Windows, and prebuilt Windows ROCm packages frequently omit runtime dependencies or fail silently.

RDNA Forge provides an evidence-driven engineering foundation before building custom optimizations: every backend is probed honestly, gated by actual hardware ISA capabilities, and measured with reproducible baselines.

**Current Project Status:** Milestone 1 complete (`Forge Core`, `forge doctor`, live Vulkan/HIP verification, and reproducible Vulkan, PyTorch/Triton, and `llama.cpp` baselines verified on RX 6900 XT).

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
| **Triton (`triton-windows`)** | `3.9.0.post30` | **Partially verified** | Community RDNA2 support (`triton-windows#41`); FP16 GEMM, RMSNorm, RoPE, tiled Attention verified |
| **llama.cpp (`Vulkan`)** | `b12532` | **Verified** | Runs out of the box; `213.17 tok/s` `tg128` on Qwen2.5-1.5B Q4_K_M |
| **llama.cpp (`ROCm/HIP`)** | `b12532` (`rocm-10.0`) | **Partially verified** | Zip omits `hipblas.dll`/`rocblas.dll` (requires ROCm `bin` on `PATH`); `53.71 tok/s` `tg128` |
| **ADLX** (`amdadlx64.dll`) | `1.4.1.122` | **Verified** (version probe) | Dynamically loaded at runtime without vendoring proprietary SDK headers |
| **DirectML** / **AMF** | `DirectML.dll` / `amfrt64.dll` | **Partially verified** | Present in Windows `System32`; DirectML is in upstream maintenance mode |

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

Run live GPU correctness tests on physical AMD hardware:

```powershell
cargo test --workspace -- --ignored --nocapture
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
  AMF: [PARTIALLY VERIFIED] Detected (amfrt64.dll present)
  llama.cpp: [NOT TESTED] Not in PATH (external binary)

CAPABILITIES
  Device discovery: VERIFIED (2 GPU(s) enumerated)
  HIP transfer: VERIFIED (AMD Radeon RX 6900 XT (gfx1030) 4 MiB H2D+D2H bit-exact in 2.718 ms (3.09 GB/s))
  GEMM (Vulkan FP32): VERIFIED (64x64x64 passed (max_abs_err=2.86e-6, 0.010 ms))
  RMSNorm (Vulkan FP32): VERIFIED (128x256 passed (max_abs_err=2.38e-7, 0.005 ms))
  RoPE (Vulkan FP32): VERIFIED (64x8x64 passed (max_abs_err=6.35e-6, 0.010 ms))
```

Machine-readable output is supported via `forge doctor --json` and `--output <path>` on benchmark commands.

---

## 6. Benchmark Methodology & Verified Baseline Summary

All benchmarks use explicit warmups (`10` kernel / `1` `llama-bench` warmup), repeated runs (`50` kernel / `5` `llama-bench` reps), median and dispersion (`min`, `p95`, `IQR`, `MAD`, `stddev`), and CPU FP64/FP32 accuracy verification.

| Workload | Backend | Shape / Model | Median Latency | Median Throughput | Status |
| :--- | :--- | :--- | ---: | ---: | :--- |
| GEMM FP32 (baseline WGSL) | Vulkan (`ash` + `naga`) | `2048x2048x2048` | 12.743 ms | 1348.18 GFLOP/s | Verified |
| GEMM FP32 (`rocBLAS`) | PyTorch ROCm 10.2 | `2048x2048x2048` | 0.828 ms | 20.75 TFLOP/s | Verified |
| GEMM FP16 (`rocBLAS`) | PyTorch ROCm 10.2 | `4096x4096x4096` | 3.975 ms | 34.58 TFLOP/s | Verified |
| GEMM FP16 (Triton) | `triton-windows` 3.9 | `2048x2048x2048` | 0.884 ms | 19.43 TFLOP/s | Verified |
| GEMM INT8 (`torch._int_mm`) | PyTorch / `hipBLASLt` | `2048x2048x2048` | — | — | Unsupported (`gfx1030`) |
| RMSNorm FP16 | PyTorch eager vs Triton | `4096x4096` | 1.038 ms vs **0.093 ms** | 64.66 vs **723.24 GB/s** | Verified (11.2x Triton speedup) |
| RoPE FP16 | PyTorch eager vs Triton | `1x2048x32x128` | 0.589 ms vs **0.082 ms** | 56.94 vs **411.21 GB/s** | Verified (7.2x Triton speedup) |
| Attention FP16 (Flash/Efficient SDPA) | PyTorch SDPA | `1x16x1024x128` | — | — | Unsupported (No kernel) |
| Attention FP16 (Tiled Triton) | `triton-windows` 3.9 | `1x16x1024x64` | 0.657 ms | 6.54 TFLOP/s | Verified |
| LLM Prefill (`pp512`) | `llama.cpp` `b12532` Vulkan vs ROCm | Qwen2.5-1.5B Q4_K_M | 107.01 ms vs 139.27 ms | **4784.47** vs 3676.34 tok/s | Verified |
| LLM Decode (`tg128`) | `llama.cpp` `b12532` Vulkan vs ROCm | Qwen2.5-1.5B Q4_K_M | 600.47 ms vs 2383.07 ms | **213.17** vs 53.71 tok/s | Verified (Vulkan 3.97x faster) |

Full dispersion metrics, VRAM/RSS usage, model SHA-256, stability results (`50/50` Vulkan, `100/100` Triton), and exact reproduction steps are in [BENCHMARKS.md](BENCHMARKS.md).

---

## 7. Architecture Overview

RDNA Forge separates Rust orchestration and safety policy from GPU compute backends:

- [`crates/forge-core`](crates/forge-core): Target parser (`GfxTarget`), PCI ID table, capability policy, hardware profile schema ([rx6900xt.toml](profiles/gfx1030/rx6900xt.toml)).
- [`crates/forge-device`](crates/forge-device): Windows OS, Registry, Vulkan (`ash`), HIP `R0600` ABI, and dynamic ADLX device correlation.
- [`crates/forge-runtime`](crates/forge-runtime): Backend probing, pure-Rust WGSL->SPIR-V (`naga`), RAII Vulkan compute engine, FP64 CPU references, HIP transfer verifier.
- [`crates/forge-benchmark`](crates/forge-benchmark): Statistical harness, process RSS tracking, and `llama-bench` / Triton adapters.
- [`apps/forge-cli`](apps/forge-cli): `forge` binary (`doctor`, `bench`, `profile`).
- [`modules/ai/triton`](modules/ai/triton): Reproducible PyTorch ROCm + Triton kernel suite.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module contracts and extension points.

---

## 8. Project Roadmap

- **Milestone 1 (Complete):** Feasibility audit, clean repository foundation, multi-source Windows discovery, `forge doctor [--verify]`, Vulkan/PyTorch/Triton/`llama.cpp` baseline harness, and RX 6900 XT verification.
- **Milestone 2 (Proposed Next):**
  - Profile and diagnose the `3.97x` `llama.cpp` Windows WDDM `ROCm0` vs `Vulkan0` `tg128` decode gap.
  - Add optimized Vulkan subgroup/coop-matrix GEMM & DP4A quantized dot-product shaders plus Triton fused RMSNorm+RoPE / FlashAttention tuning for `gfx1030`.
  - Add live ADLX power/clock/temperature sampling (`forge-profiler` / `forge-tuning`) during benchmark runs.
- **Future (Deferred Until Core Compute & Tuning Mature):** Hardware video pipelines (`AMF`), FidelityFX graphics research, optional Tauri desktop UI, Linux support.

---

## 9. Contributing, License, and Security

- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).
- **License**: Licensed under the [MIT License](LICENSE).
- **Security**: Report vulnerabilities privately per [SECURITY.md](SECURITY.md).
- **Changelog**: Tracked in [CHANGELOG.md](CHANGELOG.md).
