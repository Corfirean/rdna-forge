# Benchmarks & Reproducible Baseline

All measurements on this page were captured on physical target hardware using the checked-in `forge` CLI and `modules/ai/triton/bench_suite.py`. No numbers are extrapolated, smoothed, or cherry-picked.

---

## 1. Test Environment & Provenance

| Component | Verified Value |
| :--- | :--- |
| **Discrete GPU** | AMD Radeon RX 6900 XT (`gfx1030`, RDNA2, PCI `1002:73AF`, `0000:03:00.0`, 80 CUs / 40 WGPs, `17,163,091,968` B / `15.98 GiB` VRAM) |
| **Resizable BAR** | Disabled (`256 MiB` host-visible device-local BAR heap; `isLargeBar = 0`) |
| **CPU** | AMD Ryzen 7 7700X (8 cores / 16 logical threads) |
| **System RAM** | `33,397,133,312` B (`31.10 GiB`) |
| **Operating System** | Windows 11 Home (`10.0.26300`, Build `26300`, x64) |
| **AMD Driver** | AMD Software: Adrenalin Edition `26.11.2` (`32.0.21045.29003`) |
| **Vulkan Runtime** | Loader `1.4.309.0`, Device API `1.4.315` (`AMD proprietary shader compiler`) |
| **HIP Driver DLL** | `amdhip64_6.dll` (`10.0.3652.0`, HIP `6.4.50101`) |
| **Rust Toolchain** | `rustc 1.98.1` (`48a229cea 2026-09-01`), release profile (`cargo build --release`) |
| **Python / ROCm Stack** | Python `3.11.9`, `torch==2.15.0+rocm10.2.0` (HIP `7.17.26434`), `rocm-sdk-device-gfx1030==10.2.0`, `triton-windows==3.9.0.post30` |
| **llama.cpp Binary** | Official release `b12532` (commit `08c252e22`, built with Clang `22.1.8` for Windows x86_64) |
| **GGUF Model Artifact** | `qwen2.5-1.5b-instruct-q4_k_m.gguf` (`1,111,370,240` bytes / `1.04 GiB`, Qwen2 1.5B Q4_K_M) |
| **Model Source URL** | `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| **Model SHA-256** | `6A1A2EB6D15622BF3C96857206351BA97E1AF16C30D7A74EE38970E434E9407E` |

---

## 2. Measurement Methodology

- **Warmup & Repetition**:
  - Vulkan kernels (`forge bench vulkan`): `10` warmup dispatches + `50` measured dispatches per kernel; `50` consecutive dispatches for stability.
  - PyTorch ROCm & Triton kernels (`forge bench triton`): `10` warmup dispatches + `50` measured dispatches per kernel; `100` consecutive dispatches for stability.
  - `llama.cpp` inference (`forge bench llama`): `1` warmup + `5` measured repetitions per test (`pp512`, `tg128`), `ngl = 99`, single-device execution pinned to RX 6900 XT (`Vulkan0` vs `ROCm0`).
- **Timing Mechanisms**:
  - Vulkan suite uses hardware GPU timestamp queries (`VK_QUERY_TYPE_TIMESTAMP` converted via `timestampPeriod`).
  - PyTorch/Triton suite uses `torch.cuda.Event(enable_timing=True)` bracketed by `torch.cuda.synchronize()` before start and `end_ev.synchronize()` after stop (required on Windows WDDM to prevent negative `hipEventElapsedTime` skews).
- **Correctness Validation**:
  - Every kernel output is compared against an exact CPU FP64 (or FP32 for attention) reference, reporting `all_finite`, `max_abs_err`, and `rmse`.
  - For matrices larger than `1024x1024`, the top `32` (Vulkan) or `64` (PyTorch/Triton) rows are validated against exact CPU FP64 multiplication to avoid multi-second host verification delays.
- **Scope Separation**: Isolated kernel throughput improvements do not imply proportional end-to-end LLM token throughput improvements. End-to-end inference is measured separately via `llama-bench`.

---

## 3. Baseline Results — Vulkan Compute (`ash` + `naga` WGSL->SPIR-V)

Note: The initial WGSL shaders are clean unvectorized baseline implementations (`16x16` tiled FP32 GEMM, `256`-thread workgroup RMSNorm, 1D RoPE) compiled at runtime via `naga` without `VK_KHR_cooperative_matrix` or subgroup vectorization.

| Kernel | Shape | Dtype | Median (ms) | Min / P95 / Max (ms) | IQR / MAD (ms) | Throughput | Max Abs Err | RMSE | Accuracy |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `gemm_f32_1024` | `1024x1024x1024` | `f32` | 1.687 | 1.529 / 2.589 / 2.872 | 0.430 / 0.104 | **1272.90 GFLOP/s** | 6.87e-5 | 6.14e-6 | Pass |
| `gemm_f32_2048` | `2048x2048x2048` | `f32` | 12.743 | 12.323 / 13.704 / 13.773 | 0.995 / 0.280 | **1348.18 GFLOP/s** | 9.54e-5 | 1.21e-5 | Pass |
| `rmsnorm_f32_4096x4096` | `4096x4096` | `f32` | 0.342 | 0.153 / 1.258 / 1.370 | 0.224 / 0.073 | **393.03 GB/s** | 3.58e-7 | 3.60e-8 | Pass |
| `rope_f32_1024x32x128` | `1024x32x128` | `f32` | 0.038 | 0.020 / 0.326 / 0.507 | 0.028 / 0.011 | **876.09 GB/s** | 1.13e-4 | 4.14e-6 | Pass |

- **Stability (`vulkan_gemm_f32_512x512x512`)**: `50/50` passed, `all_finite = true`, `all_correct = true`, median `0.325 ms` (`min=0.218`, `p95=1.269`, `IQR=0.089`, `MAD=0.047`).
- **Peak Process RSS**: `488,370,176` B (`465.75 MiB`).

---

## 4. Baseline Results — PyTorch ROCm `10.2.0` & `triton-windows` `3.9.0.post30`

| Benchmark | Backend | Shape | Dtype | Status | Median (ms) | IQR / MAD (ms) | Throughput | Peak VRAM Alloc / Reserved | Max Abs Err |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `pytorch_gemm_f32_2048` | `rocBLAS` | `2048x2048x2048` | `f32` | **Verified** | 0.828 | 0.057 / 0.017 | **20.75 TFLOP/s** | 96.0 / 112.0 MiB | 4.86e-4 |
| `pytorch_gemm_f32_4096` | `rocBLAS` | `4096x4096x4096` | `f32` | **Verified** | 7.196 | 0.363 / 0.080 | **19.10 TFLOP/s** | 288.0 / 320.0 MiB | 8.61e-4 |
| `pytorch_gemm_f16_2048` | `rocBLAS` | `2048x2048x2048` | `f16` | **Verified** | 0.546 | 0.033 / 0.017 | **31.44 TFLOP/s** | 64.0 / 192.0 MiB | 6.24e-2 |
| `pytorch_gemm_f16_4096` | `rocBLAS` | `4096x4096x4096` | `f16` | **Verified** | 3.975 | 0.390 / 0.172 | **34.58 TFLOP/s** | 160.0 / 160.0 MiB | 1.24e-1 |
| `pytorch_int8_gemm_2048` | `hipBLASLt` (`_int_mm`) | `2048x2048x2048` | `i8->i32` | **Unsupported** | — | — | — | — | `HIPBLAS_STATUS_INVALID_VALUE` |
| `pytorch_rmsnorm_f16_4096x4096` | PyTorch eager | `4096x4096` | `f16` | **Verified** | 1.038 | 0.137 / 0.051 | **64.66 GB/s** | 256.0 / 290.0 MiB | 3.91e-3 |
| `triton_rmsnorm_f16_4096x4096` | `triton-windows` | `4096x4096` | `f16` | **Verified** | 0.093 | 0.031 / 0.013 | **723.24 GB/s** | 65.5 / 66.0 MiB | 3.89e-3 |
| `triton_rmsnorm_f16_unaligned` | `triton-windows` | `127x4099` | `f16` | **Verified** | 0.038 | 0.016 / 0.008 | **54.51 GB/s** | 3.5 / 68.0 MiB | 3.87e-3 |
| `pytorch_rope_f16_2048x32x128` | PyTorch eager | `1x2048x32x128` | `f16` | **Verified** | 0.589 | 0.073 / 0.034 | **56.94 GB/s** | 129.0 / 178.0 MiB | 1.99e-3 |
| `triton_rope_f16_2048x32x128` | `triton-windows` | `1x2048x32x128` | `f16` | **Verified** | 0.082 | 0.016 / 0.007 | **411.21 GB/s** | 33.5 / 36.0 MiB | 1.95e-3 |
| `pytorch_sdpa_math_1024` | SDPA Math | `1x16x1024x128` | `f16` | **Verified** | 1.590 | 0.493 / 0.145 | **5.40 TFLOP/s** | 192.0 / 226.0 MiB | 1.57e-4 |
| `pytorch_sdpa_efficient_attention_1024` | SDPA Efficient | `1x16x1024x128` | `f16` | **Unsupported** | — | — | — | — | `RuntimeError: No available kernel` |
| `pytorch_sdpa_flash_attention_1024` | SDPA Flash | `1x16x1024x128` | `f16` | **Unsupported** | — | — | — | — | `RuntimeError: No available kernel` |
| `triton_gemm_f16_2048` | `triton-windows` | `2048x2048x2048` | `f16` | **Verified** | 0.884 | 0.059 / 0.027 | **19.43 TFLOP/s** | 24.0 / 32.0 MiB | 6.25e-2 |
| `triton_gemm_f16_unaligned` | `triton-windows` | `1027x513x1029` | `f16` | **Verified** | 0.122 | 0.030 / 0.016 | **8.89 TFLOP/s** | 4.0 / 32.0 MiB | 3.12e-2 |
| `triton_attention_f16_1024` | `triton-windows` | `1x16x1024x64` | `f16` | **Verified** | 0.657 | 0.064 / 0.031 | **6.54 TFLOP/s** | 9.5 / 34.0 MiB | 1.24e-4 |

- **Stability (`triton_gemm_f16_512x512x512`)**: `100/100` passed, `all_finite = true`, `all_correct = true`, median `0.057 ms` (`min=0.003`, `p95=0.141`, `IQR=0.022`, `MAD=0.011`).
- **Peak Process RSS**: `1,782,730,752` B (`1.66 GiB`).

Key takeaways:
1. Fused Triton kernels outperform unfused PyTorch eager ops on `gfx1030` by **11.2x** for `4096x4096` RMSNorm (`0.093 ms` vs `1.038 ms`) and **7.2x** for RoPE (`0.082 ms` vs `0.589 ms`).
2. `triton-windows` `3.9.0.post30` executes tiled online-softmax attention (`6.54 TFLOP/s`) natively on `gfx1030` where PyTorch's built-in Flash and Efficient SDPA backends report no available kernel.
3. Unvectorized WGSL Vulkan GEMM (`1.35 TFLOP/s` FP32) leaves an order of magnitude of headroom compared to `rocBLAS` (`20.75 TFLOP/s` FP32) and Triton (`19.43 TFLOP/s` FP16).

---

## 5. Baseline Results — `llama.cpp` (`b12532`) Vulkan vs ROCm/HIP

Model: `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (`1.04 GiB`), `-p 512 -n 128 -r 5 -ngl 99`, single-device pinned (`-dev Vulkan0` vs `-dev ROCm0`). ROCm runs required adding TheRock `_rocm_sdk_libraries\bin` and `_rocm_sdk_core\bin` to `PATH` so `ggml-hip.dll` could resolve `hipblas.dll`.

| Backend | Device | Test | Median (tok/s) | Mean ± Stddev (tok/s) | Median Latency (ms) | IQR / MAD (ms) | Offloaded Layers | Peak Device Buffers (Verbose) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **Vulkan** (`b12532`) | `Vulkan0` (`RX 6900 XT`) | `pp512` | **4784.47** | 4790.25 ± 82.00 | 107.01 | 1.25 / 0.73 | 29/29 | `934.69` model + `14.00` KV + `309.27` compute = **1257.96 MiB** |
| **Vulkan** (`b12532`) | `Vulkan0` (`RX 6900 XT`) | `tg128` | **213.17** | 206.10 ± 13.01 | 600.47 | 26.85 / 4.36 | 29/29 | `934.69` model + `7.00` KV + `77.25` compute = **1018.94 MiB** |
| **ROCm/HIP** (`b12532`) | `ROCm0` (`RX 6900 XT`) | `pp512` | **3676.34** | 3531.84 ± 779.72 | 139.27 | 50.10 / 22.42 | 29/29 | `934.70` model + `14.00` KV + `309.27` compute = **1257.97 MiB** |
| **ROCm/HIP** (`b12532`) | `ROCm0` (`RX 6900 XT`) | `tg128` | **53.71** | 53.96 ± 0.58 | 2383.07 | 22.37 / 8.39 | 29/29 | `934.70` model + `7.00` KV + `77.25` compute = **1018.95 MiB** |

Empirical observations:
- **Vulkan wins substantially on Windows 11 `gfx1030`** in both prefill (`1.30x` faster, far lower variance: `CV 1.7%` vs `23.7%`) and especially token generation (`3.97x` faster: `213.17 tok/s` vs `53.71 tok/s`).
- Diagnostic flag sensitivity check: passing `-sm none` (`split-mode none`) alongside `-dev Vulkan0` dropped Vulkan `tg128` from `213.17 tok/s` to `46.34 tok/s`, and `ROCm0` remained in the `53–59 tok/s` range under `-sm none` (`70.39 ± 81.21` across 5 reps) despite `GGML_HIP_GRAPHS` being enabled in `ggml-hip/CMakeLists.txt`. Isolating the Windows WDDM `ROCm0` decode bottleneck is a primary target for Milestone 2.

---

## 6. Milestone 2 Optimizations & Root-Cause Findings (Verified on RX 6900 XT)

### 6.1 Vulkan Compute (`gemm_f32_tiled_vec4` & INT8 `DP4A` `OpSDot`)

Measured on AMD Radeon RX 6900 XT (`gfx1030`, Adrenalin `26.11.2`, Vulkan `1.4.315`), `forge bench vulkan --size 2048 --warmup 5 --iters 20 --stability-runs 0`:

| Kernel | Backend / Shader Strategy | Shape | Dtype | Median (ms) | Min / P95 (ms) | IQR / MAD (ms) | Median Throughput | Speedup vs Baseline | Max Abs Err | RMSE | Accuracy |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `gemm_f32_baseline_2048` | Naive `16x16` LDS tile | `2048x2048x2048` | `f32` | 11.990 | 11.764 / 12.197 | 0.398 / 0.208 | 1432.90 GFLOP/s | 1.00x | 9.54e-5 | 1.21e-5 | Pass |
| `gemm_f32_tiled_vec4_2048` | `128x64x16` tile, `8x4` reg block (`vec4` FMA, bank-padded LDS) | `2048x2048x2048` | `f32` | **1.262** | 1.256 / 1.286 | 0.008 / 0.004 | **13610.84 GFLOP/s** | **9.50x** | 9.54e-5 | 1.21e-5 | Pass |
| `gemm_i8_dp4a_2048` | SPIR-V 1.6 `OpSDot` (`dot4I8Packed` / `V_DOT4_I32_I8`), `64x64x64` | `2048x2048x2048` | `i8->i32` | **0.567** | 0.519 / 1.485 | 0.130 / 0.026 | **30.29 TOPS** | — (INT8) | **0.00e0** | **0.00e0** | Pass (Bit-Exact) |

- **Live ADLX Telemetry Captured During Run**: `13` samples @ `100 ms` — `VRAMClk=1978 MHz`, `Temp=42.0 C` (`42.0-43.0 C`), `Hotspot=48.0 C` (`max 49.0 C`), `Power=39.0 W` (`max 46.0 W`), `VRAM=14473-14537/16368 MiB` (automatic warning emitted by `forge-profiler` for `88.8%` WDDM VRAM occupancy from pre-existing desktop/compute allocations).

---

### 6.2 `llama.cpp` (`b12532`) Windows WDDM Decode Root Cause & Clean Back-to-Back Comparison

#### Root Cause of Milestone 1's `53.71 tok/s` `ROCm0` Decode Outlier
Diagnostic testing (`llama-bench` `-fa 0/1`, `GGML_CUDA_DISABLE_GRAPHS=1`, `HIP_VISIBLE_DEVICES=0`, ADLX `IADLXGPUMetrics::GPUVRAM`, and Windows `\GPU Process Memory(*)\Dedicated Usage`) established why Milestone 1 recorded `ROCm0 tg128 = 53.71 tok/s`:
1. **`hipMemGetInfo` blindness to non-HIP WDDM VRAM on Windows**: During the contention window, `llama-bench` (`ggml_cuda_init`) reported `16227 MiB free` via HIP, while ADLX `GPUVRAM` (`14330–14470 MiB`, `88%`) and Windows Performance Counters showed `~12.4 GiB` of dedicated VRAM already held by another desktop/compute process.
2. **Silent WDDM oversubscription / host-memory paging**: With insufficient contiguous physical VRAM remaining, HIP allocations migrated into WDDM shared system memory without failing initialization, collapsing `tg128` memory-bandwidth-bound decode from `233.5 tok/s` to `53.71 tok/s` (`4.3x` penalty).
3. **`forge-profiler` Mitigation**: RDNA Forge's ADLX telemetry layer automatically flags >50% dedicated VRAM occupancy prior to benchmark runs so WDDM eviction artifacts are caught immediately.

#### Clean Back-to-Back Comparison & Flag Sensitivity (`Qwen2.5-1.5B-Instruct-Q4_K_M`, `-p 512 -n 128 -r 3 -ngl 99`)

| Comparison Run | Backend / Config | Test | Throughput (tok/s) | Notes |
| :--- | :--- | :--- | ---: | :--- |
| Clean `forge bench llama` (default `-fa`) | `Vulkan0` | `pp512` | 5227.71 median (`97.94 ms`) | `stddev=68.78` |
| Clean `forge bench llama` (default `-fa`) | `Vulkan0` | `tg128` | **240.21** median (`532.87 ms`) | `stddev=2.17` |
| Clean `forge bench llama` (default `-fa`) | `ROCm0` | `pp512` | **7369.46** median (`69.48 ms`) | **+41.0% faster** prefill vs `Vulkan0` |
| Clean `forge bench llama` (default `-fa`) | `ROCm0` | `tg128` | 233.50 median (`548.17 ms`) | `stddev=0.20` |
| Direct `llama-bench` (`Vulkan0`, `-fa 0` vs `1`) | `Vulkan0` (`-fa 0` / `-fa 1`) | `pp512` | **6211.14** / 5198.30 (avg) | `-fa 0` yields +19.5% faster prefill on `Vulkan0` |
| Direct `llama-bench` (`Vulkan0`, `-fa 0` vs `1`) | `Vulkan0` (`-fa 0` / `-fa 1`) | `tg128` | 236.55 / **239.15** (avg) | Decode virtually unchanged (~1%) |
| Direct `llama-bench` (`ROCm0`, `HIP_VISIBLE_DEVICES=0`) | `ROCm0` (`-fa 0` / `-fa 1`) | `pp512` | 6630.27 / **7352.69** (avg) | `-fa 1` improves `ROCm0` prefill by +10.9% |
| Direct `llama-bench` (`ROCm0`, `HIP_VISIBLE_DEVICES=0`) | `ROCm0` (`-fa 0` / `-fa 1`) | `tg128` | 223.29 / **234.72** (avg) | `-fa 1` improves `ROCm0` decode by +5.1% |
| `GGML_CUDA_DISABLE_GRAPHS=1` (`HIP_VISIBLE_DEVICES=0`, `-fa 1`) | `ROCm0` (graphs OFF) | `tg128` | 228.78 ± 1.74 (avg) | Confirms HIP graphs ARE active on Windows (+2.6% decode vs disabled) |

---

### 6.3 Milestone 2 Triton Kernels & Hardware Stability Constraint

Five new `gfx1030` Triton kernels were added to `modules/ai/triton/bench_suite.py` and correctness-verified on physical RX 6900 XT hardware via `python modules/ai/triton/bench_suite.py --smoke` (`0` warmup, `1` measured iteration, conservative `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=1, num_warps=4` geometry):

| Kernel | Shape | Dtype | Smoke Verification Status | Max Abs Err | RMSE | Peak VRAM Alloc | Reference & Purpose |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `triton_gemm_f16_swizzled_256` | `256x256x256` | `f16` | **Verified** | 1.56e-2 | 3.32e-3 | 384 KiB | L2/Infinity-Cache `GROUP_SIZE_M=8` swizzled FP16 GEMM |
| `triton_w8a16_gemm_256` | `256x256x256` | `f16xint8->f16` | **Verified** | 3.15e-2 | 4.55e-3 | 320.5 KiB | Per-channel INT8 weight-only dequant GEMM (fills `hipBLASLt` gap) |
| `triton_w8a16_gemm_unaligned` | `259x131x261` | `f16xint8->f16` | **Verified** | 2.50e-2 | 3.18e-3 | — | Arbitrary bounds-masked W8A16 dequant GEMM |
| `triton_w4a16_gemm_256` | `256x256x256` | `f16xint4->f16` | **Verified** | 9.00e-3 | 1.39e-3 | 290 KiB | Group-128 packed 4-bit `uint8` weight-only dequant GEMM |
| `triton_fused_rmsnorm_rope_f16_256x8x128` | `256x8x128` | `f16` | **Verified** | 3.78e-3 | 1.99e-4 | 1.00 MiB | Single-pass RMSNorm + RoPE (vs PyTorch `4.63 MiB` / `2.65e-3` max err) |
| `triton_causal_attention_f16_256` | `1x4x256x64` | `f16` | **Verified** | 9.94e-4 | 5.32e-5 | **512 KiB** | Causal tiled online-softmax FlashAttention (**70x** less VRAM than PyTorch SDPA's `35.38 MiB` / `9.71e-4` max err) |

> [!WARNING]
> **Observed Stability Limit on Full-Size ROCm/Triton Sweeps:** Full-size (`2048³` / `4096³`) consecutive Triton/ROCm benchmark sweeps triggered hard system resets (`Windows Kernel-Power Event ID 41, BugcheckCode = 0x0`, without BSOD or dump) on this Navi 21 host, consistent with high-TDP transient power spikes / PSU OCP or WDDM reset during back-to-back ROCm bursts. Accordingly, steady-state `2048³` throughput numbers for the Milestone 2 Triton kernels are **intentionally omitted** until verified on a power-limited or higher-headroom PSU configuration, and `forge bench triton` provides `--smoke` for safe low-power verification with ADLX telemetry opt-in (`--telemetry`).

---

## 7. Milestone 3B — Vulkan Compute LLM Decode Kernels (`gfx1030`)

Pure-Rust `naga` (`WGSL` -> SPIR-V 1.3 `OpGroupNonUniformFAdd` + `OpBitFieldSExtract`) kernels leveraging Vulkan 1.3 `subgroup_size_control` (`required_subgroup_size = 32`, `REQUIRE_FULL_SUBGROUPS`) on AMD Radeon RX 6900 XT (`warmup = 10`, `iters = 50`, `--stability-runs 0`, full FP64 CPU reference check):

| Kernel | Strategy | Shape | Dtype | Median (ms) | Min / P95 / Max (ms) | IQR / MAD (ms) | Throughput | Compression vs FP32 | Max Abs Err | RMSE | Accuracy |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `gemv_w8a32_wave32_4096x4096` | Wave32 `extractBits` + `subgroupAdd` (0 LDS, 0 barriers) | `4096x4096` | `i8*f32->f32` | **0.124** | 0.114 / 0.135 / 0.137 | 0.009 / 0.004 | **135.87 GB/s** (270.95 GFLOP/s) | **4.00x** | 8.54e-4 | 1.69e-4 | Pass |
| `gemv_w8a32_wave32_11008x4096` | Wave32 `extractBits` + `subgroupAdd` (LLaMA-7B MLP shape) | `11008x4096` | `i8*f32->f32` | **0.218** | 0.132 / 0.247 / 0.258 | 0.021 / 0.012 | **207.35 GB/s** (413.73 GFLOP/s) | **4.00x** | 9.77e-4 | 1.71e-4 | Pass |
| `gemv_q4_0_wave32_4096x4096` | Wave32 block-32 nibble unpack + `subgroupAdd` | `4096x4096` | `q4_0->f32` | **0.124** | 0.123 / 0.132 / 0.132 | 0.001 / 0.000 | **84.87 GB/s** (270.73 GFLOP/s) | **6.40x** | 7.63e-6 | 2.58e-6 | Pass |
| `gemv_q4_0_wave32_11008x4096` | Wave32 block-32 nibble unpack + `subgroupAdd` | `11008x4096` | `q4_0->f32` | **0.301** (`0.089` min) | 0.089 / 0.308 / 0.313 | 0.209 / 0.007 | **93.97 GB/s** (300.07 GFLOP/s) | **6.40x** | 1.14e-5 | 2.67e-6 | Pass |
| `rmsnorm_rope_subgroup_1024x32x128` | Single-pass Wave32 subgroup `RMSNorm+RoPE` (2 barriers, 64 B LDS) | `1024x32x128` | `f32` | **0.120** | 0.118 / 0.121 / 0.128 | 0.001 / 0.001 | **280.55 GB/s** | — (fused) | 1.63e-4 | 4.15e-6 | Pass |
| `rope_f32_1024x32x128` (ref) | Standalone unfused RoPE baseline at identical tensor shape | `1024x32x128` | `f32` | 0.088 | 0.088 / 0.089 / 0.093 | 0.000 / 0.000 | 379.23 GB/s | — | 1.13e-4 | 4.14e-6 | Pass |
| `rmsnorm_f32_4096x4096` (ref) | Unfused RMSNorm (workgroup tree reduction, 128 MiB Infinity Cache working set) | `4096x4096` | `f32` | 0.163 | 0.136 / 0.510 / 0.567 | 0.138 / 0.025 | 823.83 GB/s | — | 3.58e-7 | 3.60e-8 | Pass |

Observed hardware telemetry & behavior via `forge-profiler` ADLX:
- **Sub-millisecond fence idle-clock effect**: Because `execute_timed_loop` waits on a host fence after each `~0.1 ms` dispatch, ADLX recorded idle/low SCLK (`52–84 MHz`, `41–42 W`, `36–37 C`, `~1982–1988 MHz` VRAM clock) across these runs. Even under `75–84 MHz` SMU idle P-states, `gemv_q4_0_wave32_11008x4096` peaked at **`0.089 ms` (`~317.8 GB/s`)** when clocks briefly boosted (accounting for the `min=0.089` vs `median=0.301` spread).
- **Single-pass fusion benefit**: `rmsnorm_rope_subgroup_1024x32x128` executes full `RMSNorm + RoPE` (with `pos_offset` support for KV-cache single-token decode) in `0.120 ms` (`IQR = 0.001 ms`), adding only `0.032 ms` over standalone `rope_f32` while eliminating the intermediate `16 MiB` global VRAM round-trip.

---

## 8. Milestone 3C — Batched Vulkan Compute (`batch=16`) & AMF Hardware Video Probe

Batching `dispatches_per_submit = 16` compute dispatches into a single Vulkan command buffer separated by `SHADER_WRITE -> SHADER_WRITE | SHADER_READ` `vkCmdPipelineBarrier` barriers eliminates per-dispatch queue submission and host-fence round-trip overhead (`warmup = 10`, `iters = 50`, `--stability-runs 0`, `--batch-dispatches 16`, full FP64 CPU reference check):

| Kernel | Shape | Dtype | `batch=1` Median | `batch=16` Median (ms) | `batch=16` Min / P95 / Max (ms) | `batch=16` IQR / MAD (ms) | `batch=16` Throughput | Speedup vs `batch=1` | Max Abs Err | Accuracy |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `gemv_w8a32_wave32_4096x4096` | `4096x4096` | `i8*f32->f32` | 0.124 ms (`135.87 GB/s`) | **0.0328** | 0.0314 / 0.0353 / 0.0375 | 0.0003 / 0.0002 | **512.41 GB/s** (`1021.83 GFLOP/s`) | **3.77x** | 8.545e-4 | Pass |
| `gemv_w8a32_wave32_11008x4096` | `11008x4096` | `i8*f32->f32` | 0.218 ms (`207.35 GB/s`) | **0.0541** | 0.0505 / 0.0556 / 0.0560 | 0.0018 / 0.0009 | **835.67 GB/s** (`1667.48 GFLOP/s`) | **4.03x** | 9.766e-4 | Pass |
| `gemv_q4_0_wave32_4096x4096` | `4096x4096` | `q4_0->f32` | 0.124 ms (`84.87 GB/s`) | **0.0298** | 0.0295 / 0.0327 / 0.0377 | 0.0003 / 0.0002 | **353.30 GB/s** (`1127.03 GFLOP/s`) | **4.16x** | 7.629e-6 | Pass |
| `gemv_q4_0_wave32_11008x4096` | `11008x4096` | `q4_0->f32` | 0.301 ms (`93.97 GB/s`) | **0.0644** | 0.0604 / 0.0647 / 0.0648 | **0.0003** / 0.0001 | **438.74 GB/s** (`1400.98 GFLOP/s`) | **4.67x** | 1.144e-5 | Pass |
| `rmsnorm_rope_subgroup_1024x32x128` | `1024x32x128` | `f32` | 0.120 ms (`280.55 GB/s`) | **0.0358** | 0.0353 / 0.0368 / 0.0389 | 0.0002 / 0.0001 | **936.98 GB/s** | **3.34x** | 1.626e-4 | Pass |
| `rope_f32_1024x32x128` (ref) | `1024x32x128` | `f32` | 0.088 ms (`379.23 GB/s`) | **0.0255** | 0.0250 / 0.0302 / 0.0336 | 0.0002 / 0.0001 | **1317.99 GB/s** | **3.48x** | 1.128e-4 | Pass |
| `rmsnorm_f32_4096x4096` (ref) | `4096x4096` | `f32` | 0.163 ms (`823.83 GB/s`) | **0.1386** | 0.1316 / 0.1422 / 0.1432 | 0.0021 / 0.0012 | **968.40 GB/s** | **1.18x** | 3.576e-7 | Pass |

Observed hardware behavior via `forge-profiler` ADLX & Vulkan timestamps:
- **Jitter elimination & Infinity Cache residency**: `gemv_q4_0_wave32_11008x4096` IQR collapses from `0.209 ms` (`batch=1`) to **`0.0003 ms`** (`batch=16`), stabilizing throughput at `438.74 GB/s`. Working sets (`~11–64 MiB`) fit inside Navi 21's `128 MiB` Infinity Cache across consecutive barriered dispatches, yielding `835–1318 GB/s` effective throughput even while sub-millisecond submissions keep sampled ADLX board telemetry at `36 C`, `36–37 W`, and `~1978–1986 MHz` VRAM clock.

### 8.1 AMF Hardware Video Capability Probe (`amfrt64.dll` `1.4.37.0`)

Verified live on RX 6900 XT (VCN 3.0, Windows 11 Build `26300`, Adrenalin `26.11.2`) via `forge video caps [--json]`:
- **Hardware Encoders (2)**: `H.264/AVC` (`AMFVideoEncoderVCE_AVC`, `128x128..4096x4096`, `NV12`), `HEVC/H.265` (`AMFVideoEncoderHW_HEVC`, `128x128..8192x4352`, `10-bit`, `NV12`/`P010`). `AV1` encode (`AMFVideoEncoderHW_AV1`) cleanly reports `NOT_SUPPORTED` (`CreateComponent returned error 30`, expected for RDNA2 VCN 3.0).
- **Hardware Decoders (5)**: `H.264/AVC` (`..4096x4096`), `HEVC/H.265`, `VP9`, `AV1` (`..8192x4320`, `10-bit`, `interlaced=yes`, `NV12`/`P010`), and `MJPEG` (`..8192x4320`, `NV12`/`YUY2`).

---

## 9. Milestone 4B — RX 6900 XT Vulkan Compute Practice (GGUF `Q4_K` GEMV & Causal FlashAttention)

Pure-Rust `naga` WGSL -> SPIR-V 1.3 Wave32 shaders targeting RDNA2 (`gfx1030`, Windows 11 `26300`, Adrenalin `26.11.2`):

### 9.1 GGUF `Q4_K` Super-Block Wave32 GEMV (`warmup = 10`, `iters = 50`, `--stability-runs 0`)

Implements upstream `llama.cpp` `block_q4_K` (`256` weights / `144` bytes per super-block = **`7.11x`** compression vs FP32: `f16` `(d, dmin)`, 12-byte packed 6-bit sub-block scales & mins, 128-byte 4-bit nibbles) with zero LDS and a single `subgroupAdd`:

| Kernel | Shape | Mode | Median (ms) | Min / P95 / Max (ms) | IQR / MAD (ms) | Throughput | Max Abs Err | RMSE | Accuracy |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `gemv_q4_k_wave32_4096x4096` | `4096x4096` | `batch=1` | 0.2001 | 0.0469 / 0.2017 / 0.2034 | 0.1500 / 0.0008 | 47.33 GB/s (`167.71 GFLOP/s`) | 1.526e-4 | 3.621e-5 | Pass |
| `gemv_q4_k_wave32_4096x4096` | `4096x4096` | **`batch=16`** | **0.0491** (**4.08x**) | 0.0442 / 0.0500 / 0.0502 | **0.0006** / 0.0003 | **192.73 GB/s** (`682.90 GFLOP/s`) | 1.526e-4 | 3.621e-5 | Pass |
| `gemv_q4_k_wave32_11008x4096` | `11008x4096` | `batch=1` | 0.2268 | 0.1016 / 0.4707 / 0.4718 | 0.3377 / 0.1188 | 112.10 GB/s (`397.64 GFLOP/s`) | 1.602e-4 | 3.595e-5 | Pass |
| `gemv_q4_k_wave32_11008x4096` | `11008x4096` | **`batch=16`** | **0.1063** (**2.13x**) | 0.1051 / 0.1072 / 0.1078 | **0.0006** / 0.0003 | **239.14 GB/s** (`848.27 GFLOP/s`) | 1.602e-4 | 3.595e-5 | Pass |

- **Tradeoff vs `Q4_0` & batching stability**: `Q4_K` achieves higher weight compression (`7.11x` vs `Q4_0`'s `6.40x`) and sub-block dynamic range via asymmetric 6-bit `(scale, min)` unpacking, at higher per-super-block ALU density than `Q4_0` (`0.1063 ms` / `239.14 GB/s` vs `Q4_0`'s `0.0644 ms` / `438.74 GB/s` at `11008x4096`, `batch=16`). Batching `batch=16` collapses `11008x4096` IQR from `0.3377 ms` to **`0.0006 ms`** (`CV = 0.51%`).

### 9.2 Tiled Wave32 Causal FlashAttention (`warmup = 5`, `iters = 20`, `--stability-runs 0`, `batch = 1`)

Streams `BC = 16` key/value tiles through **`16 KiB` LDS** (`head_dim <= 128`), keeping score accumulators in registers with workgroup-uniform upper-triangle tile skipping and online softmax:

| Kernel | Shape (`H x S x D`) | Dtype | Median (ms) | Min / P95 / Max (ms) | IQR / MAD (ms) | CV | Throughput | Score HBM Saved | Max Abs Err | RMSE | Accuracy |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `flash_attn_f32_wave32_16x1024x64` | `16x1024x64` (causal) | `f32` | **2.3772** | 2.3673 / 2.3873 / 2.3884 | 0.0074 / 0.0037 | 0.23% | **904.26 GFLOP/s** | **64.0 MiB** | 1.267e-7 | 1.579e-8 | Pass |
| `flash_attn_f32_wave32_8x1024x128` | `8x1024x128` (causal) | `f32` | **1.9096** | 1.8986 / 1.9243 / 1.9399 | 0.0105 / 0.0051 | 0.49% | **1125.66 GFLOP/s** | **32.0 MiB** | 1.416e-7 | 1.445e-8 | Pass |

- **Comparison context vs Milestone 1 Triton Attention**: Milestone 1's Triton FP16 non-causal `1x16x1024x64` kernel reached `0.657 ms` (`6.54 TFLOP/s`) using `triton-windows` in FP16; this standalone Vulkan kernel executes full **FP32** causal online softmax (`1.27e-7` / `1.42e-7` error vs FP64 CPU reference) in pure Rust + Adrenalin Vulkan with zero Python/ROCm runtime dependencies and zero `S x S` score matrix allocation.

---

## 10. Milestone 4C — AMD AMF VCN 3.0 Hardware Video Encode Benchmark (`amfrt64.dll` `1.4.37.0`)

Pure-Rust dynamic COM vtable hardware video encoding via `amfrt64.dll` (`1.4.37.0`) on RX 6900 XT (`gfx1030`, VCN 3.0, Windows 11 Build `26300`, Adrenalin `26.11.2`). A 16-surface ring of synthetic moving-gradient frames is pre-allocated in `AMF_MEMORY_HOST`, populated with `NV12` (8-bit 4:2:0) or `P010` (10-bit 4:2:0, MSB-aligned `u16`) Y/UV planes, and converted to `AMF_MEMORY_DX11` VRAM textures via `AMFData::Convert(AMF_MEMORY_DX11)` **before** timing so CPU host-to-device memory copies are excluded from VCN 3.0 hardware encode measurements (`--warmup 15 --frames 120 --preset speed --fps 60`):

| Resolution | Codec | Component ID | Surface Format | Target Bitrate | Encode FPS | Throughput (MPix/s) | Avg Frame (ms) | Effective Bitrate | Output Packets | Annex-B NAL |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `1920x1080` (1080p) | `H.264/AVC` | `AMFVideoEncoderVCE_AVC` | `NV12` (8-bit) | 12.0 Mbps | **413.13 FPS** | **856.66 MPix/s** | **2.421 ms** | 13.00 Mbps | `120/120` | **VALID** |
| `1920x1080` (1080p) | `HEVC/H.265` | `AMFVideoEncoderHW_HEVC` | `NV12` (8-bit) | 12.0 Mbps | **417.74 FPS** | **866.22 MPix/s** | **2.394 ms** | 12.81 Mbps | `120/120` | **VALID** |
| `1920x1080` (1080p) | `HEVC/H.265 10-bit` | `AMFVideoEncoderHW_HEVC` | `P010` (10-bit) | 12.0 Mbps | **412.85 FPS** | **856.09 MPix/s** | **2.422 ms** | 10.77 Mbps | `120/120` | **VALID** |
| `3840x2160` (4K UHD) | `H.264/AVC` | `AMFVideoEncoderVCE_AVC` | `NV12` (8-bit) | 35.0 Mbps | **120.30 FPS** | **997.79 MPix/s** | **8.313 ms** | 36.07 Mbps | `120/120` | **VALID** |
| `3840x2160` (4K UHD) | `HEVC/H.265` | `AMFVideoEncoderHW_HEVC` | `NV12` (8-bit) | 35.0 Mbps | **117.94 FPS** | **978.23 MPix/s** | **8.479 ms** | 38.08 Mbps | `120/120` | **VALID** |
| `3840x2160` (4K UHD) | `HEVC/H.265 10-bit` | `AMFVideoEncoderHW_HEVC` | `P010` (10-bit) | 35.0 Mbps | **116.46 FPS** | **965.94 MPix/s** | **8.587 ms** | 35.00 Mbps | `120/120` | **VALID** |

Observed VCN 3.0 hardware characteristics via `forge-profiler` ADLX:
- **Dedicated fixed-function VCN 3.0 media engine**: While sustaining **`412.85–417.74 FPS`** at `1080p` (`~6.9x` real-time 1080p60) and **`116.46–120.30 FPS`** at `4K UHD` (`~2.0x` real-time 4K60, approaching **`998 MPix/s`**), the 3D GFX shader core stays completely idle at **`68–73 MHz`** (`62–138 MHz` range) and total GPU ASIC power draw is only **`37.0 W`** at 1080p (max `41.0 W`) and **`44.0 W`** at 4K (max `45.0 W`, `36–37 C` core / `43–44 C` hotspot).
- **Zero 10-bit `P010` throughput penalty**: Hardware 10-bit Main10 HEVC (`P010`) encodes within `~1.2%` of 8-bit `NV12` HEVC throughput on VCN 3.0 (`412.85` vs `417.74 FPS` at 1080p; `116.46` vs `117.94 FPS` at 4K).

---

## 11. Reproduction Commands

```powershell
# 1. Build release CLI and optional desktop GUI
cargo build --release -p forge-cli -p forge-desktop

# 2. Diagnostics + live ADLX telemetry + AMF video probe & VCN hardware encode benchmark + Vulkan/HIP verification
.\target\release\forge.exe doctor --verify --python <venv>\Scripts\python.exe
.\target\release\forge.exe video caps
.\target\release\forge.exe video bench --codec all --width 1920 --height 1080 --fps 60 --bitrate-mbps 12 --preset speed --warmup 15 --frames 120 --output ..\video_bench_1080p.json
.\target\release\forge.exe video bench --codec all --width 3840 --height 2160 --fps 60 --bitrate-mbps 35 --preset speed --warmup 15 --frames 120 --output ..\video_bench_4k.json

# 3. Vulkan compute benchmark (baseline + tiled vec4 + INT8 DP4A OpSDot + Wave32 decode + FlashAttention + stability)
# Note: decode/elementwise kernels default to --batch-dispatches 16 (override with 1..=256); heavy GEMM & FlashAttention stay at 1.
.\target\release\forge.exe bench vulkan --kernel all --warmup 5 --iters 20 --stability-runs 30 --output ..\vulkan_bench.json

# 3b. Batched Milestone 3C / 4B decode, elementwise & FlashAttention benchmarks
.\target\release\forge.exe bench vulkan --kernel gemv-w8 --batch-dispatches 16 --warmup 10 --iters 50 --stability-runs 0 --output ..\gemv_w8.json
.\target\release\forge.exe bench vulkan --kernel gemv-q4 --batch-dispatches 16 --warmup 10 --iters 50 --stability-runs 0 --output ..\gemv_q4.json
.\target\release\forge.exe bench vulkan --kernel gemv-q4-k --batch-dispatches 16 --warmup 10 --iters 50 --stability-runs 0 --output ..\gemv_q4_k.json
.\target\release\forge.exe bench vulkan --kernel fused-rms-rope --batch-dispatches 16 --warmup 10 --iters 50 --stability-runs 0 --output ..\fused_rms_rope.json
.\target\release\forge.exe bench vulkan --kernel flash-attn --warmup 5 --iters 20 --stability-runs 0 --output ..\flash_attn.json
.\target\release\forge.exe bench vulkan --kernel rmsnorm --batch-dispatches 16 --warmup 10 --iters 50 --stability-runs 0 --output ..\rmsnorm.json
.\target\release\forge.exe bench vulkan --kernel rope --batch-dispatches 16 --warmup 10 --iters 50 --stability-runs 0 --output ..\rope.json

# 4. Safe 256x256 Triton verification pass (smoke mode)
.\target\release\forge.exe bench triton --python <venv>\Scripts\python.exe --smoke --output ..\triton_smoke.json

# 5. llama.cpp Vulkan vs ROCm/HIP comparison (ensure TheRock ROCm bin dirs are on PATH for HIP)
$env:PATH = "<venv>\Lib\site-packages\_rocm_sdk_libraries\bin;<venv>\Lib\site-packages\_rocm_sdk_core\bin;" + $env:PATH
.\target\release\forge.exe bench llama `
  --model <models>\qwen2.5-1.5b-instruct-q4_k_m.gguf `
  --vulkan-bin <llama-vk>\llama-bench.exe `
  --hip-bin <llama-hip>\llama-bench.exe `
  --prompt-tokens 512 --gen-tokens 128 --reps 5 --ngl 99 `
  --vulkan-device Vulkan0 --hip-device ROCm0 --flash-attn 1 `
  --output ..\llama_bench.json
```
