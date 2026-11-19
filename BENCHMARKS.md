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

## 6. Reproduction Commands

```powershell
# 1. Build release CLI
cargo build --release -p forge-cli

# 2. Vulkan compute benchmark & stability check
.\target\release\forge.exe bench vulkan --kernel all --warmup 10 --iters 50 --stability-runs 50 --output vulkan_bench.json

# 3. PyTorch ROCm 10.2 + triton-windows benchmark suite
.\target\release\forge.exe bench triton --python <venv>\Scripts\python.exe --warmup 10 --iters 50 --output triton_bench.json

# 4. llama.cpp Vulkan vs ROCm/HIP comparison (ensure TheRock ROCm bin dirs are on PATH for HIP)
$env:PATH = "<venv>\Lib\site-packages\_rocm_sdk_libraries\bin;<venv>\Lib\site-packages\_rocm_sdk_core\bin;" + $env:PATH
.\target\release\forge.exe bench llama `
  --model <models>\qwen2.5-1.5b-instruct-q4_k_m.gguf `
  --vulkan-bin <llama-vk>\llama-bench.exe `
  --hip-bin <llama-hip>\llama-bench.exe `
  --prompt-tokens 512 --gen-tokens 128 --reps 5 --ngl 99 `
  --vulkan-device Vulkan0 --hip-device ROCm0 `
  --output llama_bench.json
```
