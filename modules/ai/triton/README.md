# Forge AI — PyTorch ROCm & Triton Benchmark Module

Reproducible Python benchmark and numerical correctness suite for AMD Radeon GPUs on Windows using:

- PyTorch + ROCm/TheRock multi-arch wheels (`torch[device-gfx1030]`)
- `triton-windows` (`>=3.9,<3.10` for PyTorch 2.15)

## Measured Kernels

1. `pytorch_gemm_f32` (`2048x2048`, `4096x4096`)
2. `pytorch_gemm_f16` (`2048x2048`, `4096x4096`)
3. `pytorch_int8_gemm` (`torch._int_mm` probe; records explicit error if unsupported on `gfx1030`)
4. `pytorch_rmsnorm_f16` (`4096x4096`) and `triton_rmsnorm_f16` (`4096x4096` + unaligned `127x4099`)
5. `pytorch_rope_f16` (`1x2048x32x128`) and `triton_rope_f16` (`1x2048x32x128`)
6. `pytorch_sdpa_f16` (`MATH`, `EFFICIENT_ATTENTION`, `FLASH_ATTENTION` tested individually)
7. `triton_gemm_f16` (`2048x2048x2048` + unaligned `1027x513x1029`)
8. `triton_attention_f16` (`1x16x1024x64` tiled online-softmax attention)
9. Repeated stability workload (`100` consecutive iterations)

## Usage

Run directly or via `forge bench triton`:

```powershell
forge bench triton --python <path-to-venv>\Scripts\python.exe --warmup 10 --iters 50
```
