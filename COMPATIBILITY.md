# Compatibility & Upstream Feasibility Audit

**Audit Date:** 2026-11-19  
**Reference Hardware:** AMD Radeon RX 6900 XT (`gfx1030`, RDNA2, 16 GiB VRAM) + AMD Ryzen 7 7700X iGPU (`gfx1036`, RDNA2)  
**Host OS & Driver:** Windows 11 Home (Version `10.0.26300`, Build `26300`, x64), AMD Software: Adrenalin Edition `26.11.2` (`32.0.21045.29003`)

Status legend:
- **Verified**: Directly built, executed, and validated on target hardware.
- **Partially verified**: Functional with confirmed constraints, missing sub-components, or community-only support.
- **Experimental**: Available upstream or in preview without stability guarantees on target hardware.
- **Unsupported**: Confirmed broken, excluded, or dropped by upstream.
- **Not tested**: Not executed on physical hardware in this repository yet.

---

## 1. Upstream Ecosystem Feasibility Audit

| Upstream Project | Version / Tag Checked | License | Windows & `gfx1030` Status | Key Findings & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| [ROCm/TheRock](https://github.com/ROCm/TheRock) | `therock-10.2` (2026-11-18) | MIT | **Verified** (Python wheels) / **Partially verified** (SDK CLI/installer) | `SUPPORTED_GPUS.md` marks `gfx1030` on Windows as Build Passing, Sanity Tested, and Release Ready. Stable multi-arch wheels (`rocm-sdk-device-gfx1030==10.2.0`, `torch==2.15.0+rocm10.2.0`) run on Windows 11. `docs/development/windows_support.md` marks `rocm_smi_lib`, `amdsmi`, `ROCR-Runtime`, `rocminfo`, `rocprofiler-*`, `aqlprofile`, `rccl`, `hipFile`, `rocJPEG`, `hipSPARSELt`, `rocRoller`, and `rocgdb` as Unsupported on Windows; native Windows installers are still planned (`#1987`). `HSA_OVERRIDE_GFX_VERSION` is unsupported on Windows (`ROCm/ROCm#2654`). |
| [triton-lang/triton](https://github.com/triton-lang/triton) | `v3.9.0` (2026-10-27) | MIT | **Unsupported** | PR `#12042` (merged 2026-09-30) explicitly removed RDNA1 and RDNA2 target support upstream. |
| [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows) | `v3.9.0-windows.post30` (2026-10-28) | MIT | **Partially verified** | Succeeds `woct0rdho/triton-windows` (archived). `triton_windows-3.9.0.post30-cp311-cp311-win_amd64.whl` runs FP16 GEMM, RMSNorm, RoPE, and tiled attention on `gfx1030` with PyTorch `2.15.0+rocm10.2.0`. No RDNA2 CI exists; community maintenance tracked in issue `#41`. RDNA2 lacks WMMA/MFMA matrix cores (`V_DOT4_I32_I8`/`V_DOT8_I32_I4` only). |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `v1.0.0` / `b12532` (2026-11-19) | MIT | **Verified** (Vulkan) / **Partially verified** (ROCm) | Release workflow compiles Windows ROCm for `gfx1030`. Vulkan (`llama-b12532-bin-win-vulkan-x64.zip`) runs out of the box. Official ROCm zip (`llama-b12532-bin-win-rocm-10.0-x64.zip`) bundles `amdhip64_7.dll` and `amd_comgr.dll` but omits `hipblas.dll` / `rocblas.dll`, silently failing to load `ggml-hip.dll` unless ROCm runtime libs are on `PATH`. With TheRock 10.2 libs on `PATH`, `ROCm0` runs all layers on RX 6900 XT, but decode (`tg128`) measured `53.71 tok/s` vs Vulkan's `213.17 tok/s`. |
| [GPUOpen-LibrariesAndSDKs/ADLX](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX) | `v2.0` (2026-09-20) | Custom AMD ADLX SDK License (PDF) | **Verified** (dynamic probe) | Driver ships `amdadlx64.dll` (`1.4.1.122`). `ADLXQueryVersion` (`const char**`) and `ADLXQueryFullVersion` work dynamically via `libloading` without vendoring SDK files. |
| [GPUOpen-LibrariesAndSDKs/FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) | `v2.4.0` (2026-10-28) | MIT (headers/samples) / Proprietary binary (`signedbin`) | **Not tested** | FSR 4.1.2 ML-Upscaler docs list `RX 6900 XT` (`Full INT8`, 4K `3190 us`). Prebuilt `signedbin` DLLs prohibit reverse engineering/decompilation. Deferred from Milestone 1. |
| [GPUOpen-LibrariesAndSDKs/AMF](https://github.com/GPUOpen-LibrariesAndSDKs/AMF) | `v1.5.3` (2026-09-29) | MIT + Media Technologies patent disclaimer | **Partially verified** (DLL present) | `amfrt64.dll` detected in Windows `System32`. Deferred from Milestone 1. |
| [microsoft/DirectML](https://github.com/microsoft/DirectML) | `2026-04-27` commit | MIT | **Partially verified** (maintenance mode) | `DirectML.dll` (`1.15.6`) ships with Windows 11. Upstream README explicitly marks DirectML in maintenance mode (security fixes only; recommends Windows ML on Windows 11 24H2+). |
| [Blinue/Magpie](https://github.com/Blinue/Magpie) | `v0.13.0` (2026-10-17) | **GPL-3.0** | **Unsupported** (for MIT code inclusion) | Active GPL-3.0 Windows window-scaling project. Incompatible with direct inclusion in MIT-licensed RDNA Forge crates. |
| [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) | Retired (`rocm-7.2.4`; active in `ROCm/rocm-libraries`) | MIT | **Verified** (via PyTorch ROCm 10.2) | Built for Windows by TheRock; `torch.matmul` FP32 (`20.75 TFLOP/s`) and FP16 (`34.58 TFLOP/s`) verified on `gfx1030`. |
| [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) | Retired (`v1.5.0` in `ROCm/rocm-libraries`) | MIT | **Unsupported** (`gfx1030`) | `projects/hipblaslt/cmake/tensilelite_supported_architectures.cmake` excludes all `gfx10XX` targets (extops unsupported). Causes `torch._int_mm` to fail with `HIPBLAS_STATUS_INVALID_VALUE` on `gfx1030`. |

---

## 2. Hardware & Runtime Compatibility Matrix

| GPU / Target | Architecture | OS / Driver | Subsystem / Backend | Status | Evidence & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `forge-device` discovery (Vulkan + HIP + Registry + ADLX) | **Verified** | Enumerates `0x1002:0x73AF`, 15.98 GiB VRAM, 80 CUs / 40 WGPs, ReBAR disabled (`256 MiB` BAR). |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | Vulkan compute (`ash` + `naga` WGSL->SPIR-V) | **Verified** | GEMM, RMSNorm, RoPE, edge cases (`1x1x1`, unaligned), and 50-run stability loop pass. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | Adrenalin HIP driver (`amdhip64_6.dll` `6.4.50101`) | **Verified** | `hipGetDevicePropertiesR0600` (`1472` bytes) and 4 MiB bit-exact H2D/D2H round-trip pass. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | TheRock `rocm==10.2.0` + `torch==2.15.0+rocm10.2.0` | **Verified** (FP32/FP16) / **Unsupported** (INT8, Flash/Efficient SDPA) | FP32/FP16 GEMM, RMSNorm, RoPE, Math SDPA pass. `torch._int_mm`, Flash SDPA, and Efficient SDPA fail cleanly. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `triton-windows` `3.9.0.post30` | **Partially verified** | FP16 GEMM (aligned & unaligned), RMSNorm, RoPE, tiled Attention, and 100-iter stability pass. Community-maintained without RDNA2 CI. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `llama.cpp` `b12532` Vulkan (`Vulkan0`) | **Verified** | Offloads 29/29 layers cleanly (`4784.47 tok/s` prefill, `213.17 tok/s` decode on Qwen2.5-1.5B Q4_K_M). |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `llama.cpp` `b12532` ROCm (`ROCm0`) | **Partially verified** | Requires `hipblas.dll`/`rocblas.dll` on `PATH`; `tg128` decode measured `53.71 tok/s` (nearly 4x slower than Vulkan). |
| AMD Radeon Graphics (`gfx1036`, Raphael iGPU) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | Device discovery (`Vulkan1` / `ROCm1` / Registry) | **Partially verified** | Detected accurately as integrated `gfx1036` with 0.50 GiB dedicated VRAM; compute kernels not benchmarked separately. |
| Other RDNA2 (`gfx1031`, `gfx1032`, `gfx1034`) | RDNA2 | Windows 11 | All backends | **Not tested** | PCI IDs mapped in `forge-core`; no physical hardware tested. |
| RDNA3 (`gfx1100`–`gfx1103`) & RDNA3.5 (`gfx115x`) | RDNA3 / RDNA3.5 | Windows 11 | All backends | **Not tested** | Capability table models WMMA/BF16 support; no physical hardware tested. |
| RDNA4 (`gfx1200`, `gfx1201`) | RDNA4 | Windows 11 | All backends | **Not tested** | Capability table models WMMA/BF16/FP8 support; no physical hardware tested. |

---

## 3. Confirmed Technical Blockers

1. **No INT8 GEMM via `hipBLASLt` / `torch._int_mm` on `gfx1030`**: `hipBLASLt` omits all `gfx10XX` targets, causing `HIPBLAS_STATUS_INVALID_VALUE` at runtime. Quantized matmul on `gfx1030` requires custom DP4A (`V_DOT4_I32_I8` / `V_DOT8_I32_I4`) HIP or Vulkan shaders (as `llama.cpp` uses) rather than `hipBLASLt`.
2. **No PyTorch Flash/Efficient SDPA Kernels for `gfx1030` on Windows**: Both `SDPBackend.FLASH_ATTENTION` and `SDPBackend.EFFICIENT_ATTENTION` raise `RuntimeError: No available kernel`. Workloads currently fall back to slow unfused math SDPA unless replaced by Triton or custom attention kernels.
3. **Upstream Triton Dropped RDNA1/RDNA2**: `triton-lang/triton` PR `#12042` deleted RDNA1/RDNA2 target handling. Windows users depend on `triton-lang/triton-windows` community maintenance (`#41`).
4. **Incomplete `llama.cpp` Windows ROCm Standalone Distribution**: `llama-b12532-bin-win-rocm-10.0-x64.zip` omits `hipblas.dll` and `rocblas.dll`, causing silent backend load failure without an external ROCm `bin` directory on `PATH`.
5. **Slow `llama.cpp` ROCm Token Decode on Windows WDDM**: Even with all 29 layers offloaded and `GGML_HIP_GRAPHS` compiled in by default, `ROCm0` `tg128` measured `53.71 tok/s` vs `Vulkan0` `213.17 tok/s`.
6. **Windows ROCm Telemetry & Profiling Gaps**: `rocm_smi_lib`, `amdsmi`, `rocprofiler-sdk`, `aqlprofile`, and `rocgdb` are unsupported on Windows per `ROCm/TheRock`. Hardware monitoring on Windows must use ADLX (`amdadlx64.dll`), Vulkan/HIP queries, or ETW/WDDM telemetry.
7. **Windows Platform Quirks Verified Live**:
   - `hipEventElapsedTime` on Windows WDDM can emit negative timestamps if streams are not synchronized before recording and after event completion.
   - `hipDeviceProp_tR0600::memoryBusWidth` returns `0` on Windows `amdhip64_6.dll`.
   - `Win32_VideoController.AdapterRAM` caps at `4 GiB` (`u32` overflow), requiring Registry `HardwareInformation.qwMemorySize` or HIP `hipDeviceTotalMem`.
   - Vulkan `maxBufferSize` / `maxMemoryAllocationSize` caps at `2 GiB` (`0x80000000`) on Adrenalin `26.11.2`.
   - `LongPathsEnabled = 0` on default Windows 11 installations risks `pip install` path failures for deep ROCm/LLVM wheel trees unless short venv paths are used or long paths are enabled.

---

## 4. Upstream Issue & PR Opportunities

1. **`ggml-org/llama.cpp`**:
   - **Packaging**: Either document the external `rocm[libraries]` `PATH` dependency prominently in Windows ROCm release notes or ship the minimal required `hipblas.dll` / `rocblas.dll` + Tensile files in a companion runtime archive (mirroring `cudart-llama-bin-win-cuda-*.zip`).
   - **Performance investigation**: Profile why Windows `ROCm0` decode (`tg128` `53.71 tok/s` with `b12532`) trails `Vulkan0` (`213.17 tok/s`) by 3.97x on `gfx1030` (WDDM launch overhead / HIP graph capture validation).
2. **`triton-lang/triton-windows` (Issue `#41`)**:
   - Contribute automated RDNA2 (`gfx1030`) regression matrix results covering PyTorch `2.15.0+rocm10.2.0` + `triton-windows` `3.9.0.post30` (verified passing for FP16 GEMM, RMSNorm, RoPE, and tiled attention).
3. **`ROCm/TheRock` (Issue `#1987` & Windows HIP runtime)**:
   - Report Windows WDDM `hipEventElapsedTime` negative skew when events are queued without pre-sync, and track native Windows installer delivery.
4. **`ROCm/rocm-libraries`**:
   - Track `gfx1030` INT8 DP4A GEMM gaps where `hipBLASLt` excludes `gfx10XX` and PyTorch `torch._int_mm` fails without a `rocBLAS` fallback.
