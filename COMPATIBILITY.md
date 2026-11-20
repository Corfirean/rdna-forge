# Compatibility & Upstream Feasibility Audit

**Audit Date:** 2026-11-20  
**Reference Hardware:** AMD Radeon RX 6900 XT (`gfx1030`, RDNA2, 16 GiB VRAM) + AMD Ryzen 7 7700X iGPU (`gfx1036`, RDNA2)  
**Host OS & Driver:** Windows 11 Home (Version `10.0.26300`, Build `26300`, x64), AMD Software: Adrenalin Edition `26.11.2` (`32.0.21045.29003`)

Status legend:
- **Verified**: Directly built, executed, and validated on target hardware.
- **Partially verified**: Functional with confirmed constraints, missing sub-components, or community-only support.
- **Experimental**: Available upstream or in preview without stability guarantees on target hardware.
- **Unsupported**: Confirmed broken, excluded, or dropped by upstream.
- **Not tested**: Not executed on physical hardware in this repository yet.

### Evidence levels and the capability ladder

A built-in hardware profile (`profiles/gfx*/`) or a PCI-ID mapping is a *software description* of a GPU. It is **not** evidence that the GPU works. Claims in this repository use one of these evidence levels:

| Evidence level | Meaning | Applies today to |
| :--- | :--- | :--- |
| **Hardware-tested** | Executed by a maintainer on the physical GPU, with results recorded in this repository | RX 6900 XT (`gfx1030`) only; Raphael iGPU (`gfx1036`) partially |
| **Community-tested** | Reported by an external user through a hardware report (`forge report`) and reviewed | None yet |
| **Build-tested** | Compiles and passes hardware-independent tests in CI | All crates and all 18 profile files (parse/validation only) |
| **Experimental** | Implemented but not validated on target hardware | RDNA3/RDNA3.5/RDNA4 planners and AV1 encode gating; ROCm/Triton paths on `gfx1030` are partial |
| **Unsupported** | Confirmed broken or dropped upstream | See table below |
| **Unknown** | No information | Any GPU not listed |

For a given GPU and feature, the stages are cumulative and a later stage is only claimed when the earlier ones and its own check have actually run: *detected* -> *runtime available* -> *capability discovered* -> *compiles* -> *executes* -> *correctness verified* -> *performance measured* -> *hardware qualified*. `forge doctor` reports the first three; `forge doctor --verify`, `cargo test -- --ignored` (hardware qualification workflow) and `forge bench` cover the later ones. RDNA1, RDNA3, RDNA3.5 and RDNA4 entries stop at *detected / capability discovered by profile* until someone runs those stages on real hardware. No cross-generation performance improvement is claimed.

---

## 1. Upstream Ecosystem Feasibility Audit

| Upstream Project | Version / Tag Checked | License | Windows & `gfx1030` Status | Key Findings & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| [ROCm/TheRock](https://github.com/ROCm/TheRock) | `therock-10.2` (2026-11-18) | MIT | **Verified** (Python wheels) / **Partially verified** (SDK CLI/installer) | `SUPPORTED_GPUS.md` marks `gfx1030` on Windows as Build Passing, Sanity Tested, and Release Ready. Stable multi-arch wheels (`rocm-sdk-device-gfx1030==10.2.0`, `torch==2.15.0+rocm10.2.0`) run on Windows 11. `docs/development/windows_support.md` marks `rocm_smi_lib`, `amdsmi`, `ROCR-Runtime`, `rocminfo`, `rocprofiler-*`, `aqlprofile`, `rccl`, `hipFile`, `rocJPEG`, `hipSPARSELt`, `rocRoller`, and `rocgdb` as Unsupported on Windows; native Windows installers are still planned (`#1987`). `HSA_OVERRIDE_GFX_VERSION` is unsupported on Windows (`ROCm/ROCm#2654`). |
| [triton-lang/triton](https://github.com/triton-lang/triton) | `v3.9.0` (2026-10-27) | MIT | **Unsupported** | PR `#12042` (merged 2026-09-30) explicitly removed RDNA1 and RDNA2 target support upstream. |
| [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows) | `v3.9.0-windows.post30` (2026-10-28) | MIT | **Partially verified** | Succeeds `woct0rdho/triton-windows` (archived). Runs FP16 GEMM, L2-swizzled GEMM, W8A16/W4A16 weight-only dequant GEMM, RMSNorm, RoPE, fused RMSNorm+RoPE, and tiled causal/non-causal attention on `gfx1030` (`num_stages=1`, `num_warps=4`). No RDNA2 CI (`#41`). Full `2048³`/`4096³` consecutive stress triggered `Kernel-Power 41` resets on this host; `--smoke` (`256x256`) passes 8/8 kernels cleanly. |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `v1.0.0` / `b12532` (2026-11-19) | MIT | **Verified** (Vulkan) / **Partially verified** (ROCm) | Vulkan runs out of the box (`240.21 tok/s` `tg128`, `5227.71` `pp512` default; `6211.14` `pp512` with `-fa 0`). ROCm zip omits `hipblas.dll`/`rocblas.dll` (requires external ROCm `bin` on `PATH`). With TheRock libs + uncontended VRAM, `ROCm0` reaches `7369.46 tok/s` `pp512` (+41%) and `233.50 tok/s` `tg128` (`-fa 1` recommended), **but degrades 4.3x to `53.71 tok/s` under WDDM VRAM contention because `hipMemGetInfo` does not reflect non-HIP WDDM allocations**. |
| [GPUOpen-LibrariesAndSDKs/ADLX](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX) | `V1.4` / `v2.0` (`1.4.1.122`) | Custom AMD ADLX SDK License (PDF) | **Verified** (COM telemetry + read-only tuning) | `forge-profiler` dynamically loads `amdadlx64.dll` via pure-Rust `V1.4` COM vtables (no vendored SDK headers). Live RX 6900 XT telemetry (`monitor`, `doctor`), read-only tuning inspection (`tune show`), and dry-run range validation verified. `SetPowerLimit` / `ResetToFactory` require explicit `--apply` and were verified via dry-run bounds checking without mutating user settings. Must not poll ADLX concurrently with active ROCm/HIP kernels. |
| [GPUOpen-LibrariesAndSDKs/FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) | `v2.4.0` (2026-10-28) | MIT (headers/samples) / Proprietary binary (`signedbin`) | **Not tested** | FSR 4.1.2 ML-Upscaler docs list `RX 6900 XT` (`Full INT8`, 4K `3190 us`). Prebuilt `signedbin` DLLs prohibit reverse engineering/decompilation. Deferred from Milestone 1. |
| [GPUOpen-LibrariesAndSDKs/AMF](https://github.com/GPUOpen-LibrariesAndSDKs/AMF) | `v1.5.3` (2026-09-29) | MIT + Media Technologies patent disclaimer | **Verified** (`1.4.37.0` COM probe + VCN 3.0 encode/decode) | Pure-Rust dynamic `amfrt64.dll` DX11 COM probe verifies 2 HW encoders (`H.264/AVC`, `HEVC/H.265`), 5 HW decoders (`H.264/AVC`, `HEVC/H.265`, `VP9`, `AV1`, `MJPEG`), and clean `AV1 Encode` `UnsupportedCapability` gating on RDNA2. Live VCN 3.0 DX11 encode (`forge video bench`) achieves `412.85–417.74 FPS` (`1080p`) / `116.46–120.30 FPS` (`4K UHD`); live VCN 3.0 DX11 decode (`forge video decode-bench`, `NV12`/`P010`) achieves `753.89–849.64 FPS` (`1080p`) and `307.05 FPS` (`4K HEVC`). AV1 encode (`VCN 4.0+`) is implemented & gated but **Not tested** on physical RDNA3/4 hardware. |
| [microsoft/DirectML](https://github.com/microsoft/DirectML) | `2026-04-27` commit | MIT | **Partially verified** (maintenance mode) | `DirectML.dll` (`1.15.6`) ships with Windows 11. Upstream README explicitly marks DirectML in maintenance mode (security fixes only; recommends Windows ML on Windows 11 24H2+). |
| [Blinue/Magpie](https://github.com/Blinue/Magpie) | `v0.13.0` (2026-10-17) | **GPL-3.0** | **Unsupported** (for MIT code inclusion) | Active GPL-3.0 Windows window-scaling project. Incompatible with direct inclusion in MIT-licensed RDNA Forge crates. |
| [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) | Retired (`rocm-7.2.4`; active in `ROCm/rocm-libraries`) | MIT | **Verified** (via PyTorch ROCm 10.2) | Built for Windows by TheRock; `torch.matmul` FP32 (`20.75 TFLOP/s`) and FP16 (`34.58 TFLOP/s`) verified on `gfx1030`. |
| [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) | Retired (`v1.5.0` in `ROCm/rocm-libraries`) | MIT | **Unsupported** (`gfx1030`) | `projects/hipblaslt/cmake/tensilelite_supported_architectures.cmake` excludes all `gfx10XX` targets (extops unsupported). Causes `torch._int_mm` to fail with `HIPBLAS_STATUS_INVALID_VALUE` on `gfx1030`. |

---

## 2. Hardware & Runtime Compatibility Matrix

| GPU / Target | Architecture | OS / Driver | Subsystem / Backend | Status | Evidence & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `forge-device` discovery (Vulkan + HIP + Registry + ADLX) | **Verified** | Enumerates `0x1002:0x73AF`, 15.98 GiB VRAM, 80 CUs / 40 WGPs, ReBAR disabled (`256 MiB` BAR). |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | Vulkan compute (`ash` + `naga` WGSL->SPIR-V 1.3 & 1.6, Wave32 subgroups) | **Verified** | Baseline GEMM, register-blocked `128x64` `vec4` GEMM (**13.61 TFLOP/s**, 9.50x speedup), `OpSDot` INT8 DP4A GEMM (**30.29 TOPS**, bit-exact), Vulkan 1.3 `subgroup_size_control` Wave32 `W8A32` GEMV (`207.35 GB/s` single -> **`835.67 GB/s`** at `batch=16`), `Q4_0` GEMV (`93.97 GB/s` single -> **`438.74 GB/s`** / `IQR=0.0003 ms`), GGUF `Q4_K` GEMV (`7.11x` wt comp, **`239.14 GB/s`** / `IQR=0.0006 ms` at `batch=16`), fused subgroup `RMSNorm+RoPE` (**`936.98 GB/s`** at `batch=16`), `16 KiB` LDS tiled causal FlashAttention (**`1125.66 GFLOP/s`** at `8x1024x128`, `904.26 GFLOP/s` at `16x1024x64`), edge cases, and stability pass. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | AMD AMF (`amfrt64.dll` `1.4.37.0` COM DX11 probe & VCN 3.0 encode/decode, `forge video {caps,bench,decode-bench}`) | **Verified** | Pure-Rust COM vtable DX11 probe verifies **2 HW encoders** (`H.264/AVC`, `HEVC/H.265` 10-bit), **5 HW decoders** (`H.264/AVC`, `HEVC/H.265`, `VP9`, `AV1`, `MJPEG`), and clean `AV1 Encode` `UnsupportedCapability` gating (`--codec all` auto-skips AV1 on RDNA2; `--codec av1` fails cleanly). Live VCN 3.0 DX11 encode (`H.264 NV12`, `HEVC NV12`, `HEVC 10-bit P010`) achieves **`412.85–417.74 FPS`** (`1080p`, `37 W`) and **`116.46–120.30 FPS`** (`4K UHD`, `966–998 MPix/s`, `44 W`); live VCN 3.0 DX11 decode achieves **`753.89–849.64 FPS`** (`1080p`, `34 W`) and **`307.05 FPS`** (`4K HEVC`, `2546.79 MPix/s`, `38 W`) with verified `AMF_SURFACE_IID` surfaces. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | ADLX live telemetry & tuning (`forge monitor`, `forge tune`) | **Verified** (read-only + dry-run) | Live `monitor` (`table`/`jsonl`/`csv`) and `tune show` verified (`Power` `-10..=12%`, `GFX` `500..=4000 MHz` / `900..=1200 mV`, `VRAM` `2000..=2624 MHz` step 2 + Default/Fast Timing, `Fan` `20..=100%` / `25..=100C` 5-pt curve + Zero-RPM). Opt-in `--apply` validated by hardware range gates, tested read-only (`dry_run=true`). |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | Adrenalin HIP driver (`amdhip64_6.dll` `6.4.50101`) | **Verified** | `hipGetDevicePropertiesR0600` (`1472` bytes) and 4 MiB bit-exact H2D/D2H round-trip pass. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | TheRock `rocm==10.2.0` + `torch==2.15.0+rocm10.2.0` | **Verified** (FP32/FP16) / **Unsupported** (INT8, Flash/Efficient SDPA) | FP32/FP16 GEMM, RMSNorm, RoPE, Math SDPA pass. `torch._int_mm`, Flash SDPA, and Efficient SDPA fail cleanly; Triton W8A16/W4A16/causal kernels fill the missing quantized & fused-attention paths. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `triton-windows` `3.9.0.post30` | **Partially verified** | M1 suite passed (`100/100` stability); M2 kernels (L2-swizzled FP16, W8A16, W4A16, fused RMSNorm+RoPE, causal FlashAttention) verified via `--smoke` (`256x256`). Large `2048³`/`4096³` stress sweeps caused `Kernel-Power 41` resets on this host. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `llama.cpp` `b12532` Vulkan (`Vulkan0`) | **Verified** | Uncontended `240.21 tok/s` `tg128` / `5227.71 tok/s` `pp512` (default) and `6211.14 tok/s` `pp512` with `-fa 0`. |
| AMD Radeon RX 6900 XT (`gfx1030`) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | `llama.cpp` `b12532` ROCm (`ROCm0`) | **Partially verified** | Requires `hipblas.dll`/`rocblas.dll` on `PATH`. Fast when VRAM uncontended (`7369.46` `pp512`, `233.50` `tg128`, HIP graphs active), but collapses to `53.71 tok/s` under WDDM VRAM pressure. |
| AMD Radeon Graphics (`gfx1036`, Raphael iGPU) | RDNA2 | Windows 11 `26300` / Adrenalin `26.11.2` | Device discovery & ADLX (`Vulkan1` / `ROCm1` / `amdadlx64.dll`) | **Partially verified** | Detected as integrated `gfx1036` (`0x164E`, `0.50 GiB`); ADLX `IADLXGPUTuningServices` cleanly reports all manual tuning flags `false` (no tuning ranges exposed). Compute kernels not benchmarked separately. |
| AMD Radeon RX 6950 XT / RX 6800 XT (`gfx1030`), RX 6700 XT (`gfx1031`), RX 6600 XT (`gfx1032`) | RDNA2 | Windows 11 | Built-in profiles (`rx6950xt`, `rx6800xt`, `rx6700xt`, `rx6600xt`), PCI auto-match & LLM VRAM planner | **Not tested** | Profile definitions under `profiles/gfx1030`..`gfx1032`, CU/VRAM disambiguation on shared PCI IDs (`0x73BF`), and dynamic VCN 3.0 / WDDM memory-priority tweaks ready; no physical hardware tested. |
| AMD Radeon RX 7900 XTX / RX 7900 XT / PRO W7900 (`gfx1100`), RX 7800 XT / RX 7700 XT (`gfx1101`), RX 7600 XT (`gfx1102`), Radeon 780M (`gfx1103`) | RDNA3 | Windows 11 | Built-in profiles (`rx7900xtx`, `rx7900xt`, `pro_w7900`, `rx7800xt`, `rx7700xt`, `rx7600xt`, `radeon780m`), WMMA/BF16 capability gating & `rocm_wmma_hipblaslt` | **Not tested** | Models 12–96 CUs, `4–48 GiB` VRAM, WMMA/BF16, `VCN 4.0`, and `ROCBLAS_USE_HIPBLASLT=1` LLM profiles; no physical hardware tested. |
| AMD Radeon 890M (`gfx1150`, Strix Point) & Radeon 8060S (`gfx1151`, Strix Halo) | RDNA3.5 | Windows 11 | Built-in profiles (`radeon890m`, `radeon8060s`), `0x150E`/`0x1586` PCI mapping & unified APU memory planner | **Not tested** | Handles shared RAM scaling & BIOS UMA / Adrenalin VGM warnings for Strix Point / Strix Halo APUs alongside WMMA/BF16 and `ROCBLAS_USE_HIPBLASLT=1`; no physical hardware tested. |
| AMD Radeon RX 9070 XT / RX 9070 / AI PRO R9700 (`gfx1201`), RX 9060 XT (`gfx1200`) | RDNA4 | Windows 11 | Built-in profiles (`rx9070xt`, `rx9070`, `ai_pro_r9700`, `rx9060xt`), FP8/BF16/WMMA capability gating & `rocm_wmma_hipblaslt` | **Not tested** | Models 32–64 CUs, `16–32 GiB` VRAM, native FP8/BF16/WMMA, `VCN 5.0`, and `ROCBLAS_USE_HIPBLASLT=1` LLM profiles; no physical hardware tested. |
| RDNA1 (`gfx1010`, `gfx1012`) & Other AMD (`gfx1034`, `gfx1152`) | RDNA1 / RDNA2 / RDNA3.5 | Windows 11 | PCI lookup & Vulkan-only fallback policy | **Not tested** | RDNA1 triggers Vulkan-only LLM guidance (no Windows ROCm / INT8 DP4A); no physical hardware tested. |

---

## 3. Confirmed Technical Blockers

1. **No INT8 GEMM via `hipBLASLt` / `torch._int_mm` on `gfx1030`**: `hipBLASLt` omits all `gfx10XX` targets, causing `HIPBLAS_STATUS_INVALID_VALUE` at runtime. Quantized matmul on `gfx1030` requires custom DP4A (`V_DOT4_I32_I8` / `V_DOT8_I32_I4`) HIP or Vulkan shaders (as `llama.cpp` uses) rather than `hipBLASLt`.
2. **No PyTorch Flash/Efficient SDPA Kernels for `gfx1030` on Windows**: Both `SDPBackend.FLASH_ATTENTION` and `SDPBackend.EFFICIENT_ATTENTION` raise `RuntimeError: No available kernel`. Workloads currently fall back to slow unfused math SDPA unless replaced by Triton or custom attention kernels.
3. **Upstream Triton Dropped RDNA1/RDNA2**: `triton-lang/triton` PR `#12042` deleted RDNA1/RDNA2 target handling. Windows users depend on `triton-lang/triton-windows` community maintenance (`#41`).
4. **Incomplete `llama.cpp` Windows ROCm Standalone Distribution**: `llama-b12532-bin-win-rocm-10.0-x64.zip` omits `hipblas.dll` and `rocblas.dll`, causing silent backend load failure without an external ROCm `bin` directory on `PATH`.
5. **Windows WDDM `hipMemGetInfo` vs True Physical VRAM Occupancy**: On Windows 11 WDDM, `hipMemGetInfo` only reflects allocations known to HIP and still reported `16227 MiB free` when ADLX (`IADLXGPUMetrics::GPUVRAM`) and Windows PerfCounters showed `14.3–14.5 GiB` (`88%`) already allocated by another process. Running `llama.cpp` ROCm in that state triggered silent WDDM system-RAM spill (`53.71 tok/s` vs clean `233.50 tok/s`).
6. **Windows ROCm Telemetry & Profiling Gaps**: `rocm_smi_lib`, `amdsmi`, `rocprofiler-sdk`, `aqlprofile`, and `rocgdb` are unsupported on Windows per `ROCm/TheRock`. Hardware monitoring on Windows uses ADLX (`amdadlx64.dll` via `forge-profiler`), Vulkan/HIP queries, or ETW/WDDM counters—avoiding concurrent ADLX background polling during active ROCm/HIP dispatches.
7. **Windows Platform Quirks Verified Live**:
   - `hipEventElapsedTime` on Windows WDDM can emit negative timestamps if streams are not synchronized before recording and after event completion.
   - `hipDeviceProp_tR0600::memoryBusWidth` returns `0` on Windows `amdhip64_6.dll`.
   - `Win32_VideoController.AdapterRAM` caps at `4 GiB` (`u32` overflow), requiring Registry `HardwareInformation.qwMemorySize` or HIP `hipDeviceTotalMem`.
   - Vulkan `maxBufferSize` / `maxMemoryAllocationSize` caps at `2 GiB` (`0x80000000`) on Adrenalin `26.11.2`.
   - `LongPathsEnabled = 0` on default Windows 11 installations risks `pip install` path failures for deep ROCm/LLVM wheel trees unless short venv paths are used or long paths are enabled.
   - Full `2048³`/`4096³` consecutive ROCm/Triton benchmark bursts triggered `Kernel-Power 41` (`BugcheckCode = 0x0`) hard resets on the RX 6900 XT test host (requiring conservative kernel configs / `--smoke` verification mode).

---

## 4. Upstream Issue & PR Opportunities

1. **`ggml-org/llama.cpp`**:
   - **Packaging**: Either document the external `rocm[libraries]` `PATH` dependency prominently in Windows ROCm release notes or ship the minimal required `hipblas.dll` / `rocblas.dll` + Tensile files in a companion runtime archive (mirroring `cudart-llama-bin-win-cuda-*.zip`).
   - **WDDM VRAM contention caveat**: Report the verified finding that `hipMemGetInfo` misses non-HIP Windows WDDM VRAM pressure (causing `tg128` to drop from `233.5` to `53.71 tok/s`), and note `-fa 0` vs `-fa 1` tradeoffs (`-fa 1` faster on `ROCm0`; `-fa 0` faster prefill on `Vulkan0`).
2. **`triton-lang/triton-windows` (Issue `#41`)**:
   - Contribute automated RDNA2 (`gfx1030`) regression matrix results covering PyTorch `2.15.0+rocm10.2.0` + `triton-windows` `3.9.0.post30` (verified passing for FP16 GEMM, RMSNorm, RoPE, and tiled attention).
3. **`ROCm/TheRock` (Issue `#1987` & Windows HIP runtime)**:
   - Report Windows WDDM `hipEventElapsedTime` negative skew when events are queued without pre-sync, and track native Windows installer delivery.
4. **`ROCm/rocm-libraries`**:
   - Track `gfx1030` INT8 DP4A GEMM gaps where `hipBLASLt` excludes `gfx10XX` and PyTorch `torch._int_mm` fails without a `rocBLAS` fallback.
