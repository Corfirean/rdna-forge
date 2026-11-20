# Architecture

RDNA Forge uses a modular Rust workspace for device discovery, runtime probing, policy enforcement, benchmarking, and CLI orchestration, paired with native GPU compute technologies (WGSL/SPIR-V via Vulkan, HIP runtime FFI, Triton/PyTorch ROCm modules, and external `llama.cpp` execution).

---

## 1. Workspace Structure

```text
rdna-forge/
├── apps/
│   ├── forge-cli/        # `forge` CLI binary (doctor, video, bench, monitor, tune, profile)
│   └── forge-desktop/    # Optional Tauri v2 + WebView2 desktop GUI (embedded vanilla ES UI)
├── crates/
│   ├── forge-core/       # Domain types, gfx target parser, profile schema, capability policy
│   ├── forge-device/     # Windows OS, Registry, Vulkan, HIP ABI, and ADLX discovery
│   ├── forge-profiler/   # Dynamic ADLX COM telemetry sampler & WDDM VRAM pressure checks
│   ├── forge-runtime/    # Backend probing, AMF COM probe, naga WGSL->SPIR-V, RAII Vulkan, live verification
│   └── forge-benchmark/  # Statistical harness, RSS memory tracking, ADLX/Vulkan/Triton/llama-bench runners
├── modules/
│   └── ai/triton/        # Reproducible PyTorch ROCm + Triton kernel suite (bench_suite.py)
├── profiles/
│   ├── gfx1030/          # RDNA2 Navi 21 profiles (rx6900xt.toml, rx6950xt.toml, rx6800xt.toml)
│   ├── gfx1031/..gfx1032/ # RDNA2 Navi 22/23 profiles (rx6700xt.toml, rx6600xt.toml)
│   ├── gfx1100/..gfx1103/ # RDNA3 & 780M profiles (rx7900xtx/xt, pro_w7900, rx7800/7700/7600xt, radeon780m)
│   ├── gfx1150/..gfx1151/ # RDNA3.5 Strix Point / Strix Halo APU profiles (radeon890m, radeon8060s)
│   └── gfx1200/..gfx1201/ # RDNA4 profiles (rx9060xt, rx9070, rx9070xt, ai_pro_r9700)
└── tests/
    ├── integration/      # Hardware-independent workspace integration tests
    └── gpu/              # Ignored live-GPU correctness and stability tests
```

Crate dependency flow (`forge-cli`, `forge-desktop` -> `forge-benchmark` -> {`forge-runtime`, `forge-profiler`} -> `forge-device` -> `forge-core`). No circular dependencies exist.

---

## 2. Module Responsibilities

### `forge-core`
- **`architecture`**: Defines `GpuArchitecture` (with `supports_hipblaslt`, `supports_rocwmma_fattn`, `supports_av1_encode`, `vcn_generation`), conservative ISA `ArchitectureCapabilities` (`has_wmma`, `has_mfma`, `has_dp4a_i8`, `has_dot8_i4`, `has_native_fp16`, `has_native_bf16`, `has_native_fp8`, wave sizes), `GfxTarget::parse` (`gfx1030`, `gfx1030:xnack-`, `gfx90a`), and expanded PCI ID lookup (`lookup_amd_pci_device` covering RDNA1–RDNA4 discrete, Radeon PRO workstation, and Strix Point / Strix Halo / Krackan Point APUs, returning `None` for unknown IDs).
- **`profile`**: Parses, validates, and embeds all 18 TOML hardware profiles (`BuiltinProfileInfo`, `builtin_profiles`, `builtin_profile_by_id`) plus token/CU/VRAM-disambiguated auto-matching (`match_profile_for_device`).
- **`policy`**: Enforces `validate_kernel_execution` (checks vendor, `maxBufferSize`, VRAM budget, FP16/BF16/FP8/INT8/WMMA/MFMA flags) and `select_preferred_device` (deterministic ordering: discrete AMD > integrated AMD > other, breaking ties by VRAM).
- **`error`**: Typed `ForgeError` enum using `thiserror` across all library crates.

### `forge-device`
Correlates four independent Windows sources without relying on broken `Win32_VideoController.AdapterRAM` (`u32` 4 GiB cap):
1. **OS / CPU / RAM (`os.rs`)**: Calls `ntdll.dll!RtlGetVersion` directly (avoiding manifest-skewed version APIs), reads `ProcessorNameString` from `HKLM`, and queries `GlobalMemoryStatusEx`.
2. **Windows Display Registry (`registry.rs`)**: Enumerates `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}` for AMD adapters, Adrenalin software version, `MatchingDeviceId` (`VEN_`/`DEV_`), and 64-bit `HardwareInformation.qwMemorySize`.
3. **Vulkan (`vulkan.rs`)**: Loads `vulkan-1.dll` dynamically via `ash` and inspects `PhysicalDeviceProperties2` / `Features2`, `SubgroupProperties`, `Vulkan13Properties` (`maxBufferSize`), `CooperativeMatrixFeaturesKHR`, `ShaderCorePropertiesAMD`, `PCIBusInfoPropertiesEXT`, and memory heaps (identifying Resizable BAR via `DEVICE_LOCAL | HOST_VISIBLE` heap size).
4. **HIP & ADLX (`hip.rs`, `adlx.rs`)**: Dynamically loads `amdhip64_7.dll` / `amdhip64_6.dll` / `amdhip64.dll` via `libloading` and reads `HipDevicePropR0600` (`1472` bytes, `gcnArchName` at offset `1160`, verified by ABI tests). Dynamically loads `amdadlx64.dll` and invokes `ADLXQueryVersion` (`const char**`) / `ADLXQueryFullVersion`.

### `forge-profiler`
- **`adlx_ffi`**: Pure-Rust `#[repr(C)]` COM vtable bindings matching ADLX `V1.4` C ABI (`IADLXSystem`, `IADLXGPUList`, `IADLXGPU`, `IADLXPerformanceMonitoringServices`, `IADLXGPUMetrics`), dynamically loading `amdadlx64.dll` via `libloading` without vendoring AMD SDK headers. Serializes process-global `ADLXInitialize`/`ADLXTerminate` lifecycle with a `Mutex<()>` and deterministically releases all child COM interfaces (including `IADLXGPUTuningServices`) before `ADLXTerminate()`.
- **`adlx_tuning` & `tuning`**: Pure-Rust ADLX `V1.4` tuning COM bindings (`IADLXGPUTuningServices`, `IADLXGPUPresetTuning`, `IADLXManualPowerTuning(1)`, `IADLXManualGraphicsTuning2(_1)`, `IADLXManualVRAMTuning2(_1)`, `IADLXManualFanTuning(1)`, and `QueryInterface` fallback to base interfaces). Exposes strictly read-only `read_all_gpu_tuning()`, both active (`GetFanTuningStates`) and VBIOS factory default (`GetDefaultFanTuningStates`) 5-point fan curves, external fan controller (`FanControl.exe` / flat PWM override) detection via Win32 `CreateToolhelp32Snapshot`, hardware-validated `IntRange` checks (`min..=max` & `step` alignment), and opt-in `set_gpu_power_limit` / `reset_gpu_tuning_to_factory` (dry-run by default; mutation gated behind explicit `dry_run = false` / `--apply`).
- **`sampler` & `types`**: Exposes single-shot `snapshot_all_gpus()` / `snapshot_for_device()`, background `TelemetryGuard` sampling, `TELEMETRY_TABLE_HEADER` / `TELEMETRY_CSV_HEADER` / `MonitorRecord` streaming formatters, and automatic Windows WDDM VRAM pressure (>50% dedicated VRAM) and competing-workload (>20% usage) warnings.

### `forge-runtime`
- **`probe`**: Audits Vulkan, HIP driver vs ROCm SDK tools (`rocm-sdk`/`hipcc`), Python PyTorch ROCm + `triton-windows`, ADLX, DirectML (flagging maintenance mode), AMF (upgraded with live codec counts from `amf`), and `llama-bench`.
- **`amf`**: Pure-Rust dynamic COM vtable probe and VCN hardware video encode engine (`amfrt64.dll`, zero vendored headers) initializing a DX11 AMF context under a `Mutex` guard, querying VCN encode/decode codec capabilities (`H.264/AVC`, `HEVC/H.265`, `VP9`, `AV1`, `MJPEG`, resolution bounds, 10-bit/interlaced support, native/supported surface formats), and benchmarking hardware VCN encoding (`run_amf_encode_benchmark` across `H.264/AVC NV12`, `HEVC/H.265 NV12`, and `HEVC/H.265 10-bit P010` using a pre-uploaded 16-surface `AMF_MEMORY_DX11` ring and Annex-B NAL start-code verification).
- **`shaders`**: Embeds WGSL compute kernels (`gemm_f32.wgsl`, `gemm_f32_tiled_vec4.wgsl`, `gemm_i8_dp4a.wgsl`, `gemv_w8a32_wave32.wgsl`, `gemv_q4_0_wave32.wgsl`, `gemv_q4_k_wave32.wgsl`, `rmsnorm_f32.wgsl`, `rope_f32.wgsl`, `rmsnorm_rope_f32_subgroup.wgsl`, `flash_attention_f32_wave32.wgsl`) and compiles them purely in Rust via `naga` to SPIR-V 1.3 (`OpGroupNonUniformFAdd` subgroups + `OpBitFieldSExtract` unpacks) or SPIR-V 1.6 (`OpSDot` / `V_DOT4_I32_I8` when `shader_integer_dot_product` is supported by the device).
- **`vulkan_compute`**: `VulkanComputeEngine` manages instance, device (enabling `PhysicalDeviceVulkan13Features::{shader_integer_dot_product, subgroup_size_control, compute_full_subgroups}` when supported, with `requiredSubgroupSize = 32` for Wave32 kernels), queue, command pool, timestamp `QueryPool`, batched multi-dispatch submissions (`dispatches_per_submit` `1..=256` with inter-dispatch `SHADER_WRITE -> SHADER_WRITE | SHADER_READ` memory barriers), `DEVICE_LOCAL` storage buffers, and staging transfers with strict RAII `Drop` cleanup.
- **`reference`**: FP64-accumulated FP32, quantized `Q4_0`/`Q4_K`/`W8A32`, fused `RMSNorm+RoPE`, exact `i64` INT8, and two-pass `ref_flash_attention_f32` CPU reference implementations plus `compare_slices` / `compare_i32_exact` evaluators and IEEE-754 half / `Q4_K` super-block packing helpers.
- **`verify`**: Shared live HIP transfer and Vulkan compute capability checks (`VULKAN_VERIFY_CAPABILITIES`, `default_untested_capabilities`, `run_live_capability_checks`) consumed by both `forge-cli` and `forge-desktop`.
- **`hip_verify`**: RAII-guarded `hipMalloc` / `hipMemcpy` / `hipEvent*` 4 MiB bit-exact H2D+D2H verification against `amdhip64_*.dll`.

### `forge-benchmark`
- **`stats`**: Computes `min`, `median`, `mean`, `p95`, `max`, `MAD`, `IQR`, `stddev`, and `CV%` over warmed, repeated runs.
- **`memory`**: Captures process peak working set via Win32 `GetProcessMemoryInfo`.
- **`vulkan_suite`**, **`video`**, **`llama`**, **`llm_optimizer`**, **`triton`**: Orchestrates reproducible compute, VCN hardware video encode (`run_video_benchmark_suite`), GGUF v2/v3 header inspection (including MoE `<arch>.expert_count` & `-ncmoe` calculation), multi-GPU anti-WDDM-spill & APU unified-memory VRAM planning (`LlmTargetGpuContext`, `plan_llm_configuration_for_device`, `plan_llm_configuration_for_gpu`), architecture-tailored `llama.cpp` tweaks (including `ROCBLAS_USE_HIPBLASLT=1` on RDNA3/3.5/4 and Vulkan-only guidance on RDNA1), 1-click dry-run-safe LM Studio config & Windows/Ollama environment patching (`inspect_gguf_model`, `plan_llm_configuration`, `autotune_llm_model`, `apply_lm_studio_config`, `apply_windows_and_ollama_env_tweaks`), and LLM benchmarks (with configurable `--batch-dispatches` defaulting to `16` for sub-ms decode/elementwise kernels) and attaches ADLX telemetry (`TelemetryGuard` for native Vulkan and AMF VCN; non-overlapping pre/post snapshots for ROCm/HIP workloads, opt-in via `--telemetry` on `forge bench triton`).

### `forge-cli`
Thin CLI boundary (`clap` + `anyhow`) exposing:
- `forge doctor [--json] [--verify] [--python]`
- `forge video {caps [--json], bench [--codec all|h264|hevc|hevc10] [--width W] [--height H] [--fps F] [--bitrate-mbps M] [--preset speed|balanced|quality] [--warmup N] [--frames N] [--json] [--output PATH]}`
- `forge llm {discover [--json], inspect [--model PATH] [--json], plan [--model PATH] [--device N] [--context N] [--safety-margin-mib M] [--json] [--output PATH], autotune [--model PATH] [--device N] [--vulkan-bin PATH] [--hip-bin PATH] [--rocm-lib-dir PATH] [--context-depth D] [--reps R] [--json] [--output PATH], apply-lmstudio [--model PATH] [--device N] [--profile ID] [--context N] [--safety-margin-mib M] [--apply] [--json], apply-env [--model PATH] [--device N] [--context N] [--kv-quant f16|q8-0|q4-0] [--modelfile-out PATH] [--apply] [--json]}`
- `forge monitor [--interval-ms N] [--count N] [--duration-secs S] [--device-id HEX] [--format table|jsonl|csv] [--output PATH]` (with Win32 `SetConsoleCtrlHandler` graceful termination and clean `ADLXTerminate`)
- `forge tune {show [--device-id HEX] [--json], power-limit <PCT> [--device-id HEX] [--apply], reset [--device-id HEX] [--apply]}`
- `forge bench {vulkan [--batch-dispatches N],llama [--flash-attn] [--hip-disable-graphs],triton [--smoke] [--telemetry]}`
- `forge profile {show <PATH> [--json], list [--json], match [--device N] [--json]}`

### `forge-desktop`
Optional Windows-native Tauri v2 + Edge WebView2 application (`apps/forge-desktop`) serving embedded vanilla ES assets (`ui/`) with zero `node_modules` dependencies:
- **`commands`**: Thread-safe Tauri IPC handlers (`get_diagnostics` with `gpu_metadata` matched profiles & VCN/ISA metadata, `run_live_verification`, `get_telemetry_snapshot`, `get_tuning_report`, `set_power_limit`, `reset_tuning`, `get_video_caps`, `run_video_benchmark`, `run_vulkan_benchmark`, `get_llm_assets`, `get_llm_plan`, `run_llm_autotune`, `apply_lm_studio_patch`, `apply_env_and_ollama_patch`) synchronized via an RAII `BusyGuard` so live ADLX polling pauses automatically during exclusive GPU runs or tuning operations, forwarding selected `device_index` into LLM planning, autotuning, and patching.
- **`ui`**: Six views (`Overview`, `Monitor` with 4 rolling Canvas 2D strip charts, `Tuning` with active vs factory VBIOS fan-curve plot, external fan manager diagnostics & explicit `Unlock Hardware Writes` confirmation gate, `Video (AMF)` with codec capabilities table & VCN hardware encode benchmark runner, `Benchmarks`, and `LLM Optimizer` with GGUF/MoE header analysis, anti-WDDM-spill VRAM planner, RDNA2 tweak commands, empirical autotuner, and 1-Click LM Studio / Ollama / Windows `HKCU\Environment` Auto-Patcher).

---

## 3. Dynamic Loading & Licensing Boundary

RDNA Forge keeps all core Rust crates MIT-licensed and avoids vendoring external SDK binaries or proprietary headers:
- `vulkan-1.dll`, `amdhip64_*.dll`, and `amdadlx64.dll` are loaded dynamically at runtime if present on the user system.
- Missing DLLs or unsupported runtimes yield explicit `SupportStatus::Unsupported` diagnostics or typed `ForgeError::RuntimeUnavailable` errors rather than crashes.
- GPL-3.0 third-party projects (such as Magpie) are excluded from the codebase.

---

## 4. Extension Points

The current architecture provides clean seams for future milestones without premature plugin abstractions:
1. **HIP C++ / Composable Kernel modules** under `modules/ai/hip/` invoked through `forge-runtime` and benchmarked via `forge-benchmark`.
2. **FidelityFX & Hardware Video (`AMF`)** integration gated cleanly above `forge-device` / `forge-profiler`.
3. **Subgroup / Cooperative Matrix Vulkan shaders** (`mul_mat` / quantized dot-product WGSL or GLSL/SPIR-V kernels) gated behind `GpuDeviceInfo` capability checks.
