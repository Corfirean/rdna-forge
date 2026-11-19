# Architecture

RDNA Forge uses a modular Rust workspace for device discovery, runtime probing, policy enforcement, benchmarking, and CLI orchestration, paired with native GPU compute technologies (WGSL/SPIR-V via Vulkan, HIP runtime FFI, Triton/PyTorch ROCm modules, and external `llama.cpp` execution).

---

## 1. Workspace Structure

```text
rdna-forge/
├── apps/
│   └── forge-cli/        # `forge` CLI binary (doctor, bench, profile)
├── crates/
│   ├── forge-core/       # Domain types, gfx target parser, profile schema, capability policy
│   ├── forge-device/     # Windows OS, Registry, Vulkan, HIP ABI, and ADLX discovery
│   ├── forge-runtime/    # Backend probing, naga WGSL->SPIR-V, RAII Vulkan compute, HIP transfer check
│   └── forge-benchmark/  # Statistical harness, RSS memory tracking, Vulkan/Triton/llama-bench runners
├── modules/
│   └── ai/triton/        # Reproducible PyTorch ROCm + Triton kernel suite (bench_suite.py)
├── profiles/
│   └── gfx1030/          # Declarative hardware target profiles (rx6900xt.toml)
└── tests/
    ├── integration/      # Hardware-independent workspace integration tests
    └── gpu/              # Ignored live-GPU correctness and stability tests
```

Crate dependency flow (`forge-cli` -> `forge-benchmark` -> `forge-runtime` -> `forge-device` -> `forge-core`). No circular dependencies exist.

---

## 2. Module Responsibilities

### `forge-core`
- **`architecture`**: Defines `GpuArchitecture`, conservative ISA `ArchitectureCapabilities` (`has_wmma`, `has_mfma`, `has_dp4a_i8`, `has_dot8_i4`, `has_native_fp16`, `has_native_bf16`, `has_native_fp8`, wave sizes), `GfxTarget::parse` (`gfx1030`, `gfx1030:xnack-`, `gfx90a`), and explicit PCI ID lookup (`lookup_amd_pci_device`) returning `None` for unknown IDs instead of guessing.
- **`profile`**: Parses and validates TOML hardware target files (`profiles/gfx1030/rx6900xt.toml`).
- **`policy`**: Enforces `validate_kernel_execution` (checks vendor, `maxBufferSize`, VRAM budget, FP16/BF16/FP8/INT8/WMMA/MFMA flags) and `select_preferred_device` (deterministic ordering: discrete AMD > integrated AMD > other, breaking ties by VRAM).
- **`error`**: Typed `ForgeError` enum using `thiserror` across all library crates.

### `forge-device`
Correlates four independent Windows sources without relying on broken `Win32_VideoController.AdapterRAM` (`u32` 4 GiB cap):
1. **OS / CPU / RAM (`os.rs`)**: Calls `ntdll.dll!RtlGetVersion` directly (avoiding manifest-skewed version APIs), reads `ProcessorNameString` from `HKLM`, and queries `GlobalMemoryStatusEx`.
2. **Windows Display Registry (`registry.rs`)**: Enumerates `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}` for AMD adapters, Adrenalin software version, `MatchingDeviceId` (`VEN_`/`DEV_`), and 64-bit `HardwareInformation.qwMemorySize`.
3. **Vulkan (`vulkan.rs`)**: Loads `vulkan-1.dll` dynamically via `ash` and inspects `PhysicalDeviceProperties2` / `Features2`, `SubgroupProperties`, `Vulkan13Properties` (`maxBufferSize`), `CooperativeMatrixFeaturesKHR`, `ShaderCorePropertiesAMD`, `PCIBusInfoPropertiesEXT`, and memory heaps (identifying Resizable BAR via `DEVICE_LOCAL | HOST_VISIBLE` heap size).
4. **HIP & ADLX (`hip.rs`, `adlx.rs`)**: Dynamically loads `amdhip64_7.dll` / `amdhip64_6.dll` / `amdhip64.dll` via `libloading` and reads `HipDevicePropR0600` (`1472` bytes, `gcnArchName` at offset `1160`, verified by ABI tests). Dynamically loads `amdadlx64.dll` and invokes `ADLXQueryVersion` (`const char**`) / `ADLXQueryFullVersion`.

### `forge-runtime`
- **`probe`**: Audits Vulkan, HIP driver vs ROCm SDK tools (`rocm-sdk`/`hipcc`), Python PyTorch ROCm + `triton-windows`, ADLX, DirectML (flagging maintenance mode), AMF, and `llama-bench`.
- **`shaders`**: Embeds WGSL compute kernels (`gemm_f32.wgsl`, `rmsnorm_f32.wgsl`, `rope_f32.wgsl`) and compiles them offline/in-memory to SPIR-V 1.3 via pure-Rust `naga` (no `glslc` prerequisite).
- **`vulkan_compute`**: `VulkanComputeEngine` manages instance, device, queue, command pool, timestamp `QueryPool`, `DEVICE_LOCAL` storage buffers, and staging transfers with strict RAII `Drop` cleanup.
- **`reference`**: FP64-accumulated CPU reference implementations and `compare_slices` accuracy evaluator (`abs_tol + rel_tol * |expected|`, checking `is_finite()`).
- **`hip_verify`**: RAII-guarded `hipMalloc` / `hipMemcpy` / `hipEvent*` 4 MiB bit-exact H2D+D2H verification against `amdhip64_*.dll`.

### `forge-benchmark`
- **`stats`**: Computes `min`, `median`, `mean`, `p95`, `max`, `MAD`, `IQR`, `stddev`, and `CV%` over warmed, repeated runs.
- **`memory`**: Captures process peak working set via Win32 `GetProcessMemoryInfo`.
- **`vulkan_suite`**, **`llama`**, **`triton`**: Orchestrates reproducible benchmarks and parses structured JSON from `llama-bench -o json` and `modules/ai/triton/bench_suite.py`.

### `forge-cli`
Thin CLI boundary (`clap` + `anyhow`) exposing `forge doctor [--json] [--verify] [--python]`, `forge bench {vulkan,llama,triton}`, and `forge profile show`.

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
2. **ADLX telemetry & tuning (`forge-tuning`)** expanding `forge-device::adlx` into clock, temperature, power, and fan sampling during benchmark runs.
3. **Subgroup / Cooperative Matrix Vulkan shaders** (`mul_mat` / quantized dot-product WGSL or GLSL/SPIR-V kernels) gated behind `GpuDeviceInfo` capability checks.
