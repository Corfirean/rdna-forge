# Development Guide

## Prerequisites

### Core Rust & Vulkan Development (Required)
- **OS**: Windows 11 (x86_64)
- **GPU Driver**: AMD Software: Adrenalin Edition (`26.11.2` tested; provides `vulkan-1.dll`, `amdhip64_6.dll`, `amdadlx64.dll`)
- **Rust Toolchain**: Stable (`rustc 1.87+`; `rust-toolchain.toml` specifies `stable` with `rustfmt` and `clippy`)
- **Build Tools**: Visual Studio Build Tools (`MSVC` x64 linker)

Note: `glslc` / LunarG Vulkan SDK is **not** required to build or run RDNA Forge. WGSL shaders compile to SPIR-V in pure Rust via `naga`.

### Optional ROCm & Triton Environment (Python 3.11+)
To run PyTorch ROCm + `triton-windows` benchmarks on `gfx1030`:

```powershell
# Use a short path when Windows LongPathsEnabled is 0
python -m venv C:\venvs\rdna-forge
C:\venvs\rdna-forge\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\rdna-forge\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1030]==2.15.0"
C:\venvs\rdna-forge\Scripts\python.exe -m pip install "triton-windows<3.10"
```

### Optional `llama.cpp` Binaries
Download official Windows `x64` release zips from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) (`b12532` verified):
- `llama-b12532-bin-win-vulkan-x64.zip`
- `llama-b12532-bin-win-rocm-10.0-x64.zip`

---

## Build, Format, and Lint

```powershell
# Format check
cargo fmt --all -- --check

# Clippy with warnings denied
cargo clippy --workspace --all-targets -- -D warnings

# Release binary build
cargo build --release -p forge-cli
```

---

## Testing

### Hardware-Independent Tests (CI Safe)
Runs unit tests (`forge-core`, `forge-device`, `forge-runtime`, `forge-benchmark`, `forge-cli`), WGSL->SPIR-V compilation tests, and CLI/profile integration tests without requiring a physical GPU:

```powershell
cargo test --workspace
python -m py_compile modules/ai/triton/bench_suite.py
```

### Live GPU Correctness Tests
Runs device discovery, HIP 4 MiB H2D/D2H verification, Vulkan edge-case tests (`1x1x1`, non-aligned shapes, dimension/VRAM bounds checks), and a 50-dispatch stability check on physical AMD hardware:

```powershell
cargo test --workspace -- --ignored --nocapture
```

---

## Running Diagnostics & Benchmarks

```powershell
# Read-only hardware & runtime audit
.\target\release\forge.exe doctor
.\target\release\forge.exe doctor --json

# Live verification (HIP memory round-trip + Vulkan compute correctness)
.\target\release\forge.exe doctor --verify --python C:\venvs\rdna-forge\Scripts\python.exe

# Vulkan compute benchmark suite (writes JSON outside repo)
.\target\release\forge.exe bench vulkan --kernel all --warmup 10 --iters 50 --stability-runs 50 --output ..\vulkan_bench.json

# PyTorch ROCm + Triton benchmark suite
.\target\release\forge.exe bench triton --python C:\venvs\rdna-forge\Scripts\python.exe --warmup 10 --iters 50 --output ..\triton_bench.json

# llama.cpp Vulkan vs ROCm/HIP comparison
$env:PATH = "C:\venvs\rdna-forge\Lib\site-packages\_rocm_sdk_libraries\bin;C:\venvs\rdna-forge\Lib\site-packages\_rocm_sdk_core\bin;" + $env:PATH
.\target\release\forge.exe bench llama `
  --model ..\models\qwen2.5-1.5b-instruct-q4_k_m.gguf `
  --vulkan-bin ..\llama-vk\llama-bench.exe `
  --hip-bin ..\llama-hip\llama-bench.exe `
  --prompt-tokens 512 --gen-tokens 128 --reps 5 --ngl 99 `
  --vulkan-device Vulkan0 --hip-device ROCm0 `
  --output ..\llama_bench.json
```

---

## Windows Engineering Pitfalls

1. **Windows Long Path Limit (`LongPathsEnabled = 0`)**: ROCm/TheRock and PyTorch wheels contain deeply nested paths. Create virtual environments under short paths (e.g., `C:\venvs\...`) or enable `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`.
2. **WMI `AdapterRAM` 4 GiB Truncation**: `Win32_VideoController.AdapterRAM` is a 32-bit unsigned integer (~4 GiB cap on 16 GiB cards). RDNA Forge reads 64-bit VRAM from `hipDeviceTotalMem` / `hipDeviceProp_tR0600` and Registry `HardwareInformation.qwMemorySize`.
3. **`llama.cpp` ROCm Windows Missing `hipblas.dll`**: `llama-b12532-bin-win-rocm-10.0-x64.zip` omits `hipblas.dll` and `rocblas.dll`. Without ROCm `bin` directories on `PATH`, `llama-bench --list-devices` reports `(none)`.
4. **Windows WDDM `hipEventElapsedTime` Skew**: Synchronize the device before recording the start event (`torch.cuda.synchronize()`) and synchronize the end event (`end_ev.synchronize()`) before calling `elapsed_time()`. Unsynchronized calls on Windows WDDM can produce negative durations.
5. **Multi-GPU Ryzen + Radeon Systems**: Ryzen 7000 CPUs expose an integrated RDNA2 GPU (`gfx1036`). Always pin `llama-bench` with explicit device selectors (`-dev Vulkan0`, `-dev ROCm0`) to ensure benchmarks target the discrete GPU.
6. **Resizable BAR**: On systems with ReBAR disabled, the host-visible `DEVICE_LOCAL` Vulkan heap is capped at `256 MiB`, and `maxBufferSize` is `2 GiB`. Keep temporary outputs and downloaded models outside the repository.
