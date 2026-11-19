# Security Policy

## Supported Versions

Only the current development tip (`main` and active release branches) is supported with security updates.

## Reporting a Vulnerability

Please **do not** open public GitHub issues for suspected security vulnerabilities, driver-crash exploits, unsafe FFI memory corruption bugs, or credential exposure.

Report vulnerabilities privately through GitHub Security Advisories for this repository, including:

1. Affected crate, CLI subcommand, or runtime integration (`forge-device`, `forge-runtime`, etc.).
2. Windows build, AMD Adrenalin driver version, and GPU `gfx` target.
3. Minimal reproduction steps or proof-of-concept input.
4. Observed impact (process crash, out-of-bounds memory access, DLL hijacking, or GPU reset/TDR).

## Security Design Expectations

- **DLL Loading**: RDNA Forge dynamically loads system/runtime DLLs (`vulkan-1.dll`, `amdhip64_*.dll`, `amdadlx64.dll`) via standard Windows/runtime search paths and explicitly checks `HIP_PATH` / `ROCM_PATH` before falling back. Do not place untrusted DLLs in working directories alongside `forge.exe`.
- **Bounds & Allocation Checks**: GPU compute wrappers validate non-zero dimensions, checked arithmetic, `maxBufferSize`, and VRAM capacity before allocating or dispatching device buffers.
- **Secrets & Privacy**: Diagnostic reports (`forge doctor --json`) inspect local hardware and Python paths; review output before posting publicly if directory paths contain local usernames.
