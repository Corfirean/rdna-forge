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

- **DLL Loading** (`forge_core::loader`): no native library is loaded by bare name, so the Windows default search order (application directory, current directory, `PATH`) is never used.
  - *System and AMD-driver libraries* (`ntdll.dll`, `pdh.dll`, `vulkan-1.dll`, `amdadlx64.dll`, `amfrt64.dll`) are resolved only inside the directory returned by `GetSystemDirectoryW` and loaded by absolute path with `LOAD_LIBRARY_SEARCH_SYSTEM32`.
  - *Vendor SDK libraries* (the HIP runtime, `amdhip64_*.dll`) are accepted only if their canonical path (symlinks, junctions and `..` resolved) lies under System32, `%ProgramFiles%` or `%ProgramFiles(x86)%`, obtained from `SHGetKnownFolderPath` rather than environment variables. `HIP_PATH` / `ROCM_PATH` only add candidates; a candidate outside those roots is rejected and reported, never silently replaced by a bare-name search. A HIP SDK installed elsewhere (for example `D:\ROCm`) is therefore not detected; the driver's System32 copy is still used.
  - Dependencies of vendor libraries resolve from their own directory and System32 only. After loading, the module's real path is read back and compared with the validated path.
  - Failures are typed: not installed, refused (untrusted location), or rejected by the OS loader. The loaded path is reported by `forge doctor --json` for ADLX, HIP and Vulkan.
  - `llama-bench` and Python are *executed*, not loaded: `llama-bench` paths must resolve to an existing `llama-bench*.exe`; the Python interpreter is whatever the user passes with `--python` (or `python` from the normal executable search) and is trusted as the user's own choice. `reg.exe` is launched from System32 by absolute path.
- **GPU Tuning Safety** (`forge tune`): read-only by default. `power-limit` and `reset` mutate only with `--apply`; requested values are validated against the hardware range and step reported by ADLX. After a mutation the value is read back: a contradicting read-back is an error, an unreadable one is reported as `NOT CONFIRMED`. There is no automatic overclocking, no voltage or fan changes, and nothing is tuned at startup. **No automatic rollback is implemented**; the previous value is printed so it can be re-applied explicitly. The `--apply` paths were not exercised against live hardware during Milestone 8 qualification (only read-only inspection and dry runs were).
- **Configuration Patching** (`forge llm apply-lmstudio`, `apply-env`): dry-run by default and prints the exact files and `HKCU\Environment` keys it would change. An existing LM Studio config that is not a valid JSON object is refused rather than replaced. Backups (`.json.bak`) are never overwritten, so repeated applies keep the original; writes use a temp file and rename, and the result is re-read and verified. Unrelated user fields are preserved. Model paths containing `..` cannot move the written file outside the LM Studio config directory. Nothing is sent over the network.
- **Desktop trust boundary**: the Tauri window loads only bundled assets under a strict CSP (`default-src 'self'`) and has only the `core:default` capability (no filesystem or shell plugins). Because the webview is still a separate trust domain, the backend re-validates paths it receives (benchmark executables, model paths) and does not rely on the frontend for safety checks; the frontend does choose the `dry_run` flag, so treat the desktop app as trusted-local software.
- **Supply chain**: CI uses actions pinned to commit SHAs, read-only `GITHUB_TOKEN` permissions by default (write access only in the release publish job), `--locked` builds against the committed `Cargo.lock`, and `cargo-deny` (RustSec advisories, a license allow-list, crates.io-only sources; see `deny.toml`).
- **Bounds & Allocation Checks**: GPU compute wrappers validate non-zero dimensions, checked arithmetic, `maxBufferSize`, and VRAM capacity before allocating or dispatching device buffers.
- **Secrets & Privacy**: Diagnostic reports (`forge doctor --json`) inspect local hardware and Python paths; review output before posting publicly if directory paths contain local usernames.
