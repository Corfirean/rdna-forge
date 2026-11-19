# Contributing to RDNA Forge

RDNA Forge is a disciplined engineering repository. Every change—code, documentation, tests, or benchmarks—must be verifiable, minimal, and free of speculative claims.

## Branch Strategy

- `main`: stable, reviewed, releasable code.
- `feat/<name>`: isolated feature work.
- `fix/<name>`: targeted bug fixes.
- `perf/<name>`: measured performance improvements with reproducible before/after benchmarks.
- `docs/<name>`: documentation-only changes.

Do not push directly to `main`.

## Commit Format

Use Conventional Commits:

`type(scope): concise description`

Examples:
- `feat(core): add Radeon device discovery`
- `feat(benchmark): add GEMM benchmark harness`
- `fix(device): handle unavailable telemetry`
- `perf(gfx1030): optimize quantized dot kernel`
- `docs(architecture): document runtime selection`

Stage files explicitly (`git add <file>...`). Never run `git add .` blindly.

## Coding Standards

- **Rust**: stable toolchain (`rustfmt` and `clippy` clean with `-D warnings`).
- **Error handling**: typed library errors (`ForgeError` via `thiserror`) in crates; contextual reporting (`anyhow`) only at CLI boundaries.
- **Hardware gating**: every specialized GPU kernel must verify device architecture and memory/buffer limits before dispatch. Never assume all Radeon GPUs share the same ISA capabilities.
- **No silent vendor blobs**: never vendor proprietary AMD SDK binaries or license-incompatible code (for example GPL-3.0 components into MIT-licensed crates).
- **Zero repository pollution**: keep scratch scripts, raw diagnostic dumps, downloaded GGUF models, virtual environments, and unreviewed benchmark outputs outside the repository.

## Testing & Review Requirements

Every pull request must pass hardware-independent checks before review:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python -m py_compile modules/ai/triton/bench_suite.py
```

Changes touching device discovery, shaders, runtimes, or benchmarks must additionally be tested on physical AMD hardware:

```powershell
cargo test --workspace -- --ignored --nocapture
forge doctor --verify
```

## Documentation & Benchmark Claims

- Never document unimplemented features or untested hardware combinations as supported.
- Every compatibility entry must cite exact driver/runtime versions and status (`Verified`, `Partially verified`, `Experimental`, `Unsupported`, `Not tested`).
- Every performance claim must follow [BENCHMARKS.md](BENCHMARKS.md): warmup iterations, multiple measured runs, median and dispersion metrics, accuracy verification, and reproducible commands.
