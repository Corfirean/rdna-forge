---
name: Hardware compatibility report
about: Submit verified test or benchmark results for an AMD Radeon GPU
title: "compat(gfxXXXX): hardware verification report"
labels: hardware-report
---

## Hardware & Software

- GPU Model:
- Architecture & `gfx` Target:
- VRAM:
- Resizable BAR Status:
- Windows Build:
- Adrenalin Driver Version:
- Runtime Versions (`Vulkan`, `ROCm/TheRock`, `triton-windows`, `llama.cpp`):

## Verification Commands Run

```powershell
forge doctor --verify
cargo test --workspace -- --ignored --nocapture
```

## Observed Results

Paste `forge doctor --json` and test/benchmark output without editing numbers.
