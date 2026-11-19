#!/usr/bin/env python3
"""Reproducible PyTorch ROCm/HIP and Triton benchmark & correctness suite for RDNA Forge."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Any, Callable


def peak_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32.dll")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            handle = kernel32.GetCurrentProcess()
            if kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except Exception:
            return None
    return None


def sorted_percentile(sorted_vals: list[float], pct: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_latency_stats(warmup_iters: int, samples_ms: list[float]) -> dict[str, Any]:
    s = sorted(float(x) for x in samples_ms)
    n = len(s)
    med = sorted_percentile(s, 50.0)
    q25 = sorted_percentile(s, 25.0)
    q75 = sorted_percentile(s, 75.0)
    mean_v = float(sum(s) / n)
    stddev_v = float(statistics.stdev(s)) if n > 1 else 0.0
    abs_devs = sorted(abs(x - med) for x in s)
    mad_v = sorted_percentile(abs_devs, 50.0)
    return {
        "warmup_iters": warmup_iters,
        "measured_iters": n,
        "min_ms": s[0],
        "median_ms": med,
        "mean_ms": mean_v,
        "p95_ms": sorted_percentile(s, 95.0),
        "max_ms": s[-1],
        "mad_ms": mad_v,
        "iqr_ms": max(0.0, q75 - q25),
        "stddev_ms": stddev_v,
        "cv_pct": (stddev_v / mean_v * 100.0) if mean_v > 0.0 else 0.0,
    }


def check_accuracy(actual_t: Any, expected_t: Any, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    import torch

    a = actual_t.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    e = expected_t.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite_a = bool(torch.isfinite(a).all().item())
    finite_e = bool(torch.isfinite(e).all().item())
    all_finite = finite_a and finite_e
    if not all_finite:
        return {
            "max_abs_err": float("nan"),
            "max_rel_err": float("nan"),
            "rmse": float("nan"),
            "all_finite": False,
            "passed": False,
        }
    diff = (a - e).abs()
    max_abs = float(diff.max().item())
    rel = diff / e.abs().clamp(min=1e-6)
    max_rel = float(rel.max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    passed = bool((diff <= (abs_tol + rel_tol * e.abs())).all().item())
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "rmse": rmse,
        "all_finite": True,
        "passed": passed,
    }


def bench_gpu_fn(
    fn: Callable[[], Any],
    warmup_iters: int,
    measured_iters: int,
) -> tuple[Any, dict[str, Any], int, int, int]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    out = None
    for _ in range(warmup_iters):
        out = fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    negative_events = 0
    for _ in range(measured_iters):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_ev.record()
        out = fn()
        end_ev.record()
        end_ev.synchronize()
        t1 = time.perf_counter()
        ev_ms = float(start_ev.elapsed_time(end_ev))
        if ev_ms >= 0.0:
            samples_ms.append(ev_ms)
        else:
            negative_events += 1
            samples_ms.append((t1 - t0) * 1000.0)

    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_res = int(torch.cuda.max_memory_reserved())
    stats = compute_latency_stats(warmup_iters, samples_ms)
    return out, stats, peak_alloc, peak_res, negative_events


def collect_provenance() -> dict[str, Any]:
    prov: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": None,
        "torch_hip_version": None,
        "triton_version": None,
        "triton_windows_version": None,
        "device_name": None,
        "gcn_arch_name": None,
        "total_vram_bytes": None,
        "os": platform.platform(),
    }
    try:
        import torch

        prov["torch_version"] = getattr(torch, "__version__", None)
        prov["torch_hip_version"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            prov["device_name"] = props.name
            prov["gcn_arch_name"] = getattr(props, "gcnArchName", None)
            prov["total_vram_bytes"] = int(props.total_memory)
    except Exception:
        pass
    try:
        import triton

        prov["triton_version"] = getattr(triton, "__version__", None)
    except Exception:
        pass
    try:
        prov["triton_windows_version"] = importlib.metadata.version("triton-windows")
    except Exception:
        pass
    return prov


def run_all_benchmarks(
    warmup_iters: int,
    measured_iters: int,
    stability_iters: int,
    smoke: bool = False,
) -> dict[str, Any]:
    provenance = collect_provenance()
    benchmarks: list[dict[str, Any]] = []
    stability_report: dict[str, Any] | None = None

    import torch

    if not torch.cuda.is_available():
        return {
            "provenance": provenance,
            "benchmarks": [
                {
                    "name": "device_check",
                    "backend": "pytorch_hip",
                    "shape": "n/a",
                    "dtype": "n/a",
                    "status": "unsupported",
                    "latency": None,
                    "throughput_unit": None,
                    "throughput_median": None,
                    "accuracy": None,
                    "peak_vram_allocated_bytes": None,
                    "peak_vram_reserved_bytes": None,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "error": "torch.cuda.is_available() returned False",
                    "notes": None,
                }
            ],
            "stability": None,
        }

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    def record_outcome(
        name: str,
        backend: str,
        shape: str,
        dtype_str: str,
        fn: Callable[[], tuple[Any, Any, float, float, float, str | None]],
    ) -> None:
        # fn() returns (gpu_fn, ref_tensor, flops_or_bytes, throughput_unit, abs_tol, rel_tol, note)
        if smoke:
            print(f"[smoke] running {name} ({shape})...", file=sys.stderr, flush=True)
        try:
            gpu_fn, ref_t, work_units, unit, abs_tol, rel_tol, note = fn()
            out_t, lat, peak_alloc, peak_res, neg_evs = bench_gpu_fn(
                gpu_fn, warmup_iters, measured_iters
            )
            if neg_evs > 0:
                extra = f"{neg_evs} negative hipEventElapsedTime readings replaced with perf_counter"
                note = f"{note}; {extra}" if note else extra
            acc = check_accuracy(out_t, ref_t, abs_tol, rel_tol)
            med_s = lat["median_ms"] / 1000.0
            tp = (work_units / med_s) if med_s > 0.0 else 0.0
            status = "verified" if acc["passed"] else "failed"
            benchmarks.append(
                {
                    "name": name,
                    "backend": backend,
                    "shape": shape,
                    "dtype": dtype_str,
                    "status": status,
                    "latency": lat,
                    "throughput_unit": unit,
                    "throughput_median": tp,
                    "accuracy": acc,
                    "peak_vram_allocated_bytes": peak_alloc,
                    "peak_vram_reserved_bytes": peak_res,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "error": None if acc["passed"] else "numerical accuracy check failed",
                    "notes": note,
                }
            )
        except Exception as exc:
            benchmarks.append(
                {
                    "name": name,
                    "backend": backend,
                    "shape": shape,
                    "dtype": dtype_str,
                    "status": "unsupported",
                    "latency": None,
                    "throughput_unit": None,
                    "throughput_median": None,
                    "accuracy": None,
                    "peak_vram_allocated_bytes": None,
                    "peak_vram_reserved_bytes": None,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "notes": None,
                }
            )
        finally:
            torch.cuda.empty_cache()
            time.sleep(0.02)

    if not smoke:
        # 1. PyTorch HIP GEMM (FP32 & FP16)
        for dt_name, dt, abs_tol, rel_tol in [
            ("f32", torch.float32, 1e-2, 1e-3),
            ("f16", torch.float16, 1e-1, 2e-2),
        ]:
            for n in (2048, 4096):
                def make_pt_gemm(n: int = n, dt: Any = dt, abs_tol: float = abs_tol, rel_tol: float = rel_tol):
                    a = torch.randn((n, n), device=device, dtype=dt)
                    b = torch.randn((n, n), device=device, dtype=dt)
                    # Reference in fp64 on CPU for first 64 rows to avoid massive CPU runtime on 4096^3
                    rows_ref = min(n, 64)
                    ref = (
                        a[:rows_ref].to("cpu", torch.float64)
                        @ b.to("cpu", torch.float64)
                    )

                    def run():
                        c = torch.matmul(a, b)
                        return c[:rows_ref]

                    return (
                        run,
                        ref,
                        2.0 * (n**3) / 1e12,
                        "TFLOP/s",
                        abs_tol,
                        rel_tol,
                        f"validated top {rows_ref} rows against CPU FP64 reference",
                    )

                record_outcome(
                    f"pytorch_gemm_{dt_name}_{n}",
                    "pytorch_hip_rocblas",
                    f"{n}x{n}x{n}",
                    dt_name,
                    make_pt_gemm,
                )

        # 2. Quantized INT8 GEMM check (torch._int_mm)
        def make_int8_gemm():
            n = 2048
            a = torch.randint(-127, 127, (n, n), device=device, dtype=torch.int8)
            b = torch.randint(-127, 127, (n, n), device=device, dtype=torch.int8)
            ref = a[:64].to("cpu", torch.int64) @ b.to("cpu", torch.int64)

            def run():
                c = torch._int_mm(a, b)
                return c[:64]

            return (
                run,
                ref.to(torch.float64),
                2.0 * (n**3) / 1e12,
                "TOPS",
                0.0,
                0.0,
                "torch._int_mm exact integer matmul",
            )

        record_outcome(
            "pytorch_int8_gemm_2048",
            "pytorch_hip_int_mm",
            "2048x2048x2048",
            "i8->i32",
            make_int8_gemm,
        )

        # 3. PyTorch RMSNorm & RoPE (FP16)
        def make_pt_rmsnorm():
            rows, cols = 4096, 4096
            eps = 1e-5
            x = torch.randn((rows, cols), device=device, dtype=torch.float16)
            w = torch.randn((cols,), device=device, dtype=torch.float16)
            x64 = x.to("cpu", torch.float64)
            w64 = w.to("cpu", torch.float64)
            ref = x64 * torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps) * w64

            def run():
                xf = x.float()
                rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
                return (xf * rms * w.float()).to(torch.float16)

            bytes_moved = (2 * rows * cols + cols) * 2 / 1e9
            return run, ref, bytes_moved, "GB/s", 5e-2, 2e-2, "CPU FP64 reference"

        record_outcome(
            "pytorch_rmsnorm_f16_4096x4096",
            "pytorch_hip",
            "4096x4096",
            "f16",
            make_pt_rmsnorm,
        )

        def make_pt_rope():
            seq, heads, dim = 2048, 32, 128
            theta = 10000.0
            x = torch.randn((1, seq, heads, dim), device=device, dtype=torch.float16)
            half = dim // 2
            pos = torch.arange(seq, device=device, dtype=torch.float32)[:, None, None]
            idx = torch.arange(half, device=device, dtype=torch.float32)[None, None, :]
            freqs = pos * (theta ** (-2.0 * idx / dim))
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)

            # Reference on CPU FP64
            pos_c = torch.arange(seq, dtype=torch.float64)[:, None, None]
            idx_c = torch.arange(half, dtype=torch.float64)[None, None, :]
            freqs_c = pos_c * (theta ** (-2.0 * idx_c / dim))
            xc = x.to("cpu", torch.float64).reshape(seq, heads, half, 2)
            x0c, x1c = xc[..., 0], xc[..., 1]
            cc, sc = torch.cos(freqs_c), torch.sin(freqs_c)
            ref = torch.stack((x0c * cc - x1c * sc, x0c * sc + x1c * cc), dim=-1).reshape(1, seq, heads, dim)

            def run():
                xf = x.float().reshape(seq, heads, half, 2)
                x0, x1 = xf[..., 0], xf[..., 1]
                out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
                return out.reshape(1, seq, heads, dim).to(torch.float16)

            bytes_moved = (2 * seq * heads * dim) * 2 / 1e9
            return run, ref, bytes_moved, "GB/s", 5e-2, 2e-2, "CPU FP64 reference"

        record_outcome(
            "pytorch_rope_f16_2048x32x128",
            "pytorch_hip",
            "1x2048x32x128",
            "f16",
            make_pt_rope,
        )

        # 4. PyTorch Scaled Dot-Product Attention backends individually
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            sdp_modes = [
                ("math", SDPBackend.MATH),
                ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
                ("flash_attention", SDPBackend.FLASH_ATTENTION),
            ]
        except Exception:
            sdp_modes = []

        for mode_name, backend_enum in sdp_modes:
            def make_sdpa(backend_enum: Any = backend_enum):
                b, h, s, d = 1, 16, 1024, 128
                q = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
                k = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
                v = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
                # CPU reference in fp32
                qf = q.to("cpu", torch.float32)
                kf = k.to("cpu", torch.float32)
                vf = v.to("cpu", torch.float32)
                scores = (qf @ kf.transpose(-2, -1)) / math.sqrt(d)
                probs = torch.softmax(scores, dim=-1)
                ref = probs @ vf

                def run():
                    with sdpa_kernel(backend_enum):
                        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

                flops = 4.0 * b * h * s * s * d / 1e12
                return run, ref, flops, "TFLOP/s", 5e-2, 2e-2, "4*B*H*S^2*D approx attention FLOPs"

            record_outcome(
                f"pytorch_sdpa_{mode_name}_1024",
                f"pytorch_sdpa_{mode_name}",
                "1x16x1024x128",
                "f16",
                make_sdpa,
            )

    # 5. Triton Kernels (if triton-windows is installed)
    triton_gemm_fn_for_stability: Callable[[], Any] | None = None
    triton_ref_for_stability: Any = None
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def triton_matmul_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                k_mask = offs_k + k * BLOCK_K < K
                a = tl.load(
                    a_ptrs,
                    mask=(offs_am[:, None] < M) & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None] & (offs_bn[None, :] < N),
                    other=0.0,
                )
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

        @triton.jit
        def triton_rmsnorm_kernel(
            X_ptr,
            W_ptr,
            Y_ptr,
            stride_x,
            stride_y,
            N,
            eps,
            BLOCK_SIZE: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            mean_sq = tl.sum(x * x, axis=0) / N
            rstd = tl.rsqrt(mean_sq + eps)
            y = x * rstd * w
            tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float16), mask=mask)

        @triton.jit
        def triton_rope_kernel(
            X_ptr,
            Y_ptr,
            seq_len,
            n_heads,
            head_dim,
            theta_base,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            half_dim = head_dim // 2
            total_pairs = seq_len * n_heads * half_dim
            idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = idx < total_pairs

            head_pairs = n_heads * half_dim
            pos = idx // head_pairs
            rem = idx % head_pairs
            head = rem // half_dim
            i = rem % half_dim

            exponent = -(2.0 * i.to(tl.float32)) / head_dim
            freq = tl.exp(exponent * tl.log(theta_base))
            angle = pos.to(tl.float32) * freq
            c = tl.cos(angle)
            s = tl.sin(angle)

            base = (pos * n_heads + head) * head_dim + 2 * i
            x0 = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(X_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
            tl.store(Y_ptr + base, (x0 * c - x1 * s).to(tl.float16), mask=mask)
            tl.store(Y_ptr + base + 1, (x0 * s + x1 * c).to(tl.float16), mask=mask)

        @triton.jit
        def triton_attention_kernel(
            Q_ptr,
            K_ptr,
            V_ptr,
            O_ptr,
            stride_qb,
            stride_qh,
            stride_qm,
            stride_qd,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_vd,
            stride_ob,
            stride_oh,
            stride_om,
            stride_od,
            H,
            S,
            scale,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            HEAD_DIM: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_bh = tl.program_id(1)
            b = pid_bh // H
            h = pid_bh % H

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, HEAD_DIM)
            q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

            m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            for start_n in range(0, S, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_ptrs = K_ptr + b * stride_kb + h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
                v_ptrs = V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
                k = tl.load(k_ptrs, mask=offs_n[None, :] < S, other=0.0)
                v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)

                qk = tl.dot(q, k) * scale
                qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp(qk - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
                m_i = m_ij

            acc = acc / l_i[:, None]
            o_ptrs = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
            tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < S)

        if not smoke:
            # 5a. Triton GEMM FP16 (aligned 2048 + non-aligned 1027x1029x513)
            for label, (m, k, n) in [
                ("triton_gemm_f16_2048", (2048, 2048, 2048)),
                ("triton_gemm_f16_unaligned", (1027, 513, 1029)),
            ]:
                def make_triton_gemm(m: int = m, k: int = k, n: int = n):
                    a = torch.randn((m, k), device=device, dtype=torch.float16)
                    b = torch.randn((k, n), device=device, dtype=torch.float16)
                    c = torch.empty((m, n), device=device, dtype=torch.float16)
                    rows_ref = min(m, 64)
                    ref = a[:rows_ref].to("cpu", torch.float64) @ b.to("cpu", torch.float64)
                    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 64),)

                    def run():
                        triton_matmul_kernel[grid](
                            a, b, c, m, n, k,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        )
                        return c[:rows_ref]

                    return (
                        run,
                        ref,
                        2.0 * m * n * k / 1e12,
                        "TFLOP/s",
                        1e-1,
                        2e-2,
                        f"validated top {rows_ref} rows against CPU FP64 reference",
                    )

                record_outcome(
                    label,
                    "triton_windows",
                    f"{m}x{k}x{n}",
                    "f16",
                    make_triton_gemm,
                )

            # Save a lightweight 512x512 Triton GEMM closure for the stability test
            stab_a = torch.randn((512, 512), device=device, dtype=torch.float16)
            stab_b = torch.randn((512, 512), device=device, dtype=torch.float16)
            stab_c = torch.empty((512, 512), device=device, dtype=torch.float16)
            triton_ref_for_stability = stab_a.to("cpu", torch.float64) @ stab_b.to("cpu", torch.float64)

            def run_stab_gemm():
                grid = (triton.cdiv(512, 64) * triton.cdiv(512, 64),)
                triton_matmul_kernel[grid](
                    stab_a, stab_b, stab_c, 512, 512, 512,
                    stab_a.stride(0), stab_a.stride(1),
                    stab_b.stride(0), stab_b.stride(1),
                    stab_c.stride(0), stab_c.stride(1),
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                )
                return stab_c

            triton_gemm_fn_for_stability = run_stab_gemm

            # 5b. Triton RMSNorm FP16 (4096x4096 + non-aligned 127x4099)
            for label, (rows, cols) in [
                ("triton_rmsnorm_f16_4096x4096", (4096, 4096)),
                ("triton_rmsnorm_f16_unaligned", (127, 4099)),
            ]:
                def make_triton_rmsnorm(rows: int = rows, cols: int = cols):
                    eps = 1e-5
                    x = torch.randn((rows, cols), device=device, dtype=torch.float16)
                    w = torch.randn((cols,), device=device, dtype=torch.float16)
                    y = torch.empty_like(x)
                    x64 = x.to("cpu", torch.float64)
                    w64 = w.to("cpu", torch.float64)
                    ref = x64 * torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps) * w64
                    block = triton.next_power_of_2(cols)

                    def run():
                        triton_rmsnorm_kernel[(rows,)](
                            x, w, y, x.stride(0), y.stride(0), cols, eps, BLOCK_SIZE=block
                        )
                        return y

                    bytes_moved = (2 * rows * cols + cols) * 2 / 1e9
                    return run, ref, bytes_moved, "GB/s", 5e-2, 2e-2, "CPU FP64 reference"

                record_outcome(
                    label,
                    "triton_windows",
                    f"{rows}x{cols}",
                    "f16",
                    make_triton_rmsnorm,
                )

            # 5c. Triton RoPE FP16
            def make_triton_rope():
                seq, heads, dim = 2048, 32, 128
                theta = 10000.0
                x = torch.randn((1, seq, heads, dim), device=device, dtype=torch.float16)
                y = torch.empty_like(x)
                half = dim // 2
                pos_c = torch.arange(seq, dtype=torch.float64)[:, None, None]
                idx_c = torch.arange(half, dtype=torch.float64)[None, None, :]
                freqs_c = pos_c * (theta ** (-2.0 * idx_c / dim))
                xc = x.to("cpu", torch.float64).reshape(seq, heads, half, 2)
                x0c, x1c = xc[..., 0], xc[..., 1]
                cc, sc = torch.cos(freqs_c), torch.sin(freqs_c)
                ref = torch.stack((x0c * cc - x1c * sc, x0c * sc + x1c * cc), dim=-1).reshape(1, seq, heads, dim)
                total_pairs = seq * heads * half
                grid = (triton.cdiv(total_pairs, 256),)

                def run():
                    triton_rope_kernel[grid](x, y, seq, heads, dim, theta, BLOCK_SIZE=256)
                    return y

                bytes_moved = (2 * seq * heads * dim) * 2 / 1e9
                return run, ref, bytes_moved, "GB/s", 5e-2, 2e-2, "CPU FP64 reference"

            record_outcome(
                "triton_rope_f16_2048x32x128",
                "triton_windows",
                "1x2048x32x128",
                "f16",
                make_triton_rope,
            )

            # 5d. Triton Tiled Attention FP16
            def make_triton_attn():
                b, h, s, d = 1, 16, 1024, 64
                q = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
                k = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
                v = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
                o = torch.empty_like(q)
                qf = q.to("cpu", torch.float32)
                kf = k.to("cpu", torch.float32)
                vf = v.to("cpu", torch.float32)
                scores = (qf @ kf.transpose(-2, -1)) / math.sqrt(d)
                ref = torch.softmax(scores, dim=-1) @ vf
                grid = (triton.cdiv(s, 32), b * h)
                scale = 1.0 / math.sqrt(d)

                def run():
                    triton_attention_kernel[grid](
                        q, k, v, o,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                        h, s, scale,
                        BLOCK_M=32, BLOCK_N=32, HEAD_DIM=d,
                    )
                    return o

                flops = 4.0 * b * h * s * s * d / 1e12
                return run, ref, flops, "TFLOP/s", 5e-2, 2e-2, "online-softmax tiled Triton kernel"

            record_outcome(
                "triton_attention_f16_1024",
                "triton_windows",
                "1x16x1024x64",
                "f16",
                make_triton_attn,
            )

        # 5e. Milestone 2: L2/Infinity-Cache Swizzled Triton GEMM FP16 (safe gfx1030 Wave32 tile, no autotune bursts)
        @triton.jit
        def triton_matmul_swizzled_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                k_mask = offs_k + k * BLOCK_K < K
                a = tl.load(
                    a_ptrs,
                    mask=(offs_am[:, None] < M) & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None] & (offs_bn[None, :] < N),
                    other=0.0,
                )
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

        swiz_n = 256 if smoke else 2048

        def make_triton_gemm_swizzled(n: int = swiz_n):
            a = torch.randn((n, n), device=device, dtype=torch.float16)
            b = torch.randn((n, n), device=device, dtype=torch.float16)
            c = torch.empty((n, n), device=device, dtype=torch.float16)
            rows_ref = min(n, 64)
            ref = a[:rows_ref].to("cpu", torch.float64) @ b.to("cpu", torch.float64)
            grid = (triton.cdiv(n, 64) * triton.cdiv(n, 64),)

            def run():
                triton_matmul_swizzled_kernel[grid](
                    a, b, c, n, n, n,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                    GROUP_SIZE_M=8, num_stages=1, num_warps=4,
                )
                return c[:rows_ref]

            return (
                run,
                ref,
                2.0 * (n**3) / 1e12,
                "TFLOP/s",
                1e-1,
                2e-2,
                f"L2-swizzled (GROUP_SIZE_M=8) 64x64x32 Wave32 GEMM; top {rows_ref} rows FP64 ref",
            )

        record_outcome(
            f"triton_gemm_f16_swizzled_{swiz_n}",
            "triton_windows",
            f"{swiz_n}x{swiz_n}x{swiz_n}",
            "f16",
            make_triton_gemm_swizzled,
        )

        # 5f. Milestone 2: Weight-Only W8A16 Dequant-Fused GEMM (fills hipBLASLt INT8 gap on gfx1030)
        @triton.jit
        def triton_w8a16_matmul_kernel(
            a_ptr,
            wq_ptr,
            scales_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_wk,
            stride_wn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            w_ptrs = wq_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
            scales = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=0.0).to(tl.float16)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                k_mask = offs_k + k * BLOCK_K < K
                a = tl.load(
                    a_ptrs,
                    mask=(offs_am[:, None] < M) & k_mask[None, :],
                    other=0.0,
                )
                w_q = tl.load(
                    w_ptrs,
                    mask=k_mask[:, None] & (offs_bn[None, :] < N),
                    other=0,
                )
                w = w_q.to(tl.float16) * scales[None, :]
                acc = tl.dot(a, w, acc)
                a_ptrs += BLOCK_K * stride_ak
                w_ptrs += BLOCK_K * stride_wk

            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

        w8_cases = (
            [("triton_w8a16_gemm_256", (256, 256, 256)), ("triton_w8a16_gemm_unaligned", (259, 131, 261))]
            if smoke
            else [("triton_w8a16_gemm_2048", (2048, 2048, 2048)), ("triton_w8a16_gemm_unaligned", (1027, 513, 1029))]
        )
        for label, (m, k, n) in w8_cases:
            def make_w8a16(m: int = m, k: int = k, n: int = n):
                a = torch.randn((m, k), device=device, dtype=torch.float16)
                w_q = torch.randint(-64, 64, (k, n), device=device, dtype=torch.int8)
                scales = (torch.rand((n,), device=device, dtype=torch.float16) * 0.03) + 0.01
                c = torch.empty((m, n), device=device, dtype=torch.float16)
                rows_ref = min(m, 64)
                w_deq64 = w_q.to("cpu", torch.float64) * scales.to("cpu", torch.float64)[None, :]
                ref = a[:rows_ref].to("cpu", torch.float64) @ w_deq64
                grid = (triton.cdiv(m, 64) * triton.cdiv(n, 64),)

                def run():
                    triton_w8a16_matmul_kernel[grid](
                        a, w_q, scales, c, m, n, k,
                        a.stride(0), a.stride(1),
                        w_q.stride(0), w_q.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        GROUP_SIZE_M=8, num_stages=1, num_warps=4,
                    )
                    return c[:rows_ref]

                return (
                    run,
                    ref,
                    2.0 * m * n * k / 1e12,
                    "TFLOP/s",
                    1e-1,
                    2e-2,
                    f"W8A16 per-channel dequant-fused GEMM (2x weight compression); top {rows_ref} rows FP64 ref",
                )

            record_outcome(
                label,
                "triton_windows",
                f"{m}x{k}x{n}",
                "f16xint8->f16",
                make_w8a16,
            )

        # 5g. Milestone 2: Weight-Only W4A16 Group-Quantized GEMM (group_size=128)
        @triton.jit
        def triton_w4a16_matmul_kernel(
            a_ptr,
            w_packed_ptr,
            scales_ptr,
            zeros_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_wk,
            stride_wn,
            stride_sg,
            stride_sn,
            stride_zg,
            stride_zn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_SIZE: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
        ):
            BLOCK_KP: tl.constexpr = BLOCK_K // 2
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_kp = tl.arange(0, BLOCK_KP)
            m_mask = offs_am[:, None] < M
            n_mask = offs_bn[None, :] < N
            k_packed_total = K // 2

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            num_tiles = tl.cdiv(K, BLOCK_K)
            for t in range(0, num_tiles):
                k_start = t * BLOCK_K
                g_idx = k_start // GROUP_SIZE
                scale = tl.load(
                    scales_ptr + g_idx * stride_sg + offs_bn * stride_sn,
                    mask=offs_bn < N,
                    other=0.0,
                ).to(tl.float16)
                zero = tl.load(
                    zeros_ptr + g_idx * stride_zg + offs_bn * stride_zn,
                    mask=offs_bn < N,
                    other=0.0,
                ).to(tl.float16)

                cur_kp = t * BLOCK_KP + offs_kp
                kp_mask = cur_kp < k_packed_total
                packed = tl.load(
                    w_packed_ptr + (cur_kp[:, None] * stride_wk + offs_bn[None, :] * stride_wn),
                    mask=kp_mask[:, None] & n_mask,
                    other=0,
                )
                w_even = ((packed & 0x0F).to(tl.float16) - zero[None, :]) * scale[None, :]
                w_odd = (((packed >> 4) & 0x0F).to(tl.float16) - zero[None, :]) * scale[None, :]

                k_even = 2 * cur_kp
                k_odd = k_even + 1
                a_even = tl.load(
                    a_ptr + (offs_am[:, None] * stride_am + k_even[None, :] * stride_ak),
                    mask=m_mask & (k_even[None, :] < K),
                    other=0.0,
                )
                a_odd = tl.load(
                    a_ptr + (offs_am[:, None] * stride_am + k_odd[None, :] * stride_ak),
                    mask=m_mask & (k_odd[None, :] < K),
                    other=0.0,
                )

                acc = tl.dot(a_even, w_even, acc)
                acc = tl.dot(a_odd, w_odd, acc)

            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

        w4_n = 256 if smoke else 2048

        def make_w4a16(n: int = w4_n):
            group_size = 128
            num_groups = n // group_size
            a = torch.randn((n, n), device=device, dtype=torch.float16)
            q_even = torch.randint(0, 16, (n // 2, n), device=device, dtype=torch.uint8)
            q_odd = torch.randint(0, 16, (n // 2, n), device=device, dtype=torch.uint8)
            w_packed = q_even | (q_odd << 4)
            scales = (torch.rand((num_groups, n), device=device, dtype=torch.float16) * 0.08) + 0.02
            zeros = torch.full((num_groups, n), 7.5, device=device, dtype=torch.float16)
            c = torch.empty((n, n), device=device, dtype=torch.float16)

            rows_ref = min(n, 64)
            w_u8 = torch.empty((n, n), dtype=torch.float64)
            w_u8[0::2, :] = q_even.to("cpu", torch.float64)
            w_u8[1::2, :] = q_odd.to("cpu", torch.float64)
            scales_exp = scales.to("cpu", torch.float64).repeat_interleave(group_size, dim=0)
            zeros_exp = zeros.to("cpu", torch.float64).repeat_interleave(group_size, dim=0)
            w_deq64 = (w_u8 - zeros_exp) * scales_exp
            ref = a[:rows_ref].to("cpu", torch.float64) @ w_deq64

            grid = (triton.cdiv(n, 64) * triton.cdiv(n, 64),)

            def run():
                triton_w4a16_matmul_kernel[grid](
                    a, w_packed, scales, zeros, c, n, n, n,
                    a.stride(0), a.stride(1),
                    w_packed.stride(0), w_packed.stride(1),
                    scales.stride(0), scales.stride(1),
                    zeros.stride(0), zeros.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                    GROUP_SIZE=group_size, GROUP_SIZE_M=8,
                    num_warps=4, num_stages=1,
                )
                return c[:rows_ref]

            return (
                run,
                ref,
                2.0 * (n**3) / 1e12,
                "TFLOP/s",
                1e-1,
                2e-2,
                f"W4A16 group_size=128 packed uint8 (4x weight compression); top {rows_ref} rows FP64 ref",
            )

        record_outcome(
            f"triton_w4a16_gemm_{w4_n}",
            "triton_windows",
            f"{w4_n}x{w4_n}x{w4_n}",
            "f16xint4->f16",
            make_w4a16,
        )

        # 5h. Milestone 2: Fused RMSNorm + RoPE (single VRAM pass) vs PyTorch Eager (2 ops)
        @triton.jit
        def triton_fused_rmsnorm_rope_kernel(
            X_ptr,
            W_ptr,
            Y_ptr,
            seq_len,
            n_heads,
            theta_base,
            eps,
            HEAD_DIM: tl.constexpr,
        ):
            HALF_DIM: tl.constexpr = HEAD_DIM // 2
            pos = tl.program_id(0)
            head = tl.program_id(1)
            row_offset = (pos * n_heads + head) * HEAD_DIM

            offs_h = tl.arange(0, HALF_DIM)
            even_idx = 2 * offs_h
            odd_idx = even_idx + 1

            x0 = tl.load(X_ptr + row_offset + even_idx).to(tl.float32)
            x1 = tl.load(X_ptr + row_offset + odd_idx).to(tl.float32)
            w0 = tl.load(W_ptr + even_idx).to(tl.float32)
            w1 = tl.load(W_ptr + odd_idx).to(tl.float32)

            sq_sum = tl.sum(x0 * x0 + x1 * x1, axis=0)
            rstd = tl.rsqrt((sq_sum / HEAD_DIM) + eps)
            nx0 = x0 * rstd * w0
            nx1 = x1 * rstd * w1

            exponent = -(2.0 * offs_h.to(tl.float32)) / HEAD_DIM
            freq = tl.exp(exponent * tl.log(theta_base))
            angle = pos.to(tl.float32) * freq
            c = tl.cos(angle)
            s = tl.sin(angle)

            y0 = nx0 * c - nx1 * s
            y1 = nx0 * s + nx1 * c
            tl.store(Y_ptr + row_offset + even_idx, y0.to(tl.float16))
            tl.store(Y_ptr + row_offset + odd_idx, y1.to(tl.float16))

        rr_seq, rr_heads, rr_dim = (256, 8, 128) if smoke else (2048, 32, 128)

        def make_pt_rmsnorm_rope(seq: int = rr_seq, heads: int = rr_heads, dim: int = rr_dim):
            eps, theta = 1e-5, 10000.0
            half = dim // 2
            x = torch.randn((seq, heads, dim), device=device, dtype=torch.float16)
            w = torch.randn((dim,), device=device, dtype=torch.float16)
            pos = torch.arange(seq, device=device, dtype=torch.float32)[:, None, None]
            idx = torch.arange(half, device=device, dtype=torch.float32)[None, None, :]
            freqs = pos * (theta ** (-2.0 * idx / dim))
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)

            x64 = x.to("cpu", torch.float64)
            w64 = w.to("cpu", torch.float64)
            nx64 = (x64 * torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps) * w64).reshape(seq, heads, half, 2)
            pos_c = torch.arange(seq, dtype=torch.float64)[:, None, None]
            idx_c = torch.arange(half, dtype=torch.float64)[None, None, :]
            freqs_c = pos_c * (theta ** (-2.0 * idx_c / dim))
            cc, sc = torch.cos(freqs_c), torch.sin(freqs_c)
            ref = torch.stack(
                (nx64[..., 0] * cc - nx64[..., 1] * sc, nx64[..., 0] * sc + nx64[..., 1] * cc),
                dim=-1,
            ).reshape(seq, heads, dim)

            def run():
                xf = x.float()
                nx = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * w.float()).reshape(seq, heads, half, 2)
                x0, x1 = nx[..., 0], nx[..., 1]
                return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).reshape(seq, heads, dim).to(torch.float16)

            bytes_moved = (2 * seq * heads * dim + dim) * 2 / 1e9
            return run, ref, bytes_moved, "GB/s", 5e-2, 2e-2, "CPU FP64 reference (minimum 1-pass I/O bytes)"

        record_outcome(
            f"pytorch_rmsnorm_rope_f16_{rr_seq}x{rr_heads}x{rr_dim}",
            "pytorch_hip",
            f"{rr_seq}x{rr_heads}x{rr_dim}",
            "f16",
            make_pt_rmsnorm_rope,
        )

        def make_triton_fused_rmsnorm_rope(seq: int = rr_seq, heads: int = rr_heads, dim: int = rr_dim):
            eps, theta = 1e-5, 10000.0
            half = dim // 2
            x = torch.randn((seq, heads, dim), device=device, dtype=torch.float16)
            w = torch.randn((dim,), device=device, dtype=torch.float16)
            y = torch.empty_like(x)

            x64 = x.to("cpu", torch.float64)
            w64 = w.to("cpu", torch.float64)
            nx64 = (x64 * torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps) * w64).reshape(seq, heads, half, 2)
            pos_c = torch.arange(seq, dtype=torch.float64)[:, None, None]
            idx_c = torch.arange(half, dtype=torch.float64)[None, None, :]
            freqs_c = pos_c * (theta ** (-2.0 * idx_c / dim))
            cc, sc = torch.cos(freqs_c), torch.sin(freqs_c)
            ref = torch.stack(
                (nx64[..., 0] * cc - nx64[..., 1] * sc, nx64[..., 0] * sc + nx64[..., 1] * cc),
                dim=-1,
            ).reshape(seq, heads, dim)

            grid = (seq, heads)

            def run():
                triton_fused_rmsnorm_rope_kernel[grid](
                    x, w, y, seq, heads, theta, eps, HEAD_DIM=dim
                )
                return y

            bytes_moved = (2 * seq * heads * dim + dim) * 2 / 1e9
            return run, ref, bytes_moved, "GB/s", 5e-2, 2e-2, "fused single-pass Triton RMSNorm+RoPE; CPU FP64 ref"

        record_outcome(
            f"triton_fused_rmsnorm_rope_f16_{rr_seq}x{rr_heads}x{rr_dim}",
            "triton_windows",
            f"{rr_seq}x{rr_heads}x{rr_dim}",
            "f16",
            make_triton_fused_rmsnorm_rope,
        )

        # 5i. Milestone 2: Causal Tiled FlashAttention (HEAD_DIM=64) vs PyTorch Causal SDPA Math
        @triton.jit
        def triton_causal_attention_kernel(
            Q_ptr,
            K_ptr,
            V_ptr,
            O_ptr,
            stride_qb,
            stride_qh,
            stride_qm,
            stride_qd,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_vd,
            stride_ob,
            stride_oh,
            stride_om,
            stride_od,
            H,
            S,
            scale,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            HEAD_DIM: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_bh = tl.program_id(1)
            b = pid_bh // H
            h = pid_bh % H

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, HEAD_DIM)
            q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

            m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            end_n = (pid_m + 1) * BLOCK_M
            if end_n > S:
                end_n = S
            for start_n in range(0, end_n, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_ptrs = K_ptr + b * stride_kb + h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
                v_ptrs = V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
                k = tl.load(k_ptrs, mask=offs_n[None, :] < S, other=0.0)
                v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)

                qk = tl.dot(q, k) * scale
                causal_mask = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < S)
                qk = tl.where(causal_mask, qk, float("-inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp(qk - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
                m_i = m_ij

            acc = acc / l_i[:, None]
            o_ptrs = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
            tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < S)

        ca_b, ca_h, ca_s, ca_d = (1, 4, 256, 64) if smoke else (1, 16, 1024, 64)

        def _causal_attn_inputs_and_ref():
            b, h, s, d = ca_b, ca_h, ca_s, ca_d
            q = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
            k = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
            v = torch.randn((b, h, s, d), device=device, dtype=torch.float16)
            qf = q.to("cpu", torch.float32)
            kf = k.to("cpu", torch.float32)
            vf = v.to("cpu", torch.float32)
            causal = torch.triu(torch.ones((s, s), dtype=torch.bool), diagonal=1)
            scores = (qf @ kf.transpose(-2, -1)) / math.sqrt(d)
            scores = scores.masked_fill(causal, float("-inf"))
            ref = torch.softmax(scores, dim=-1) @ vf
            flops = 2.0 * b * h * s * s * d / 1e12
            return q, k, v, ref, b, h, s, d, flops

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def make_pt_causal_attn():
                q, k, v, ref, _b, _h, _s, _d, flops = _causal_attn_inputs_and_ref()

                def run():
                    with sdpa_kernel(SDPBackend.MATH):
                        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

                return run, ref, flops, "TFLOP/s", 5e-2, 2e-2, "2*B*H*S^2*D causal attention FLOPs"

            record_outcome(
                f"pytorch_sdpa_math_causal_{ca_s}",
                "pytorch_sdpa_math",
                f"{ca_b}x{ca_h}x{ca_s}x{ca_d}",
                "f16",
                make_pt_causal_attn,
            )
        except Exception:
            pass

        def make_triton_causal_attn():
            q, k, v, ref, b, h, s, d, flops = _causal_attn_inputs_and_ref()
            o = torch.empty_like(q)
            block_m, block_n = 32, 32
            grid = (triton.cdiv(s, block_m), b * h)
            scale = 1.0 / math.sqrt(d)

            def run():
                triton_causal_attention_kernel[grid](
                    q, k, v, o,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    h, s, scale,
                    BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d,
                    num_warps=4, num_stages=1,
                )
                return o

            return run, ref, flops, "TFLOP/s", 5e-2, 2e-2, "causal tiled online-softmax FlashAttention (D=64); 2*B*H*S^2*D FLOPs"

        record_outcome(
            f"triton_causal_attention_f16_{ca_s}",
            "triton_windows",
            f"{ca_b}x{ca_h}x{ca_s}x{ca_d}",
            "f16",
            make_triton_causal_attn,
        )
    except Exception as exc:
        benchmarks.append(
            {
                "name": "triton_suite_import",
                "backend": "triton_windows",
                "shape": "n/a",
                "dtype": "n/a",
                "status": "unsupported",
                "latency": None,
                "throughput_unit": None,
                "throughput_median": None,
                "accuracy": None,
                "peak_vram_allocated_bytes": None,
                "peak_vram_reserved_bytes": None,
                "peak_rss_bytes": peak_rss_bytes(),
                "error": f"{type(exc).__name__}: {exc}",
                "notes": None,
            }
        )

    # 6. Stability loop (100 consecutive iterations unless disabled / smoke)
    if stability_iters <= 0:
        stability_report = None
    elif triton_gemm_fn_for_stability is not None and triton_ref_for_stability is not None:
        try:
            out_s, lat_s, _, _, _ = bench_gpu_fn(
                triton_gemm_fn_for_stability, 5, stability_iters
            )
            acc_s = check_accuracy(out_s, triton_ref_for_stability, 1e-1, 2e-2)
            stability_report = {
                "workload": "triton_gemm_f16_512x512x512",
                "iterations": stability_iters,
                "failures": 0 if acc_s["passed"] else 1,
                "all_finite": acc_s["all_finite"],
                "all_correct": acc_s["passed"],
                "latency": lat_s,
                "error": None if acc_s["passed"] else "stability final check failed",
            }
        except Exception as exc:
            stability_report = {
                "workload": "triton_gemm_f16_512x512x512",
                "iterations": stability_iters,
                "failures": stability_iters,
                "all_finite": False,
                "all_correct": False,
                "latency": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        try:
            sa = torch.randn((512, 512), device=device, dtype=torch.float16)
            sb = torch.randn((512, 512), device=device, dtype=torch.float16)
            sref = sa.to("cpu", torch.float64) @ sb.to("cpu", torch.float64)
            out_s, lat_s, _, _, _ = bench_gpu_fn(
                lambda: torch.matmul(sa, sb), 5, stability_iters
            )
            acc_s = check_accuracy(out_s, sref, 1e-1, 2e-2)
            stability_report = {
                "workload": "pytorch_gemm_f16_512x512x512",
                "iterations": stability_iters,
                "failures": 0 if acc_s["passed"] else 1,
                "all_finite": acc_s["all_finite"],
                "all_correct": acc_s["passed"],
                "latency": lat_s,
                "error": None if acc_s["passed"] else "stability final check failed",
            }
        except Exception as exc:
            stability_report = {
                "workload": "pytorch_gemm_f16_512x512x512",
                "iterations": stability_iters,
                "failures": stability_iters,
                "all_finite": False,
                "all_correct": False,
                "latency": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "provenance": provenance,
        "benchmarks": benchmarks,
        "stability": stability_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RDNA Forge PyTorch ROCm + Triton benchmark suite"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--stability-iters", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Run tiny 256x256 verification pass")
    args = parser.parse_args()

    if args.smoke:
        result = run_all_benchmarks(0, 1, 0, smoke=True)
    else:
        result = run_all_benchmarks(args.warmup, args.iters, args.stability_iters, smoke=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
