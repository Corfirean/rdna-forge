use std::time::Duration;

use forge_core::{ForgeError, ForgeResult, GpuDeviceInfo, SystemInfo};
use forge_device::discover_system;
use forge_profiler::{TelemetryGuard, TelemetrySummary};
use forge_runtime::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    pack_i8_b_transposed_k4, pack_i8_row_major_k4, ref_gemm_f32_exact_f64, ref_gemm_i8_to_i32,
    ref_rmsnorm_f32, ref_rope_f32, GpuTiming, NumericalAccuracy, VulkanComputeEngine,
};
use serde::{Deserialize, Serialize};

use crate::memory::peak_process_rss_bytes;
use crate::stats::{compute_latency_stats, BenchmarkResult, StabilityResult};

/// Selects which Vulkan compute kernels to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanKernelSelection {
    All,
    Gemm,
    GemmOpt,
    GemmI8,
    RmsNorm,
    Rope,
}

/// Configuration for the Vulkan compute benchmark suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanBenchConfig {
    pub device_index: usize,
    pub kernel: VulkanKernelSelection,
    pub size_override: Option<u32>,
    pub warmup_iters: usize,
    pub measured_iters: usize,
    pub stability_runs: usize,
}

impl Default for VulkanBenchConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            kernel: VulkanKernelSelection::All,
            size_override: None,
            warmup_iters: 10,
            measured_iters: 50,
            stability_runs: 50,
        }
    }
}

/// Complete report produced by the Vulkan compute benchmark suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanBenchReport {
    pub system: SystemInfo,
    pub device: GpuDeviceInfo,
    pub benchmarks: Vec<BenchmarkResult>,
    pub stability: Option<StabilityResult>,
    pub peak_rss_bytes: Option<u64>,
    #[serde(default)]
    pub telemetry: Option<TelemetrySummary>,
}

/// Executes the configured Vulkan compute benchmarks and optional repeated stability test.
pub fn run_vulkan_suite(config: &VulkanBenchConfig) -> ForgeResult<VulkanBenchReport> {
    if config.measured_iters == 0 {
        return Err(ForgeError::InvalidDimension(
            "measured_iters must be > 0".to_string(),
        ));
    }

    let system = discover_system();
    let engine = VulkanComputeEngine::new(config.device_index)?;
    let device = engine.device_info().clone();
    let guard = TelemetryGuard::start(Some(device.device_id), Duration::from_millis(250));
    let mut benchmarks = Vec::new();

    let gemm_sizes: Vec<u32> = match config.size_override {
        Some(s) => vec![s],
        None => vec![1024, 2048],
    };

    let run_baseline_gemm = matches!(
        config.kernel,
        VulkanKernelSelection::All | VulkanKernelSelection::Gemm
    );
    let run_opt_gemm = matches!(
        config.kernel,
        VulkanKernelSelection::All | VulkanKernelSelection::Gemm | VulkanKernelSelection::GemmOpt
    );

    if run_baseline_gemm || run_opt_gemm {
        for n in gemm_sizes.iter().copied() {
            let mut baseline_ms = None;
            if run_baseline_gemm {
                let base_res = bench_gemm_f32_case(&engine, n, config, false, None)?;
                baseline_ms = Some(base_res.latency.median_ms);
                benchmarks.push(base_res);
            }
            if run_opt_gemm {
                let opt_res = bench_gemm_f32_case(&engine, n, config, true, baseline_ms)?;
                benchmarks.push(opt_res);
            }
        }
    }

    if matches!(
        config.kernel,
        VulkanKernelSelection::All | VulkanKernelSelection::GemmI8
    ) {
        if engine.supports_int8_dp4a() {
            for n in gemm_sizes {
                benchmarks.push(bench_gemm_i8_case(&engine, n, config)?);
            }
        } else if matches!(config.kernel, VulkanKernelSelection::GemmI8) {
            return Err(ForgeError::UnsupportedCapability(format!(
                "Vulkan INT8 DP4A is not supported on '{}'",
                device.name
            )));
        }
    }

    if matches!(
        config.kernel,
        VulkanKernelSelection::All | VulkanKernelSelection::RmsNorm
    ) {
        let dim = config.size_override.unwrap_or(4096);
        benchmarks.push(bench_rmsnorm_case(&engine, dim, dim, config)?);
    }

    if matches!(
        config.kernel,
        VulkanKernelSelection::All | VulkanKernelSelection::Rope
    ) {
        let seq = config.size_override.unwrap_or(1024);
        benchmarks.push(bench_rope_case(&engine, seq, 32, 128, config)?);
    }

    let stability = if config.stability_runs > 0 {
        Some(run_stability_check(&engine, config.stability_runs)?)
    } else {
        None
    };

    let telemetry = guard.and_then(|g| g.stop_and_summarize());

    Ok(VulkanBenchReport {
        system,
        device,
        benchmarks,
        stability,
        peak_rss_bytes: peak_process_rss_bytes(),
        telemetry,
    })
}

fn extract_timing_samples(timings: &[GpuTiming]) -> (&'static str, Vec<f64>) {
    if timings.iter().all(|t| t.gpu_ms.is_some()) {
        (
            "vulkan_timestamp_query",
            timings.iter().filter_map(|t| t.gpu_ms).collect(),
        )
    } else {
        (
            "host_fence_wall_clock",
            timings.iter().map(|t| t.wall_ms).collect(),
        )
    }
}

fn validate_gemm_f32_accuracy(
    actual: &[f32],
    a: &[f32],
    b: &[f32],
    n: usize,
) -> ForgeResult<(Option<NumericalAccuracy>, String)> {
    if (n as u64).pow(3) <= 1024u64.pow(3) {
        let expected = ref_gemm_f32_exact_f64(a, b, n, n, n)?;
        let acc = compare_slices(actual, &expected, 1e-2, 1e-3)?;
        Ok((Some(acc), "full exact FP64 CPU reference".to_string()))
    } else {
        let sample_rows = 32usize.min(n);
        let expected_rows = ref_gemm_f32_exact_f64(&a[..sample_rows * n], b, sample_rows, n, n)?;
        let acc = compare_slices(&actual[..sample_rows * n], &expected_rows, 1e-2, 1e-3)?;
        Ok((
            Some(acc),
            format!("validated top {sample_rows} rows against FP64 CPU reference"),
        ))
    }
}

fn bench_gemm_f32_case(
    engine: &VulkanComputeEngine,
    n: u32,
    config: &VulkanBenchConfig,
    optimized: bool,
    baseline_median_ms: Option<f64>,
) -> ForgeResult<BenchmarkResult> {
    if n == 0 {
        return Err(ForgeError::InvalidDimension(
            "GEMM size must be > 0".to_string(),
        ));
    }
    let elems = (n as usize)
        .checked_mul(n as usize)
        .ok_or_else(|| ForgeError::InvalidDimension("GEMM size overflow".to_string()))?;

    let a = deterministic_f32_data(elems, 0x1001);
    let b = deterministic_f32_data(elems, 0x2002);

    let (actual, timings) = if optimized {
        engine.bench_gemm_f32_tiled_vec4(
            &a,
            &b,
            n,
            n,
            n,
            config.warmup_iters,
            config.measured_iters,
        )?
    } else {
        engine.bench_gemm_f32(&a, &b, n, n, n, config.warmup_iters, config.measured_iters)?
    };

    let (timing_source, samples_ms) = extract_timing_samples(&timings);
    let latency = compute_latency_stats(config.warmup_iters, &samples_ms)?;

    let flops = 2.0 * (n as f64).powi(3);
    let gflops = if latency.median_ms > 0.0 {
        flops / (latency.median_ms / 1000.0) / 1e9
    } else {
        0.0
    };

    let (accuracy, ref_note) = validate_gemm_f32_accuracy(&actual, &a, &b, n as usize)?;
    let notes = if optimized {
        let speedup_note = match baseline_median_ms {
            Some(base) if latency.median_ms > 0.0 => {
                format!(
                    "register-blocked 128x64 vec4 kernel ({:.2}x vs 16x16 baseline); ",
                    base / latency.median_ms
                )
            }
            _ => "register-blocked 128x64 vec4 kernel; ".to_string(),
        };
        Some(format!("{speedup_note}{ref_note}"))
    } else {
        Some(ref_note)
    };

    let name = if optimized {
        format!("gemm_f32_tiled_vec4_{n}")
    } else {
        format!("gemm_f32_baseline_{n}")
    };

    Ok(BenchmarkResult {
        name,
        backend: "vulkan_wgsl_naga".to_string(),
        shape: format!("{n}x{n}x{n}"),
        dtype: "f32".to_string(),
        timing_source: timing_source.to_string(),
        latency,
        throughput_unit: "GFLOP/s".to_string(),
        throughput_median: gflops,
        accuracy,
        peak_rss_bytes: peak_process_rss_bytes(),
        notes,
    })
}

fn bench_gemm_i8_case(
    engine: &VulkanComputeEngine,
    n: u32,
    config: &VulkanBenchConfig,
) -> ForgeResult<BenchmarkResult> {
    if n == 0 || !n.is_multiple_of(4) {
        return Err(ForgeError::InvalidDimension(format!(
            "INT8 DP4A GEMM size must be non-zero and divisible by 4, got {n}"
        )));
    }
    let n_us = n as usize;
    let elems = n_us
        .checked_mul(n_us)
        .ok_or_else(|| ForgeError::InvalidDimension("INT8 GEMM size overflow".to_string()))?;

    let a = deterministic_i8_data(elems, 0x6006);
    let b = deterministic_i8_data(elems, 0x7007);
    let a_packed = pack_i8_row_major_k4(&a, n_us, n_us)?;
    let b_t_packed = pack_i8_b_transposed_k4(&b, n_us, n_us)?;
    let k_packed = n / 4;

    let (actual, timings) = engine.bench_gemm_i8_dp4a(
        &a_packed,
        &b_t_packed,
        n,
        k_packed,
        n,
        config.warmup_iters,
        config.measured_iters,
    )?;

    let (timing_source, samples_ms) = extract_timing_samples(&timings);
    let latency = compute_latency_stats(config.warmup_iters, &samples_ms)?;

    let ops = 2.0 * (n as f64).powi(3);
    let tops = if latency.median_ms > 0.0 {
        ops / (latency.median_ms / 1000.0) / 1e12
    } else {
        0.0
    };

    let sample_rows = if (n as u64).pow(3) <= 512u64.pow(3) {
        n_us
    } else {
        32usize.min(n_us)
    };
    let expected = ref_gemm_i8_to_i32(&a[..sample_rows * n_us], &b, sample_rows, n_us, n_us)?;
    let acc = compare_i32_exact(&actual[..sample_rows * n_us], &expected)?;
    let ref_note = if sample_rows == n_us {
        "SPIR-V 1.6 OpSDot (dot4I8Packed); full exact integer CPU reference".to_string()
    } else {
        format!("SPIR-V 1.6 OpSDot (dot4I8Packed); exact integer check on top {sample_rows} rows")
    };

    Ok(BenchmarkResult {
        name: format!("gemm_i8_dp4a_{n}"),
        backend: "vulkan_wgsl_naga_sdot".to_string(),
        shape: format!("{n}x{n}x{n}"),
        dtype: "i8->i32".to_string(),
        timing_source: timing_source.to_string(),
        latency,
        throughput_unit: "TOPS".to_string(),
        throughput_median: tops,
        accuracy: Some(acc),
        peak_rss_bytes: peak_process_rss_bytes(),
        notes: Some(ref_note),
    })
}

fn bench_rmsnorm_case(
    engine: &VulkanComputeEngine,
    rows: u32,
    cols: u32,
    config: &VulkanBenchConfig,
) -> ForgeResult<BenchmarkResult> {
    let elems = (rows as usize)
        .checked_mul(cols as usize)
        .ok_or_else(|| ForgeError::InvalidDimension("RMSNorm size overflow".to_string()))?;
    let x = deterministic_f32_data(elems, 0x3003);
    let w = deterministic_f32_data(cols as usize, 0x4004);
    let eps = 1e-5_f32;

    let (actual, timings) = engine.bench_rmsnorm_f32(
        &x,
        &w,
        rows,
        cols,
        eps,
        config.warmup_iters,
        config.measured_iters,
    )?;

    let (timing_source, samples_ms) = extract_timing_samples(&timings);
    let latency = compute_latency_stats(config.warmup_iters, &samples_ms)?;

    let expected = ref_rmsnorm_f32(&x, &w, rows as usize, cols as usize, eps)?;
    let acc: NumericalAccuracy = compare_slices(&actual, &expected, 1e-4, 1e-4)?;

    // Memory moved: read x + read weight (broadcasted once per row -> minimum memory traffic 2*x + w)
    let bytes_moved = ((elems as f64) * 2.0 + (cols as f64)) * 4.0;
    let gb_s = if latency.median_ms > 0.0 {
        bytes_moved / (latency.median_ms / 1000.0) / 1e9
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        name: format!("rmsnorm_f32_{rows}x{cols}"),
        backend: "vulkan_wgsl_naga".to_string(),
        shape: format!("{rows}x{cols}"),
        dtype: "f32".to_string(),
        timing_source: timing_source.to_string(),
        latency,
        throughput_unit: "GB/s".to_string(),
        throughput_median: gb_s,
        accuracy: Some(acc),
        peak_rss_bytes: peak_process_rss_bytes(),
        notes: Some("full exact FP64 CPU reference".to_string()),
    })
}

fn bench_rope_case(
    engine: &VulkanComputeEngine,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    config: &VulkanBenchConfig,
) -> ForgeResult<BenchmarkResult> {
    let elems = (seq_len as usize)
        .checked_mul(n_heads as usize)
        .and_then(|v| v.checked_mul(head_dim as usize))
        .ok_or_else(|| ForgeError::InvalidDimension("RoPE size overflow".to_string()))?;
    let input = deterministic_f32_data(elems, 0x5005);
    let theta_base = 10_000.0_f32;

    let (actual, timings) = engine.bench_rope_f32(
        &input,
        seq_len,
        n_heads,
        head_dim,
        theta_base,
        config.warmup_iters,
        config.measured_iters,
    )?;

    let (timing_source, samples_ms) = extract_timing_samples(&timings);
    let latency = compute_latency_stats(config.warmup_iters, &samples_ms)?;

    let expected = ref_rope_f32(
        &input,
        seq_len as usize,
        n_heads as usize,
        head_dim as usize,
        theta_base,
    )?;
    let acc = compare_slices(&actual, &expected, 5e-4, 5e-4)?;

    let bytes_moved = (elems as f64) * 2.0 * 4.0;
    let gb_s = if latency.median_ms > 0.0 {
        bytes_moved / (latency.median_ms / 1000.0) / 1e9
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        name: format!("rope_f32_{seq_len}x{n_heads}x{head_dim}"),
        backend: "vulkan_wgsl_naga".to_string(),
        shape: format!("{seq_len}x{n_heads}x{head_dim}"),
        dtype: "f32".to_string(),
        timing_source: timing_source.to_string(),
        latency,
        throughput_unit: "GB/s".to_string(),
        throughput_median: gb_s,
        accuracy: Some(acc),
        peak_rss_bytes: peak_process_rss_bytes(),
        notes: Some("full exact FP64 CPU reference".to_string()),
    })
}

fn run_stability_check(
    engine: &VulkanComputeEngine,
    iterations: usize,
) -> ForgeResult<StabilityResult> {
    let n = 512u32;
    let elems = (n as usize) * (n as usize);
    let a = deterministic_f32_data(elems, 0x7777);
    let b = deterministic_f32_data(elems, 0x8888);
    let expected = ref_gemm_f32_exact_f64(&a, &b, n as usize, n as usize, n as usize)?;

    let (actual, timings) = engine.bench_gemm_f32(&a, &b, n, n, n, 2, iterations)?;
    let (_src, samples_ms) = extract_timing_samples(&timings);
    let latency = compute_latency_stats(2, &samples_ms)?;
    let acc = compare_slices(&actual, &expected, 1e-2, 1e-3)?;

    Ok(StabilityResult {
        workload: "vulkan_gemm_f32_512x512x512".to_string(),
        iterations,
        failures: usize::from(!acc.passed),
        all_finite: acc.all_finite,
        all_correct: acc.passed,
        latency,
    })
}
