use forge_core::{ForgeError, ForgeResult, GpuDeviceInfo, SystemInfo};
use forge_device::discover_system;
use forge_runtime::{
    compare_slices, deterministic_f32_data, ref_gemm_f32_exact_f64, ref_rmsnorm_f32, ref_rope_f32,
    GpuTiming, NumericalAccuracy, VulkanComputeEngine,
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
    let mut benchmarks = Vec::new();

    if matches!(
        config.kernel,
        VulkanKernelSelection::All | VulkanKernelSelection::Gemm
    ) {
        let sizes: Vec<u32> = match config.size_override {
            Some(s) => vec![s],
            None => vec![1024, 2048],
        };
        for n in sizes {
            benchmarks.push(bench_gemm_case(&engine, n, config)?);
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

    Ok(VulkanBenchReport {
        system,
        device,
        benchmarks,
        stability,
        peak_rss_bytes: peak_process_rss_bytes(),
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

fn bench_gemm_case(
    engine: &VulkanComputeEngine,
    n: u32,
    config: &VulkanBenchConfig,
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

    let (actual, timings) =
        engine.bench_gemm_f32(&a, &b, n, n, n, config.warmup_iters, config.measured_iters)?;

    let (timing_source, samples_ms) = extract_timing_samples(&timings);
    let latency = compute_latency_stats(config.warmup_iters, &samples_ms)?;

    let flops = 2.0 * (n as f64).powi(3);
    let gflops = if latency.median_ms > 0.0 {
        flops / (latency.median_ms / 1000.0) / 1e9
    } else {
        0.0
    };

    // For matrices up to 1024^3, compute full exact FP64 reference; above that, validate first 32 rows.
    let (accuracy, notes) = if (n as u64).pow(3) <= 1024u64.pow(3) {
        let expected = ref_gemm_f32_exact_f64(&a, &b, n as usize, n as usize, n as usize)?;
        let acc = compare_slices(&actual, &expected, 1e-2, 1e-3)?;
        (Some(acc), Some("full exact FP64 CPU reference".to_string()))
    } else {
        let sample_rows = 32usize.min(n as usize);
        let expected_rows = ref_gemm_f32_exact_f64(
            &a[..sample_rows * n as usize],
            &b,
            sample_rows,
            n as usize,
            n as usize,
        )?;
        let acc = compare_slices(
            &actual[..sample_rows * n as usize],
            &expected_rows,
            1e-2,
            1e-3,
        )?;
        (
            Some(acc),
            Some(format!(
                "validated first {sample_rows} rows against FP64 CPU reference"
            )),
        )
    };

    Ok(BenchmarkResult {
        name: format!("gemm_f32_{n}"),
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
    let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;

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
