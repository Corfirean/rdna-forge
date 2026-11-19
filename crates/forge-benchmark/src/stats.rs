use forge_core::{ForgeError, ForgeResult, GpuDeviceInfo, SystemInfo};
use forge_runtime::NumericalAccuracy;
use serde::{Deserialize, Serialize};

/// Statistical summary of repeated benchmark latency measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    pub warmup_iters: usize,
    pub measured_iters: usize,
    pub min_ms: f64,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
    pub mad_ms: f64,
    pub iqr_ms: f64,
    pub stddev_ms: f64,
    pub cv_pct: f64,
}

/// Complete record of a single kernel or library benchmark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub backend: String,
    pub shape: String,
    pub dtype: String,
    pub timing_source: String,
    pub latency: LatencyStats,
    pub throughput_unit: String,
    pub throughput_median: f64,
    pub accuracy: Option<NumericalAccuracy>,
    pub peak_rss_bytes: Option<u64>,
    pub notes: Option<String>,
}

/// Result of repeated kernel dispatches verifying numeric and runtime stability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityResult {
    pub workload: String,
    pub iterations: usize,
    pub failures: usize,
    pub all_finite: bool,
    pub all_correct: bool,
    pub latency: LatencyStats,
}

/// Combined hardware/system metadata for a benchmark report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEnvironment {
    pub system: SystemInfo,
    pub device: GpuDeviceInfo,
}

/// Computes robust location and dispersion statistics for latency samples in milliseconds.
pub fn compute_latency_stats(warmup_iters: usize, samples_ms: &[f64]) -> ForgeResult<LatencyStats> {
    if samples_ms.is_empty() {
        return Err(ForgeError::InvalidDimension(
            "cannot compute latency stats from zero samples".to_string(),
        ));
    }
    if samples_ms.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(ForgeError::ExecutionFailed(
            "benchmark samples contain NaN, Inf, or negative duration".to_string(),
        ));
    }

    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    let min_ms = sorted[0];
    let max_ms = sorted[n - 1];
    let median_ms = sorted_percentile(&sorted, 50.0);
    let q25 = sorted_percentile(&sorted, 25.0);
    let q75 = sorted_percentile(&sorted, 75.0);
    let iqr_ms = (q75 - q25).max(0.0);
    let p95_ms = sorted_percentile(&sorted, 95.0);

    let sum: f64 = sorted.iter().sum();
    let mean_ms = sum / (n as f64);

    let variance = if n > 1 {
        sorted
            .iter()
            .map(|&v| {
                let d = v - mean_ms;
                d * d
            })
            .sum::<f64>()
            / ((n - 1) as f64)
    } else {
        0.0
    };
    let stddev_ms = variance.sqrt();
    let cv_pct = if mean_ms > 0.0 {
        (stddev_ms / mean_ms) * 100.0
    } else {
        0.0
    };

    let mut abs_devs: Vec<f64> = sorted.iter().map(|&v| (v - median_ms).abs()).collect();
    abs_devs.sort_by(|a, b| a.total_cmp(b));
    let mad_ms = sorted_percentile(&abs_devs, 50.0);

    Ok(LatencyStats {
        warmup_iters,
        measured_iters: n,
        min_ms,
        median_ms,
        mean_ms,
        p95_ms,
        max_ms,
        mad_ms,
        iqr_ms,
        stddev_ms,
        cv_pct,
    })
}

fn sorted_percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * ((n - 1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
