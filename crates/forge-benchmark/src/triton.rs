use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use forge_core::{ForgeError, ForgeResult};
use forge_device::discover_gpus;
use forge_profiler::{TelemetryGuard, TelemetrySummary};
use serde::{Deserialize, Serialize};

use crate::stats::LatencyStats;

/// Configuration for executing the Python PyTorch ROCm + Triton benchmark suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TritonBenchConfig {
    pub python_exe: PathBuf,
    pub script_path: PathBuf,
    pub warmup_iters: usize,
    pub measured_iters: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TritonProvenance {
    pub python_version: String,
    pub torch_version: Option<String>,
    pub torch_hip_version: Option<String>,
    pub triton_version: Option<String>,
    pub triton_windows_version: Option<String>,
    pub device_name: Option<String>,
    pub gcn_arch_name: Option<String>,
    pub total_vram_bytes: Option<u64>,
    pub os: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TritonAccuracy {
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub rmse: f64,
    pub all_finite: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TritonKernelOutcome {
    pub name: String,
    pub backend: String,
    pub shape: String,
    pub dtype: String,
    pub status: String,
    pub latency: Option<LatencyStats>,
    pub throughput_unit: Option<String>,
    pub throughput_median: Option<f64>,
    pub accuracy: Option<TritonAccuracy>,
    pub peak_vram_allocated_bytes: Option<u64>,
    pub peak_vram_reserved_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub error: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TritonStabilityOutcome {
    pub workload: String,
    pub iterations: usize,
    pub failures: usize,
    pub all_finite: bool,
    pub all_correct: bool,
    pub latency: Option<LatencyStats>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TritonSuiteReport {
    pub provenance: TritonProvenance,
    pub benchmarks: Vec<TritonKernelOutcome>,
    pub stability: Option<TritonStabilityOutcome>,
    #[serde(default)]
    pub telemetry: Option<TelemetrySummary>,
}

/// Parses the JSON output produced by `modules/ai/triton/bench_suite.py`.
pub fn parse_triton_suite_json(json_str: &str) -> ForgeResult<TritonSuiteReport> {
    serde_json::from_str::<TritonSuiteReport>(json_str.trim())
        .map_err(|e| ForgeError::ParseError(format!("invalid Triton suite JSON: {e}")))
}

/// Runs `modules/ai/triton/bench_suite.py` in the specified Python environment.
pub fn run_triton_suite(config: &TritonBenchConfig) -> ForgeResult<TritonSuiteReport> {
    if !config.script_path.exists() {
        return Err(ForgeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Triton benchmark script not found: {}",
                config.script_path.display()
            ),
        )));
    }

    let primary_device_id = discover_gpus().first().map(|g| g.device_id);
    let guard = TelemetryGuard::start(primary_device_id, Duration::from_millis(100));

    let output = Command::new(&config.python_exe)
        .arg(&config.script_path)
        .arg("--warmup")
        .arg(config.warmup_iters.to_string())
        .arg("--iters")
        .arg(config.measured_iters.to_string())
        .output()
        .map_err(|e| ForgeError::RuntimeUnavailable {
            runtime: "Python/Triton".to_string(),
            reason: format!("failed to launch '{}': {e}", config.python_exe.display()),
        })?;
    let telemetry = guard.and_then(|g| g.stop_and_summarize());

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return Err(ForgeError::ExecutionFailed(format!(
            "Triton suite failed ({:?}): stderr={stderr}; stdout={stdout}",
            output.status.code()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut report = parse_triton_suite_json(&stdout)?;
    report.telemetry = telemetry;
    Ok(report)
}

/// Locates `modules/ai/triton/bench_suite.py` relative to current working directory or repo root.
pub fn resolve_default_triton_script(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    let rel = Path::new("modules/ai/triton/bench_suite.py");
    if rel.exists() {
        return Some(rel.to_path_buf());
    }
    None
}
