use std::path::{Path, PathBuf};
use std::process::Command;

use forge_core::{ForgeError, ForgeResult};
use forge_device::discover_gpus;
use forge_profiler::TelemetrySummary;
use serde::{Deserialize, Serialize};

use crate::run_with_pre_post_telemetry;

use crate::stats::LatencyStats;

/// Configuration for executing the Python PyTorch ROCm + Triton benchmark suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TritonBenchConfig {
    pub python_exe: PathBuf,
    pub script_path: PathBuf,
    pub warmup_iters: usize,
    pub measured_iters: usize,
    #[serde(default)]
    pub smoke: bool,
    #[serde(default)]
    pub collect_telemetry: bool,
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

    // Validate and resolve the interpreter up front: a clear error instead of a failed launch, and
    // bare names are searched on PATH only (never the application directory).
    let python_exe = forge_core::resolve_python_executable(&config.python_exe).map_err(|e| {
        ForgeError::RuntimeUnavailable {
            runtime: "Python/Triton".to_string(),
            reason: e.to_string(),
        }
    })?;

    let run_cmd = || {
        let mut cmd = Command::new(&python_exe);
        cmd.arg(&config.script_path);
        if config.smoke {
            cmd.arg("--smoke");
        } else {
            cmd.arg("--warmup")
                .arg(config.warmup_iters.to_string())
                .arg("--iters")
                .arg(config.measured_iters.to_string());
        }
        cmd.output()
    };

    let (output_res, telemetry) = if config.collect_telemetry {
        let primary_device_id = discover_gpus().first().map(|g| g.device_id);
        run_with_pre_post_telemetry(primary_device_id, run_cmd)
    } else {
        (run_cmd(), None)
    };
    let output = output_res.map_err(|e| ForgeError::RuntimeUnavailable {
        runtime: "Python/Triton".to_string(),
        reason: format!("failed to launch '{}': {e}", python_exe.display()),
    })?;

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
