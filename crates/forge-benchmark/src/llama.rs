use std::path::{Path, PathBuf};
use std::process::Command;

use forge_core::{ForgeError, ForgeResult, SystemInfo};
use forge_device::discover_system;
use serde::{Deserialize, Serialize};

use crate::memory::peak_process_rss_bytes;
use crate::stats::{compute_latency_stats, LatencyStats};

/// Parameters controlling an apples-to-apples `llama-bench` comparison across backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaBenchConfig {
    pub model_path: PathBuf,
    pub vulkan_bin: Option<PathBuf>,
    pub hip_bin: Option<PathBuf>,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub reps: u32,
    pub n_gpu_layers: u32,
    pub vulkan_main_gpu: i32,
    pub hip_main_gpu: i32,
    pub vulkan_device: Option<String>,
    pub hip_device: Option<String>,
}

/// Raw entry emitted by `llama-bench -o json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaBenchRawEntry {
    #[serde(default)]
    pub build_commit: String,
    #[serde(default)]
    pub build_number: u64,
    #[serde(default)]
    pub cpu_info: String,
    #[serde(default)]
    pub gpu_info: String,
    #[serde(default)]
    pub backends: String,
    #[serde(default)]
    pub model_filename: String,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub model_size: u64,
    #[serde(default)]
    pub model_n_params: u64,
    #[serde(default)]
    pub n_gpu_layers: i32,
    #[serde(default)]
    pub main_gpu: i32,
    #[serde(default)]
    pub n_threads: i32,
    #[serde(default)]
    pub n_batch: i32,
    #[serde(default)]
    pub n_ubatch: i32,
    #[serde(default)]
    pub n_prompt: u32,
    #[serde(default)]
    pub n_gen: u32,
    #[serde(default)]
    pub avg_ts: f64,
    #[serde(default)]
    pub stddev_ts: f64,
    #[serde(default)]
    pub avg_ns: u64,
    #[serde(default)]
    pub stddev_ns: u64,
    #[serde(default)]
    pub samples_ns: Vec<u64>,
    #[serde(default)]
    pub samples_ts: Vec<f64>,
}

/// Normalized summary entry with median and dispersion computed from `samples_ns` / `samples_ts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaBenchEntry {
    pub backend_label: String,
    pub gpu_info: String,
    pub backends: String,
    pub build_number: u64,
    pub build_commit: String,
    pub model_type: String,
    pub model_size_bytes: u64,
    pub test_kind: String,
    pub n_prompt: u32,
    pub n_gen: u32,
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub avg_tok_s: f64,
    pub median_tok_s: f64,
    pub stddev_tok_s: f64,
    pub latency: LatencyStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendRunOutcome {
    pub backend: String,
    pub binary_path: String,
    pub succeeded: bool,
    pub entries: Vec<LlamaBenchEntry>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaComparisonReport {
    pub system: SystemInfo,
    pub model_path: String,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub reps: u32,
    pub n_gpu_layers: u32,
    pub runs: Vec<BackendRunOutcome>,
    pub peak_rss_bytes: Option<u64>,
}

/// Parses raw JSON output from `llama-bench -o json` into normalized entries.
pub fn parse_llama_bench_json(
    json_str: &str,
    backend_label: &str,
) -> ForgeResult<Vec<LlamaBenchEntry>> {
    let raw_entries: Vec<LlamaBenchRawEntry> = serde_json::from_str(json_str.trim())
        .map_err(|e| ForgeError::ParseError(format!("invalid llama-bench JSON: {e}")))?;

    if raw_entries.is_empty() {
        return Err(ForgeError::ParseError(
            "llama-bench JSON array was empty".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(raw_entries.len());
    for raw in raw_entries {
        let samples_ms: Vec<f64> = if !raw.samples_ns.is_empty() {
            raw.samples_ns
                .iter()
                .map(|&ns| (ns as f64) / 1_000_000.0)
                .collect()
        } else if raw.avg_ns > 0 {
            vec![(raw.avg_ns as f64) / 1_000_000.0]
        } else {
            return Err(ForgeError::ParseError(
                "llama-bench entry missing both samples_ns and avg_ns".to_string(),
            ));
        };

        let latency = compute_latency_stats(1, &samples_ms)?;
        let tokens = if raw.n_prompt > 0 && raw.n_gen == 0 {
            raw.n_prompt as f64
        } else if raw.n_gen > 0 && raw.n_prompt == 0 {
            raw.n_gen as f64
        } else {
            (raw.n_prompt + raw.n_gen) as f64
        };

        let median_tok_s = if !raw.samples_ts.is_empty() {
            let mut ts = raw.samples_ts.clone();
            ts.sort_by(|a, b| a.total_cmp(b));
            let mid = ts.len() / 2;
            if ts.len().is_multiple_of(2) {
                (ts[mid - 1] + ts[mid]) / 2.0
            } else {
                ts[mid]
            }
        } else if latency.median_ms > 0.0 {
            tokens / (latency.median_ms / 1000.0)
        } else {
            raw.avg_ts
        };

        let test_kind = match (raw.n_prompt, raw.n_gen) {
            (p, 0) if p > 0 => format!("pp{p}"),
            (0, g) if g > 0 => format!("tg{g}"),
            (p, g) => format!("pp{p}+tg{g}"),
        };

        out.push(LlamaBenchEntry {
            backend_label: backend_label.to_string(),
            gpu_info: raw.gpu_info,
            backends: raw.backends,
            build_number: raw.build_number,
            build_commit: raw.build_commit,
            model_type: raw.model_type,
            model_size_bytes: raw.model_size,
            test_kind,
            n_prompt: raw.n_prompt,
            n_gen: raw.n_gen,
            n_gpu_layers: raw.n_gpu_layers,
            main_gpu: raw.main_gpu,
            avg_tok_s: raw.avg_ts,
            median_tok_s,
            stddev_tok_s: raw.stddev_ts,
            latency,
        });
    }

    Ok(out)
}

/// Runs `llama-bench` across configured Vulkan and/or HIP binaries under identical test parameters.
pub fn run_llama_comparison(config: &LlamaBenchConfig) -> ForgeResult<LlamaComparisonReport> {
    if !config.model_path.exists() {
        return Err(ForgeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("model file not found: {}", config.model_path.display()),
        )));
    }
    if config.vulkan_bin.is_none() && config.hip_bin.is_none() {
        return Err(ForgeError::RuntimeUnavailable {
            runtime: "llama-bench".to_string(),
            reason: "pass at least one of --vulkan-bin or --hip-bin".to_string(),
        });
    }

    let system = discover_system();
    let mut runs = Vec::new();

    if let Some(ref bin) = config.vulkan_bin {
        runs.push(execute_single_backend(
            "vulkan",
            bin,
            config,
            config.vulkan_main_gpu,
            config.vulkan_device.as_deref(),
        ));
    }

    if let Some(ref bin) = config.hip_bin {
        runs.push(execute_single_backend(
            "hip",
            bin,
            config,
            config.hip_main_gpu,
            config.hip_device.as_deref(),
        ));
    }

    Ok(LlamaComparisonReport {
        system,
        model_path: config.model_path.display().to_string(),
        prompt_tokens: config.prompt_tokens,
        gen_tokens: config.gen_tokens,
        reps: config.reps,
        n_gpu_layers: config.n_gpu_layers,
        runs,
        peak_rss_bytes: peak_process_rss_bytes(),
    })
}

fn execute_single_backend(
    label: &str,
    bin_path: &Path,
    config: &LlamaBenchConfig,
    main_gpu: i32,
    device_selector: Option<&str>,
) -> BackendRunOutcome {
    if !bin_path.exists() {
        return BackendRunOutcome {
            backend: label.to_string(),
            binary_path: bin_path.display().to_string(),
            succeeded: false,
            entries: Vec::new(),
            error: Some(format!("binary not found: {}", bin_path.display())),
        };
    }

    let mut cmd = Command::new(bin_path);
    if let Some(parent) = bin_path.parent() {
        cmd.current_dir(parent);
    }
    cmd.arg("-m")
        .arg(&config.model_path)
        .arg("-p")
        .arg(config.prompt_tokens.to_string())
        .arg("-n")
        .arg(config.gen_tokens.to_string())
        .arg("-r")
        .arg(config.reps.to_string())
        .arg("-ngl")
        .arg(config.n_gpu_layers.to_string())
        .arg("-mg")
        .arg(main_gpu.to_string())
        .arg("-o")
        .arg("json");

    if let Some(dev) = device_selector {
        cmd.arg("-dev").arg(dev);
    }

    match cmd.output() {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            match parse_llama_bench_json(&stdout, label) {
                Ok(entries) => BackendRunOutcome {
                    backend: label.to_string(),
                    binary_path: bin_path.display().to_string(),
                    succeeded: true,
                    entries,
                    error: None,
                },
                Err(err) => BackendRunOutcome {
                    backend: label.to_string(),
                    binary_path: bin_path.display().to_string(),
                    succeeded: false,
                    entries: Vec::new(),
                    error: Some(err.to_string()),
                },
            }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let code = out
                .status
                .code()
                .map(|c| format!("exit code {c}"))
                .unwrap_or_else(|| "terminated by signal".to_string());
            BackendRunOutcome {
                backend: label.to_string(),
                binary_path: bin_path.display().to_string(),
                succeeded: false,
                entries: Vec::new(),
                error: Some(format!("{code}; stderr: {stderr}; stdout: {stdout}")),
            }
        }
        Err(err) => BackendRunOutcome {
            backend: label.to_string(),
            binary_path: bin_path.display().to_string(),
            succeeded: false,
            entries: Vec::new(),
            error: Some(format!("failed to spawn {}: {err}", bin_path.display())),
        },
    }
}
