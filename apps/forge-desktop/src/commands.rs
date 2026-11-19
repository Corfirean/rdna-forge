use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use forge_benchmark::{
    run_vulkan_suite, VulkanBenchConfig, VulkanBenchReport, VulkanKernelSelection,
};
use forge_core::{CapabilityResult, DiagnosticReport, SupportStatus};
use forge_device::discover_all;
use forge_profiler::{
    check_competing_workload_warning, check_vram_pressure_warning, read_all_gpu_tuning,
    reset_gpu_tuning_to_factory, set_gpu_power_limit, snapshot_all_gpus, GpuTelemetrySample,
    GpuTuningReport, PowerLimitChangeResult, TuningResetResult,
};
use forge_runtime::{
    default_untested_capabilities, probe_all_backends, probe_amf_capabilities,
    run_live_capability_checks, AmfCapabilitiesReport,
};
use serde::{Deserialize, Serialize};
use tauri::State;

/// Shared desktop application state synchronizing exclusive GPU/ADLX workloads.
#[derive(Clone, Default)]
pub struct AppState {
    gpu_busy: Arc<AtomicBool>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            gpu_busy: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns `true` when an exclusive verification, tuning, or benchmark task is active.
    pub fn is_busy(&self) -> bool {
        self.gpu_busy.load(Ordering::Acquire)
    }

    /// Atomically marks the GPU busy and returns a RAII guard that releases on drop.
    pub fn acquire_gpu(&self, operation: &str) -> Result<BusyGuard, String> {
        self.gpu_busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| BusyGuard {
                flag: Arc::clone(&self.gpu_busy),
            })
            .map_err(|_| {
                format!(
                    "Another GPU operation is currently active; wait for it to finish before starting '{operation}'."
                )
            })
    }
}

/// RAII guard resetting `gpu_busy` on scope exit or task failure.
pub struct BusyGuard {
    flag: Arc<AtomicBool>,
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// Complete diagnostics overview payload returned to the Overview view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopDiagnosticsResponse {
    pub report: DiagnosticReport,
    pub telemetry: Option<Vec<GpuTelemetrySample>>,
}

/// Live ADLX telemetry polling payload with optional pause state during GPU tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopTelemetryResponse {
    pub paused: bool,
    pub pause_reason: Option<String>,
    pub samples: Vec<GpuTelemetrySample>,
    pub warnings: Vec<String>,
    pub error: Option<String>,
}

/// Validated power limit adjustment request (`dry_run` defaults to `true`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopPowerLimitRequest {
    pub pci_device_id: Option<u32>,
    pub limit_pct: i32,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

/// Validated factory reset request (`dry_run` defaults to `true`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopResetTuningRequest {
    pub pci_device_id: Option<u32>,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

/// Vulkan compute benchmark parameters submitted by the Benchmarks view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopBenchRequest {
    pub device_index: usize,
    pub kernel: String,
    pub warmup_iters: usize,
    pub measured_iters: usize,
    pub stability_runs: usize,
    pub batch_dispatches: Option<usize>,
}

impl DesktopBenchRequest {
    /// Parses kernel string and validates safe bounded benchmark parameters.
    pub fn validate_and_build_config(&self) -> Result<VulkanBenchConfig, String> {
        let kernel = match self.kernel.trim() {
            "all" => VulkanKernelSelection::All,
            "gemm" => VulkanKernelSelection::Gemm,
            "gemm_opt" | "gemm-opt" => VulkanKernelSelection::GemmOpt,
            "gemm_i8" | "gemm-i8" => VulkanKernelSelection::GemmI8,
            "gemv_w8" | "gemv-w8" => VulkanKernelSelection::GemvW8,
            "gemv_q4" | "gemv-q4" => VulkanKernelSelection::GemvQ4,
            "gemv_q4_k" | "gemv-q4-k" => VulkanKernelSelection::GemvQ4K,
            "rmsnorm" => VulkanKernelSelection::RmsNorm,
            "rope" => VulkanKernelSelection::Rope,
            "fused_rms_rope" | "fused-rms-rope" => VulkanKernelSelection::FusedRmsRope,
            "flash_attn" | "flash-attn" => VulkanKernelSelection::FlashAttn,
            other => return Err(format!("Unsupported Vulkan kernel selection '{other}'.")),
        };

        if self.warmup_iters > 100 {
            return Err(format!(
                "warmup_iters must be <= 100 (got {}).",
                self.warmup_iters
            ));
        }
        if !(1..=200).contains(&self.measured_iters) {
            return Err(format!(
                "measured_iters must be in 1..=200 (got {}).",
                self.measured_iters
            ));
        }
        if self.stability_runs > 200 {
            return Err(format!(
                "stability_runs must be <= 200 (got {}).",
                self.stability_runs
            ));
        }
        if let Some(batch) = self.batch_dispatches {
            if !(1..=256).contains(&batch) {
                return Err(format!(
                    "batch_dispatches must be in 1..=256 (got {batch})."
                ));
            }
        }

        Ok(VulkanBenchConfig {
            device_index: self.device_index,
            kernel,
            size_override: None,
            warmup_iters: self.warmup_iters,
            measured_iters: self.measured_iters,
            stability_runs: self.stability_runs,
            batch_dispatches: self.batch_dispatches,
        })
    }
}

fn default_true() -> bool {
    true
}

fn collect_sample_warnings(samples: &[GpuTelemetrySample]) -> Vec<String> {
    let mut warnings = Vec::new();
    for sample in samples {
        if let Some(w) = check_competing_workload_warning(sample) {
            warnings.push(w);
        }
        if let (Some(used_mib), Some(total_mib)) = (sample.vram_used_mib, sample.total_vram_mib) {
            if let Some(w) = check_vram_pressure_warning(f64::from(used_mib), total_mib) {
                warnings.push(w);
            }
        }
    }
    warnings
}

#[tauri::command]
pub async fn get_diagnostics(
    state: State<'_, AppState>,
) -> Result<DesktopDiagnosticsResponse, String> {
    let busy = state.is_busy();
    tauri::async_runtime::spawn_blocking(move || {
        let snapshot = discover_all();
        let (backends, missing_dependencies) = probe_all_backends(&snapshot, None);
        let telemetry = if busy { None } else { snapshot_all_gpus().ok() };

        let mut capabilities = vec![CapabilityResult {
            name: "Device discovery".to_string(),
            status: if snapshot.gpus.is_empty() {
                SupportStatus::Unsupported
            } else {
                SupportStatus::Verified
            },
            detail: Some(format!("{} GPU(s) enumerated", snapshot.gpus.len())),
        }];
        capabilities.extend(default_untested_capabilities());

        let report = DiagnosticReport {
            system: snapshot.system,
            gpus: snapshot.gpus,
            backends,
            capabilities,
            missing_dependencies,
        };
        Ok(DesktopDiagnosticsResponse { report, telemetry })
    })
    .await
    .map_err(|e| format!("Diagnostics background task failed: {e}"))?
}

#[tauri::command]
pub async fn run_live_verification(
    state: State<'_, AppState>,
) -> Result<Vec<CapabilityResult>, String> {
    let guard = state.acquire_gpu("live verification")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        let snapshot = discover_all();
        let mut capabilities = vec![CapabilityResult {
            name: "Device discovery".to_string(),
            status: if snapshot.gpus.is_empty() {
                SupportStatus::Unsupported
            } else {
                SupportStatus::Verified
            },
            detail: Some(format!("{} GPU(s) enumerated", snapshot.gpus.len())),
        }];
        capabilities.extend(run_live_capability_checks(&snapshot.gpus));
        capabilities
    })
    .await
    .map_err(|e| format!("Live verification background task failed: {e}"))
}

#[tauri::command]
pub async fn get_telemetry_snapshot(
    state: State<'_, AppState>,
) -> Result<DesktopTelemetryResponse, String> {
    if state.is_busy() {
        return Ok(DesktopTelemetryResponse {
            paused: true,
            pause_reason: Some(
                "Telemetry polling paused while an active GPU task holds exclusive access."
                    .to_string(),
            ),
            samples: Vec::new(),
            warnings: Vec::new(),
            error: None,
        });
    }

    tauri::async_runtime::spawn_blocking(move || match snapshot_all_gpus() {
        Ok(samples) => {
            let warnings = collect_sample_warnings(&samples);
            DesktopTelemetryResponse {
                paused: false,
                pause_reason: None,
                samples,
                warnings,
                error: None,
            }
        }
        Err(err) => DesktopTelemetryResponse {
            paused: false,
            pause_reason: None,
            samples: Vec::new(),
            warnings: Vec::new(),
            error: Some(err.to_string()),
        },
    })
    .await
    .map_err(|e| format!("Telemetry snapshot task failed: {e}"))
}

#[tauri::command]
pub async fn get_tuning_report(state: State<'_, AppState>) -> Result<Vec<GpuTuningReport>, String> {
    let guard = state.acquire_gpu("tuning inspection")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        read_all_gpu_tuning().map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Tuning inspection task failed: {e}"))?
}

#[tauri::command]
pub async fn set_power_limit(
    state: State<'_, AppState>,
    request: DesktopPowerLimitRequest,
) -> Result<PowerLimitChangeResult, String> {
    let guard = state.acquire_gpu("power limit change")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        set_gpu_power_limit(request.pci_device_id, request.limit_pct, request.dry_run)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Power limit task failed: {e}"))?
}

#[tauri::command]
pub async fn reset_tuning(
    state: State<'_, AppState>,
    request: DesktopResetTuningRequest,
) -> Result<TuningResetResult, String> {
    let guard = state.acquire_gpu("factory tuning reset")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        reset_gpu_tuning_to_factory(request.pci_device_id, request.dry_run)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Tuning reset task failed: {e}"))?
}

#[tauri::command]
pub async fn get_video_caps() -> Result<AmfCapabilitiesReport, String> {
    tauri::async_runtime::spawn_blocking(move || {
        probe_amf_capabilities().map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("AMF video capability probe task failed: {e}"))?
}

#[tauri::command]
pub async fn run_vulkan_benchmark(
    state: State<'_, AppState>,
    request: DesktopBenchRequest,
) -> Result<VulkanBenchReport, String> {
    let config = request.validate_and_build_config()?;
    let guard = state.acquire_gpu("Vulkan benchmark")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        run_vulkan_suite(&config).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Vulkan benchmark task failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_guard_enforces_mutual_exclusion_and_releases_on_drop() {
        let state = AppState::new();
        assert!(!state.is_busy());

        {
            let _guard = state.acquire_gpu("benchmark").expect("first acquire");
            assert!(state.is_busy());
            let conflict = state.acquire_gpu("verification");
            assert!(conflict.is_err());
            assert!(conflict
                .err()
                .unwrap()
                .contains("Another GPU operation is currently active"));
        }

        assert!(!state.is_busy());
        assert!(state.acquire_gpu("tuning").is_ok());
    }

    #[test]
    fn bench_request_validates_kernel_and_iteration_bounds() {
        let valid = DesktopBenchRequest {
            device_index: 0,
            kernel: "gemv_w8".to_string(),
            warmup_iters: 5,
            measured_iters: 20,
            stability_runs: 0,
            batch_dispatches: Some(16),
        };
        let cfg = valid.validate_and_build_config().expect("valid config");
        assert_eq!(cfg.kernel, VulkanKernelSelection::GemvW8);
        assert_eq!(cfg.batch_dispatches, Some(16));

        let q4k = DesktopBenchRequest {
            kernel: "gemv_q4_k".to_string(),
            ..valid.clone()
        }
        .validate_and_build_config()
        .expect("q4_k config");
        assert_eq!(q4k.kernel, VulkanKernelSelection::GemvQ4K);

        let fa = DesktopBenchRequest {
            kernel: "flash_attn".to_string(),
            batch_dispatches: Some(1),
            ..valid.clone()
        }
        .validate_and_build_config()
        .expect("flash_attn config");
        assert_eq!(fa.kernel, VulkanKernelSelection::FlashAttn);

        let bad_kernel = DesktopBenchRequest {
            kernel: "unknown_kernel".to_string(),
            ..valid.clone()
        };
        assert!(bad_kernel.validate_and_build_config().is_err());

        let zero_iters = DesktopBenchRequest {
            measured_iters: 0,
            ..valid.clone()
        };
        assert!(zero_iters.validate_and_build_config().is_err());

        let too_many_iters = DesktopBenchRequest {
            measured_iters: 201,
            ..valid.clone()
        };
        assert!(too_many_iters.validate_and_build_config().is_err());

        let too_many_warmup = DesktopBenchRequest {
            warmup_iters: 101,
            ..valid.clone()
        };
        assert!(too_many_warmup.validate_and_build_config().is_err());

        let too_many_stability = DesktopBenchRequest {
            stability_runs: 201,
            ..valid.clone()
        };
        assert!(too_many_stability.validate_and_build_config().is_err());

        let bad_batch = DesktopBenchRequest {
            batch_dispatches: Some(257),
            ..valid
        };
        assert!(bad_batch.validate_and_build_config().is_err());
    }

    #[test]
    fn tuning_mutation_requests_default_to_dry_run() {
        let power: DesktopPowerLimitRequest =
            serde_json::from_str(r#"{"pci_device_id":29615,"limit_pct":5}"#).unwrap();
        assert!(power.dry_run);
        assert_eq!(power.limit_pct, 5);
        assert_eq!(power.pci_device_id, Some(29615));

        let reset: DesktopResetTuningRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert!(reset.dry_run);
        assert_eq!(reset.pci_device_id, None);
    }
}
