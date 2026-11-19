use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use forge_benchmark::{
    apply_lm_studio_config, apply_windows_and_ollama_env_tweaks, autotune_llm_model,
    discover_local_llm_assets, inspect_gguf_model, plan_llm_configuration,
    run_video_benchmark_suite, run_vulkan_suite, EnvAndOllamaPatchResult, KvCacheQuant,
    LlmAutotuneConfig, LlmAutotuneReport, LlmOptimizationPlan, LmStudioPatchResult, LocalLlmAssets,
    VideoBenchReport, VideoBenchSuiteConfig, VulkanBenchConfig, VulkanBenchReport,
    VulkanKernelSelection,
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
    run_live_capability_checks, AmfCapabilitiesReport, AmfEncodeCodec, AmfQualityPreset,
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

/// AMF VCN hardware video encode benchmark parameters submitted by the Video view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopVideoBenchRequest {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_mbps: u32,
    pub preset: String,
    pub warmup_frames: u32,
    pub measured_frames: u32,
}

impl DesktopVideoBenchRequest {
    /// Parses codec/preset strings and validates safe bounded VCN benchmark parameters.
    pub fn validate_and_build_config(&self) -> Result<VideoBenchSuiteConfig, String> {
        let codecs = match self.codec.trim() {
            "all" => vec![
                AmfEncodeCodec::H264Avc,
                AmfEncodeCodec::Hevc,
                AmfEncodeCodec::Hevc10Bit,
            ],
            "h264" | "h264_avc" => vec![AmfEncodeCodec::H264Avc],
            "hevc" => vec![AmfEncodeCodec::Hevc],
            "hevc10" | "hevc_10bit" => vec![AmfEncodeCodec::Hevc10Bit],
            other => return Err(format!("Unsupported AMF video codec selection '{other}'.")),
        };

        let preset = match self.preset.trim() {
            "speed" => AmfQualityPreset::Speed,
            "balanced" => AmfQualityPreset::Balanced,
            "quality" => AmfQualityPreset::Quality,
            other => return Err(format!("Unsupported AMF quality preset '{other}'.")),
        };

        if !(128..=7680).contains(&self.width) || !self.width.is_multiple_of(2) {
            return Err(format!(
                "width must be an even integer in 128..=7680 (got {}).",
                self.width
            ));
        }
        if !(128..=4320).contains(&self.height) || !self.height.is_multiple_of(2) {
            return Err(format!(
                "height must be an even integer in 128..=4320 (got {}).",
                self.height
            ));
        }
        if !(1..=240).contains(&self.fps) {
            return Err(format!("fps must be in 1..=240 (got {}).", self.fps));
        }
        if !(1..=500).contains(&self.bitrate_mbps) {
            return Err(format!(
                "bitrate_mbps must be in 1..=500 (got {}).",
                self.bitrate_mbps
            ));
        }
        if self.warmup_frames > 120 {
            return Err(format!(
                "warmup_frames must be <= 120 (got {}).",
                self.warmup_frames
            ));
        }
        if !(10..=600).contains(&self.measured_frames) {
            return Err(format!(
                "measured_frames must be in 10..=600 (got {}).",
                self.measured_frames
            ));
        }

        Ok(VideoBenchSuiteConfig {
            codecs,
            width: self.width,
            height: self.height,
            fps: self.fps,
            target_bitrate_bps: u64::from(self.bitrate_mbps) * 1_000_000,
            preset,
            warmup_frames: self.warmup_frames,
            measured_frames: self.measured_frames,
        })
    }
}

/// Request payload for static GGUF inspection and anti-WDDM-spill VRAM planning in the LLM Optimizer view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLlmPlanRequest {
    pub model_path: Option<String>,
    pub context_tokens: u32,
    pub safety_margin_mib: u64,
}

/// Request payload for empirical `llama-bench` RDNA2 autotuning in the LLM Optimizer view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLlmAutotuneRequest {
    pub model_path: Option<String>,
    pub vulkan_bin: Option<String>,
    pub hip_bin: Option<String>,
    pub rocm_lib_dir: Option<String>,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub context_depth: u32,
    pub reps: u32,
}

/// Request payload for previewing (`dry_run = true`) or writing (`dry_run = false`) LM Studio configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLmStudioPatchRequest {
    pub model_path: Option<String>,
    pub profile_id: Option<String>,
    #[serde(default = "default_context_tokens")]
    pub context_tokens: u32,
    #[serde(default = "default_safety_margin_mib")]
    pub safety_margin_mib: u64,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

/// Request payload for previewing (`dry_run = true`) or applying (`dry_run = false`) Windows/Ollama user env variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopEnvPatchRequest {
    pub model_path: Option<String>,
    #[serde(default = "default_context_tokens")]
    pub context_tokens: u32,
    pub kv_quant: Option<String>,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

fn default_context_tokens() -> u32 {
    4096
}

fn default_safety_margin_mib() -> u64 {
    512
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

#[tauri::command]
pub async fn run_video_benchmark(
    state: State<'_, AppState>,
    request: DesktopVideoBenchRequest,
) -> Result<VideoBenchReport, String> {
    let config = request.validate_and_build_config()?;
    let guard = state.acquire_gpu("VCN video benchmark")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        run_video_benchmark_suite(&config).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("VCN video benchmark task failed: {e}"))?
}

fn resolve_desktop_model_path(explicit: Option<&str>) -> Result<PathBuf, String> {
    if let Some(s) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(PathBuf::from(s));
    }
    let assets = discover_local_llm_assets();
    assets
        .gguf_models
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| "No local .gguf model found. Specify a path to a .gguf file.".to_string())
}

#[tauri::command]
pub async fn get_llm_assets() -> Result<LocalLlmAssets, String> {
    tauri::async_runtime::spawn_blocking(discover_local_llm_assets)
        .await
        .map_err(|e| format!("LLM asset discovery task failed: {e}"))
}

#[tauri::command]
pub async fn get_llm_plan(request: DesktopLlmPlanRequest) -> Result<LlmOptimizationPlan, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let path = resolve_desktop_model_path(request.model_path.as_deref())?;
        let meta = inspect_gguf_model(&path).map_err(|e| e.to_string())?;
        plan_llm_configuration(
            &meta,
            request.context_tokens.clamp(512, 262_144),
            request.safety_margin_mib.clamp(0, 4096),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LLM plan task failed: {e}"))?
}

#[tauri::command]
pub async fn run_llm_autotune(
    state: State<'_, AppState>,
    request: DesktopLlmAutotuneRequest,
) -> Result<LlmAutotuneReport, String> {
    let guard = state.acquire_gpu("LLM autotuner")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        let model_path = resolve_desktop_model_path(request.model_path.as_deref())?;
        let assets = discover_local_llm_assets();
        let non_empty = |opt: Option<String>| -> Option<PathBuf> {
            opt.map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        };
        let vulkan_bin =
            non_empty(request.vulkan_bin).or_else(|| assets.vulkan_bench_bin.map(PathBuf::from));
        let hip_bin =
            non_empty(request.hip_bin).or_else(|| assets.hip_bench_bin.map(PathBuf::from));
        let rocm_lib_dir =
            non_empty(request.rocm_lib_dir).or_else(|| assets.rocm_lib_dir.map(PathBuf::from));

        let cfg = LlmAutotuneConfig {
            model_path,
            vulkan_bin,
            hip_bin,
            rocm_lib_dir,
            prompt_tokens: request.prompt_tokens.clamp(32, 2048),
            gen_tokens: request.gen_tokens.clamp(16, 512),
            context_depth: request.context_depth.clamp(0, 32_768),
            reps: request.reps.clamp(1, 5),
        };
        autotune_llm_model(&cfg).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LLM autotuner task failed: {e}"))?
}

#[tauri::command]
pub async fn apply_lm_studio_patch(
    request: DesktopLmStudioPatchRequest,
) -> Result<LmStudioPatchResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let model_path = resolve_desktop_model_path(request.model_path.as_deref())?;
        apply_lm_studio_config(
            &model_path,
            request.profile_id.as_deref(),
            request.context_tokens.clamp(512, 262_144),
            request.safety_margin_mib.clamp(0, 4096),
            request.dry_run,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LM Studio config patch task failed: {e}"))?
}

#[tauri::command]
pub async fn apply_env_and_ollama_patch(
    request: DesktopEnvPatchRequest,
) -> Result<EnvAndOllamaPatchResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let model_path = resolve_desktop_model_path(request.model_path.as_deref())?;
        let kv_quant = match request.kv_quant.as_deref().map(str::trim) {
            Some("f16") => KvCacheQuant::F16,
            Some("q4_0") => KvCacheQuant::Q4_0,
            _ => KvCacheQuant::Q8_0,
        };
        apply_windows_and_ollama_env_tweaks(
            &model_path,
            request.context_tokens.clamp(512, 262_144),
            kv_quant,
            None,
            request.dry_run,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Windows/Ollama env patch task failed: {e}"))?
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

        let lms_patch: DesktopLmStudioPatchRequest =
            serde_json::from_str(r#"{"profile_id":"vulkan_wddm_priority"}"#).unwrap();
        assert!(lms_patch.dry_run);
        assert_eq!(lms_patch.context_tokens, 4096);

        let env_patch: DesktopEnvPatchRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert!(env_patch.dry_run);
        assert_eq!(env_patch.context_tokens, 4096);
    }

    #[test]
    fn video_bench_request_validates_codecs_resolution_and_frame_bounds() {
        let valid = DesktopVideoBenchRequest {
            codec: "all".to_string(),
            width: 1920,
            height: 1080,
            fps: 60,
            bitrate_mbps: 12,
            preset: "speed".to_string(),
            warmup_frames: 15,
            measured_frames: 120,
        };
        let cfg = valid
            .validate_and_build_config()
            .expect("valid video bench config");
        assert_eq!(cfg.codecs.len(), 3);
        assert_eq!(cfg.target_bitrate_bps, 12_000_000);
        assert_eq!(cfg.preset, AmfQualityPreset::Speed);

        assert!(DesktopVideoBenchRequest {
            codec: "av1".to_string(),
            ..valid.clone()
        }
        .validate_and_build_config()
        .is_err());

        assert!(DesktopVideoBenchRequest {
            width: 1921,
            ..valid.clone()
        }
        .validate_and_build_config()
        .is_err());

        assert!(DesktopVideoBenchRequest {
            measured_frames: 5,
            ..valid
        }
        .validate_and_build_config()
        .is_err());
    }
}
