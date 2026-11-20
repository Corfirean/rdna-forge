use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use forge_benchmark::{
    apply_lm_studio_config, apply_windows_and_ollama_env_tweaks, autotune_llm_model,
    discover_local_llm_assets, generate_github_hardware_report, inspect_gguf_model,
    plan_llm_configuration_for_device, recommend_models_for_device, rollback_lm_studio_config,
    rollback_windows_env_tweaks, run_video_benchmark_suite, run_video_decode_benchmark_suite,
    run_vulkan_suite, EnvAndOllamaPatchResult, EnvRollbackResult, GpuModelRecommendationReport,
    HardwareReportOutput, KvCacheQuant, LlmAutotuneConfig, LlmAutotuneReport, LlmOptimizationPlan,
    LmStudioPatchResult, LmStudioRollbackResult, LocalLlmAssets, VideoBenchReport,
    VideoBenchSuiteConfig, VideoDecodeBenchReport, VideoDecodeBenchSuiteConfig, VulkanBenchConfig,
    VulkanBenchReport, VulkanKernelSelection,
};
use forge_core::{
    match_profile_for_device, ArchitectureCapabilities, CapabilityResult, DiagnosticReport,
    GpuDeviceInfo, SupportStatus,
};
use forge_device::discover_all;
use forge_profiler::{
    check_competing_workload_warning, check_vram_pressure_warning, query_top_vram_processes,
    read_all_gpu_tuning, reset_gpu_tuning_to_factory, set_gpu_power_limit, snapshot_all_gpus,
    GpuTelemetrySample, GpuTuningReport, PowerLimitChangeResult, ProcessVramUsage,
    TuningResetResult,
};
use forge_runtime::{
    default_untested_capabilities, probe_all_backends, probe_amf_capabilities,
    run_live_capability_checks, AmfCapabilitiesReport, AmfDecodeCodec, AmfEncodeCodec,
    AmfQualityPreset,
};
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::apply_gate::{
    gate_key, parse_kv_quant_strict, select_llm_target, select_tuning_target,
    validate_device_index, validate_model_path, validate_pci_device_id, validate_power_limit,
    validate_profile_id, ApplyGate, DeviceIdentity,
};

/// Shared desktop application state synchronizing exclusive GPU/ADLX workloads.
#[derive(Clone, Default)]
pub struct AppState {
    gpu_busy: Arc<AtomicBool>,
    /// Backend-side permission for mutating commands: an apply needs a matching dry-run preview.
    gate: Arc<ApplyGate>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            gpu_busy: Arc::new(AtomicBool::new(false)),
            gate: Arc::new(ApplyGate::new()),
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

/// Matched built-in profile and architecture capability metadata for a discovered GPU.
#[derive(Debug, Clone, Serialize)]
pub struct DesktopGpuProfileMatch {
    pub gpu_index: usize,
    pub matched_profile_id: Option<String>,
    pub matched_profile_name: Option<String>,
    pub vcn_generation: String,
    pub arch_capabilities: ArchitectureCapabilities,
}

/// Complete diagnostics overview payload returned to the Overview view.
#[derive(Debug, Clone, Serialize)]
pub struct DesktopDiagnosticsResponse {
    pub report: DiagnosticReport,
    pub telemetry: Option<Vec<GpuTelemetrySample>>,
    pub gpu_metadata: Vec<DesktopGpuProfileMatch>,
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

fn validate_video_bench_geometry(
    width: u32,
    height: u32,
    fps: u32,
    bitrate_mbps: u32,
    warmup_frames: u32,
    measured_frames: u32,
) -> Result<(), String> {
    if !(128..=7680).contains(&width) || !width.is_multiple_of(2) {
        return Err(format!(
            "width must be an even integer in 128..=7680 (got {width})."
        ));
    }
    if !(128..=4320).contains(&height) || !height.is_multiple_of(2) {
        return Err(format!(
            "height must be an even integer in 128..=4320 (got {height})."
        ));
    }
    if !(1..=240).contains(&fps) {
        return Err(format!("fps must be in 1..=240 (got {fps})."));
    }
    if !(1..=500).contains(&bitrate_mbps) {
        return Err(format!(
            "bitrate_mbps must be in 1..=500 (got {bitrate_mbps})."
        ));
    }
    if warmup_frames > 120 {
        return Err(format!(
            "warmup_frames must be <= 120 (got {warmup_frames})."
        ));
    }
    if !(10..=600).contains(&measured_frames) {
        return Err(format!(
            "measured_frames must be in 10..=600 (got {measured_frames})."
        ));
    }
    Ok(())
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
        let trimmed_codec = self.codec.trim();
        let skip_unsupported = trimmed_codec == "all";
        let codecs = match trimmed_codec {
            "all" => vec![
                AmfEncodeCodec::H264Avc,
                AmfEncodeCodec::Hevc,
                AmfEncodeCodec::Hevc10Bit,
                AmfEncodeCodec::Av1,
            ],
            "h264" | "h264_avc" => vec![AmfEncodeCodec::H264Avc],
            "hevc" => vec![AmfEncodeCodec::Hevc],
            "hevc10" | "hevc_10bit" => vec![AmfEncodeCodec::Hevc10Bit],
            "av1" => vec![AmfEncodeCodec::Av1],
            "av110" | "av1_10bit" => vec![AmfEncodeCodec::Av110Bit],
            other => return Err(format!("Unsupported AMF video codec selection '{other}'.")),
        };

        let preset = match self.preset.trim() {
            "speed" => AmfQualityPreset::Speed,
            "balanced" => AmfQualityPreset::Balanced,
            "quality" => AmfQualityPreset::Quality,
            other => return Err(format!("Unsupported AMF quality preset '{other}'.")),
        };

        validate_video_bench_geometry(
            self.width,
            self.height,
            self.fps,
            self.bitrate_mbps,
            self.warmup_frames,
            self.measured_frames,
        )?;

        Ok(VideoBenchSuiteConfig {
            codecs,
            width: self.width,
            height: self.height,
            fps: self.fps,
            target_bitrate_bps: u64::from(self.bitrate_mbps) * 1_000_000,
            preset,
            warmup_frames: self.warmup_frames,
            measured_frames: self.measured_frames,
            skip_unsupported,
        })
    }
}

/// AMF VCN hardware video decode benchmark parameters submitted by the Video view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopVideoDecodeBenchRequest {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    #[serde(alias = "source_bitrate_mbps")]
    pub bitrate_mbps: u32,
    pub warmup_frames: u32,
    pub measured_frames: u32,
}

impl DesktopVideoDecodeBenchRequest {
    /// Parses decode codec strings and validates safe bounded VCN decode parameters.
    pub fn validate_and_build_config(&self) -> Result<VideoDecodeBenchSuiteConfig, String> {
        let codecs = match self.codec.trim() {
            "all" => vec![
                AmfDecodeCodec::H264Avc,
                AmfDecodeCodec::Hevc,
                AmfDecodeCodec::Hevc10Bit,
            ],
            "h264" | "h264_avc" => vec![AmfDecodeCodec::H264Avc],
            "hevc" => vec![AmfDecodeCodec::Hevc],
            "hevc10" | "hevc_10bit" => vec![AmfDecodeCodec::Hevc10Bit],
            other => {
                return Err(format!(
                    "Unsupported AMF video decode codec selection '{other}'."
                ))
            }
        };

        validate_video_bench_geometry(
            self.width,
            self.height,
            self.fps,
            self.bitrate_mbps,
            self.warmup_frames,
            self.measured_frames,
        )?;

        Ok(VideoDecodeBenchSuiteConfig {
            codecs,
            width: self.width,
            height: self.height,
            fps: self.fps,
            source_bitrate_bps: u64::from(self.bitrate_mbps) * 1_000_000,
            warmup_frames: self.warmup_frames,
            measured_frames: self.measured_frames,
        })
    }
}

/// Request payload for static GGUF inspection and anti-WDDM-spill VRAM planning in the LLM Optimizer view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLlmPlanRequest {
    pub model_path: Option<String>,
    #[serde(default)]
    pub device_index: Option<usize>,
    pub context_tokens: u32,
    pub safety_margin_mib: u64,
}

/// Request payload for empirical `llama-bench` AMD Radeon autotuning in the LLM Optimizer view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLlmAutotuneRequest {
    pub model_path: Option<String>,
    #[serde(default)]
    pub device_index: Option<usize>,
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
    #[serde(default)]
    pub device_index: Option<usize>,
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
    #[serde(default)]
    pub device_index: Option<usize>,
    #[serde(default = "default_context_tokens")]
    pub context_tokens: u32,
    pub kv_quant: Option<String>,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

/// Request payload for previewing or restoring LM Studio `.json.bak` backups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLmStudioRollbackRequest {
    pub model_path: Option<String>,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

/// Request payload for previewing or deleting RDNA Forge-managed `HKCU\Environment` keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopEnvRollbackRequest {
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

/// Request payload for VRAM-aware popular LLM model recommendations in the LLM Optimizer view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopLlmRecommendRequest {
    #[serde(default)]
    pub device_index: Option<usize>,
    #[serde(default = "default_context_tokens")]
    pub context_tokens: u32,
    #[serde(default = "default_safety_margin_mib")]
    pub safety_margin_mib: u64,
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

        let gpu_metadata: Vec<DesktopGpuProfileMatch> = snapshot
            .gpus
            .iter()
            .map(|gpu| {
                let matched = match_profile_for_device(gpu);
                DesktopGpuProfileMatch {
                    gpu_index: gpu.index,
                    matched_profile_id: matched.as_ref().map(|m| m.id.clone()),
                    matched_profile_name: matched.as_ref().map(|m| m.profile.device.name.clone()),
                    vcn_generation: gpu.architecture.vcn_generation().to_string(),
                    arch_capabilities: gpu.architecture.arch_capabilities(),
                }
            })
            .collect();

        let report = DiagnosticReport {
            system: snapshot.system,
            gpus: snapshot.gpus,
            backends,
            capabilities,
            missing_dependencies,
            windows_audit: Some(snapshot.windows_audit),
        };
        Ok(DesktopDiagnosticsResponse {
            report,
            telemetry,
            gpu_metadata,
        })
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
pub async fn generate_hardware_report(
    state: State<'_, AppState>,
    verify: Option<bool>,
) -> Result<HardwareReportOutput, String> {
    let run_verify = verify.unwrap_or(false);
    let guard = if run_verify {
        Some(state.acquire_gpu("hardware report verification")?)
    } else {
        None
    };
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        generate_github_hardware_report(run_verify).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Hardware report task failed: {e}"))?
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
pub async fn get_top_vram_processes(limit: Option<usize>) -> Result<Vec<ProcessVramUsage>, String> {
    let clamped = limit.unwrap_or(10).clamp(1, 50);
    tauri::async_runtime::spawn_blocking(move || {
        query_top_vram_processes(clamped).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Top VRAM process query task failed: {e}"))?
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
    let key = prepare_power_limit(&request)?;
    let guard = state.acquire_gpu("power limit change")?;
    let pci = request.pci_device_id;
    let identity = resolve_identity(move |gpus| select_tuning_target(gpus, pci)).await?;
    enforce_gate(&state.gate, &key, &identity, request.dry_run)?;
    let dry_run = request.dry_run;
    let result = tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        set_gpu_power_limit(request.pci_device_id, request.limit_pct, request.dry_run)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Power limit task failed: {e}"))??;
    if dry_run && result.succeeded() {
        state.gate.record_preview(&key, &identity);
    }
    Ok(result)
}

#[tauri::command]
pub async fn reset_tuning(
    state: State<'_, AppState>,
    request: DesktopResetTuningRequest,
) -> Result<TuningResetResult, String> {
    let key = prepare_reset_tuning(&request)?;
    let guard = state.acquire_gpu("factory tuning reset")?;
    let pci = request.pci_device_id;
    let identity = resolve_identity(move |gpus| select_tuning_target(gpus, pci)).await?;
    enforce_gate(&state.gate, &key, &identity, request.dry_run)?;
    let dry_run = request.dry_run;
    let result = tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        reset_gpu_tuning_to_factory(request.pci_device_id, request.dry_run)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Tuning reset task failed: {e}"))??;
    if dry_run && result.succeeded() {
        state.gate.record_preview(&key, &identity);
    }
    Ok(result)
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

#[tauri::command]
pub async fn run_video_decode_benchmark(
    state: State<'_, AppState>,
    request: DesktopVideoDecodeBenchRequest,
) -> Result<VideoDecodeBenchReport, String> {
    let config = request.validate_and_build_config()?;
    let guard = state.acquire_gpu("VCN video decode benchmark")?;
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = guard;
        run_video_decode_benchmark_suite(&config).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("VCN video decode benchmark task failed: {e}"))?
}

fn resolve_desktop_model_path(explicit: Option<&str>) -> Result<PathBuf, String> {
    if let Some(path) = validate_model_path(explicit)? {
        return Ok(path);
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
        plan_llm_configuration_for_device(
            &meta,
            request.device_index,
            request.context_tokens.clamp(512, 262_144),
            request.safety_margin_mib.clamp(0, 4096),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LLM plan task failed: {e}"))?
}

#[tauri::command]
pub async fn get_llm_recommendations(
    request: DesktopLlmRecommendRequest,
) -> Result<GpuModelRecommendationReport, String> {
    tauri::async_runtime::spawn_blocking(move || {
        recommend_models_for_device(
            request.device_index,
            request.context_tokens.clamp(1024, 131_072),
            request.safety_margin_mib.clamp(0, 4096),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LLM recommendation task failed: {e}"))?
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
            device_index: request.device_index,
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
    state: State<'_, AppState>,
    request: DesktopLmStudioPatchRequest,
) -> Result<LmStudioPatchResult, String> {
    let prepared = prepare_lm_studio_patch(&request)?;
    let device_index = request.device_index;
    let identity = resolve_identity(move |gpus| select_llm_target(gpus, device_index)).await?;
    enforce_gate(&state.gate, &prepared.key, &identity, request.dry_run)?;
    let dry_run = request.dry_run;
    let result = tauri::async_runtime::spawn_blocking(move || {
        apply_lm_studio_config(
            &prepared.model_path,
            request.device_index,
            prepared.profile_id.as_deref(),
            prepared.context_tokens,
            prepared.safety_margin_mib,
            request.dry_run,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LM Studio config patch task failed: {e}"))??;
    if dry_run {
        state.gate.record_preview(&prepared.key, &identity);
    }
    Ok(result)
}

#[tauri::command]
pub async fn apply_env_and_ollama_patch(
    state: State<'_, AppState>,
    request: DesktopEnvPatchRequest,
) -> Result<EnvAndOllamaPatchResult, String> {
    let prepared = prepare_env_patch(&request)?;
    let device_index = request.device_index;
    let identity = resolve_identity(move |gpus| select_llm_target(gpus, device_index)).await?;
    enforce_gate(&state.gate, &prepared.key, &identity, request.dry_run)?;
    let dry_run = request.dry_run;
    let result = tauri::async_runtime::spawn_blocking(move || {
        apply_windows_and_ollama_env_tweaks(
            &prepared.model_path,
            request.device_index,
            prepared.context_tokens,
            prepared.kv_quant,
            None,
            request.dry_run,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Windows/Ollama env patch task failed: {e}"))??;
    if dry_run {
        state.gate.record_preview(&prepared.key, &identity);
    }
    Ok(result)
}

#[tauri::command]
pub async fn rollback_lm_studio_patch(
    state: State<'_, AppState>,
    request: DesktopLmStudioRollbackRequest,
) -> Result<LmStudioRollbackResult, String> {
    let (model_path, key) = prepare_lm_studio_rollback(&request)?;
    // Restores `.json.bak` files; independent of which GPU exists, so no device identity.
    let identity = DeviceIdentity::none();
    enforce_gate(&state.gate, &key, &identity, request.dry_run)?;
    let dry_run = request.dry_run;
    let result = tauri::async_runtime::spawn_blocking(move || {
        rollback_lm_studio_config(model_path.as_deref(), request.dry_run).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("LM Studio rollback task failed: {e}"))??;
    if dry_run {
        state.gate.record_preview(&key, &identity);
    }
    Ok(result)
}

#[tauri::command]
pub async fn rollback_env_patch(
    state: State<'_, AppState>,
    request: DesktopEnvRollbackRequest,
) -> Result<EnvRollbackResult, String> {
    let key = prepare_env_rollback();
    // Removes RDNA Forge-managed HKCU\\Environment values; independent of any GPU.
    let identity = DeviceIdentity::none();
    enforce_gate(&state.gate, &key, &identity, request.dry_run)?;
    let dry_run = request.dry_run;
    let result = tauri::async_runtime::spawn_blocking(move || {
        rollback_windows_env_tweaks(request.dry_run).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Windows/Ollama env rollback task failed: {e}"))??;
    if dry_run {
        state.gate.record_preview(&key, &identity);
    }
    Ok(result)
}

/// Authorizes a request at the backend: dry runs are always allowed (they change nothing);
/// an apply requires a matching, unexpired, unused preview made for the same GPU. The frontend's `dry_run` flag alone
/// never grants permission.
fn enforce_gate(
    gate: &ApplyGate,
    key: &str,
    identity: &DeviceIdentity,
    dry_run: bool,
) -> Result<(), String> {
    if dry_run {
        Ok(())
    } else {
        gate.authorize_apply(key, identity)
    }
}

/// Discovers the installed adapters and resolves the identity of the GPU an operation targets.
/// Runs on a blocking thread (discovery probes Vulkan, HIP and the registry). Called at preview
/// time and again at apply time, so a changed or removed device is detected.
async fn resolve_identity<F>(select: F) -> Result<DeviceIdentity, String>
where
    F: FnOnce(&[GpuDeviceInfo]) -> Result<DeviceIdentity, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(move || select(&discover_all().gpus))
        .await
        .map_err(|e| format!("GPU identity resolution failed: {e}"))?
}

/// Validates a power-limit request and returns the gate key for its exact parameters.
fn prepare_power_limit(request: &DesktopPowerLimitRequest) -> Result<String, String> {
    validate_power_limit(request.limit_pct, request.pci_device_id)?;
    Ok(gate_key(
        "power_limit",
        &[
            format!("{:?}", request.pci_device_id),
            request.limit_pct.to_string(),
        ],
    ))
}

fn prepare_reset_tuning(request: &DesktopResetTuningRequest) -> Result<String, String> {
    validate_pci_device_id(request.pci_device_id)?;
    Ok(gate_key(
        "reset_tuning",
        &[format!("{:?}", request.pci_device_id)],
    ))
}

#[derive(Debug)]
struct PreparedLmStudioPatch {
    model_path: PathBuf,
    profile_id: Option<String>,
    context_tokens: u32,
    safety_margin_mib: u64,
    key: String,
}

fn prepare_lm_studio_patch(
    request: &DesktopLmStudioPatchRequest,
) -> Result<PreparedLmStudioPatch, String> {
    validate_device_index(request.device_index)?;
    let profile_id = validate_profile_id(request.profile_id.as_deref())?;
    let model_path = resolve_desktop_model_path(request.model_path.as_deref())?;
    let context_tokens = request.context_tokens.clamp(512, 262_144);
    let safety_margin_mib = request.safety_margin_mib.clamp(0, 4096);
    let key = gate_key(
        "lm_studio_patch",
        &[
            model_path.display().to_string(),
            format!("{:?}", request.device_index),
            profile_id.clone().unwrap_or_default(),
            context_tokens.to_string(),
            safety_margin_mib.to_string(),
        ],
    );
    Ok(PreparedLmStudioPatch {
        model_path,
        profile_id,
        context_tokens,
        safety_margin_mib,
        key,
    })
}

#[derive(Debug)]
struct PreparedEnvPatch {
    model_path: PathBuf,
    context_tokens: u32,
    kv_quant: KvCacheQuant,
    key: String,
}

fn prepare_env_patch(request: &DesktopEnvPatchRequest) -> Result<PreparedEnvPatch, String> {
    validate_device_index(request.device_index)?;
    let kv_quant = parse_kv_quant_strict(request.kv_quant.as_deref())?;
    let model_path = resolve_desktop_model_path(request.model_path.as_deref())?;
    let context_tokens = request.context_tokens.clamp(512, 262_144);
    let key = gate_key(
        "env_patch",
        &[
            model_path.display().to_string(),
            format!("{:?}", request.device_index),
            context_tokens.to_string(),
            kv_quant.as_cli_str().to_string(),
        ],
    );
    Ok(PreparedEnvPatch {
        model_path,
        context_tokens,
        kv_quant,
        key,
    })
}

fn prepare_lm_studio_rollback(
    request: &DesktopLmStudioRollbackRequest,
) -> Result<(Option<PathBuf>, String), String> {
    let model_path = validate_model_path(request.model_path.as_deref())?;
    let key = gate_key(
        "lm_studio_rollback",
        &[model_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<all>".to_string())],
    );
    Ok((model_path, key))
}

fn prepare_env_rollback() -> String {
    gate_key("env_rollback", &[])
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
        assert_eq!(lms_patch.device_index, None);
        assert_eq!(lms_patch.context_tokens, 4096);

        let env_patch: DesktopEnvPatchRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert!(env_patch.dry_run);
        assert_eq!(env_patch.device_index, None);
        assert_eq!(env_patch.context_tokens, 4096);

        let lms_rb: DesktopLmStudioRollbackRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert!(lms_rb.dry_run);
        assert!(lms_rb.model_path.is_none());

        let env_rb: DesktopEnvRollbackRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert!(env_rb.dry_run);
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
        assert_eq!(cfg.codecs.len(), 4);
        assert!(cfg.skip_unsupported);
        assert_eq!(cfg.target_bitrate_bps, 12_000_000);
        assert_eq!(cfg.preset, AmfQualityPreset::Speed);

        let av1_cfg = DesktopVideoBenchRequest {
            codec: "av1".to_string(),
            ..valid.clone()
        }
        .validate_and_build_config()
        .expect("explicit av1 codec should validate cleanly");
        assert_eq!(av1_cfg.codecs, vec![AmfEncodeCodec::Av1]);
        assert!(!av1_cfg.skip_unsupported);

        assert!(DesktopVideoBenchRequest {
            codec: "vp8".to_string(),
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

    #[test]
    fn video_decode_bench_request_validates_codecs_and_bounds() {
        let valid = DesktopVideoDecodeBenchRequest {
            codec: "all".to_string(),
            width: 1920,
            height: 1080,
            fps: 60,
            bitrate_mbps: 12,
            warmup_frames: 15,
            measured_frames: 120,
        };
        let cfg = valid
            .validate_and_build_config()
            .expect("valid video decode bench config");
        assert_eq!(
            cfg.codecs,
            vec![
                AmfDecodeCodec::H264Avc,
                AmfDecodeCodec::Hevc,
                AmfDecodeCodec::Hevc10Bit
            ]
        );
        assert_eq!(cfg.source_bitrate_bps, 12_000_000);

        let hevc_cfg = DesktopVideoDecodeBenchRequest {
            codec: "hevc".to_string(),
            ..valid.clone()
        }
        .validate_and_build_config()
        .expect("single hevc decode codec");
        assert_eq!(hevc_cfg.codecs, vec![AmfDecodeCodec::Hevc]);

        assert!(DesktopVideoDecodeBenchRequest {
            codec: "av1".to_string(),
            ..valid.clone()
        }
        .validate_and_build_config()
        .is_err());

        assert!(DesktopVideoDecodeBenchRequest {
            width: 1921,
            ..valid.clone()
        }
        .validate_and_build_config()
        .is_err());

        assert!(DesktopVideoDecodeBenchRequest {
            measured_frames: 5,
            ..valid
        }
        .validate_and_build_config()
        .is_err());
    }
}

#[cfg(test)]
mod enforcement_tests {
    //! The desktop backend, not the frontend, decides whether a change may be applied.
    //! None of these tests touches hardware, the registry or any user configuration.
    use super::*;

    fn temp_gguf() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rdna_forge_enforce_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.gguf");
        std::fs::write(&p, b"x").unwrap();
        p
    }

    #[test]
    fn missing_dry_run_field_defaults_to_dry_run_on_every_mutating_request() {
        let p: DesktopPowerLimitRequest = serde_json::from_str(r#"{"limit_pct":5}"#).unwrap();
        assert!(p.dry_run);
        let r: DesktopResetTuningRequest = serde_json::from_str("{}").unwrap();
        assert!(r.dry_run);
        let l: DesktopLmStudioPatchRequest = serde_json::from_str("{}").unwrap();
        assert!(l.dry_run);
        let e: DesktopEnvPatchRequest = serde_json::from_str("{}").unwrap();
        assert!(e.dry_run);
        let lr: DesktopLmStudioRollbackRequest = serde_json::from_str("{}").unwrap();
        assert!(lr.dry_run);
        let er: DesktopEnvRollbackRequest = serde_json::from_str("{}").unwrap();
        assert!(er.dry_run);
    }

    #[test]
    fn dry_run_is_allowed_without_a_preview_and_changes_nothing() {
        let gate = ApplyGate::new();
        let req = DesktopPowerLimitRequest {
            pci_device_id: Some(0x73AF),
            limit_pct: 5,
            dry_run: true,
        };
        let key = prepare_power_limit(&req).unwrap();
        assert!(enforce_gate(&gate, &key, &DeviceIdentity::none(), req.dry_run).is_ok());
        // A dry run authorizes nothing by itself: recording happens only after it succeeded.
        assert!(gate.authorize_apply(&key, &DeviceIdentity::none()).is_err());
    }

    #[test]
    fn apply_requires_a_matching_preview_and_is_single_use() {
        let gate = ApplyGate::new();
        let apply = |limit| DesktopPowerLimitRequest {
            pci_device_id: Some(0x73AF),
            limit_pct: limit,
            dry_run: false,
        };
        let key5 = prepare_power_limit(&apply(5)).unwrap();
        // Frontend says dry_run=false but nothing was previewed: refused.
        assert!(enforce_gate(&gate, &key5, &DeviceIdentity::none(), false)
            .unwrap_err()
            .contains("no matching dry-run"));

        gate.record_preview(&key5, &DeviceIdentity::none());
        // A preview for 5% does not authorize 10%.
        let key10 = prepare_power_limit(&apply(10)).unwrap();
        assert!(enforce_gate(&gate, &key10, &DeviceIdentity::none(), false).is_err());
        // The matching apply is allowed once.
        assert!(enforce_gate(&gate, &key5, &DeviceIdentity::none(), false).is_ok());
        assert!(enforce_gate(&gate, &key5, &DeviceIdentity::none(), false).is_err());
    }

    #[test]
    fn invalid_power_limit_and_reset_requests_never_reach_the_gate() {
        for limit in [101, -101, i32::MAX, i32::MIN] {
            let req = DesktopPowerLimitRequest {
                pci_device_id: None,
                limit_pct: limit,
                dry_run: false,
            };
            assert!(prepare_power_limit(&req).is_err(), "accepted {limit}");
        }
        let bad_id = DesktopPowerLimitRequest {
            pci_device_id: Some(0),
            limit_pct: 5,
            dry_run: false,
        };
        assert!(prepare_power_limit(&bad_id).is_err());
        let bad_reset = DesktopResetTuningRequest {
            pci_device_id: Some(0x1_0000),
            dry_run: false,
        };
        assert!(prepare_reset_tuning(&bad_reset).is_err());
    }

    #[test]
    fn lm_studio_patch_validates_inputs_and_binds_the_preview_to_exact_parameters() {
        let model = temp_gguf();
        let base = DesktopLmStudioPatchRequest {
            model_path: Some(model.display().to_string()),
            device_index: Some(0),
            profile_id: Some("vulkan_wddm_priority".to_string()),
            context_tokens: 4096,
            safety_margin_mib: 512,
            dry_run: false,
        };
        let ok = prepare_lm_studio_patch(&base).unwrap();

        // Invalid inputs are rejected before the gate or any write is considered.
        let bad = |f: &dyn Fn(&mut DesktopLmStudioPatchRequest)| {
            let mut r = base.clone();
            f(&mut r);
            prepare_lm_studio_patch(&r).is_err()
        };
        assert!(bad(&|r| r.profile_id = Some("../../evil".into())));
        assert!(bad(&|r| r.device_index = Some(10_000)));
        assert!(bad(
            &|r| r.model_path = Some("C:\\nope\\missing.gguf".into())
        ));
        assert!(bad(
            &|r| r.model_path = Some(model.with_extension("txt").display().to_string())
        ));

        // A preview binds to model, device, profile, context and margin.
        let gate = ApplyGate::new();
        gate.record_preview(&ok.key, &DeviceIdentity::none());
        let mut other = base.clone();
        other.context_tokens = 8192;
        let other = prepare_lm_studio_patch(&other).unwrap();
        assert_ne!(ok.key, other.key);
        assert!(enforce_gate(&gate, &other.key, &DeviceIdentity::none(), false).is_err());
        assert!(enforce_gate(&gate, &ok.key, &DeviceIdentity::none(), false).is_ok());
    }

    #[test]
    fn env_patch_rejects_unknown_kv_quant_instead_of_defaulting() {
        let model = temp_gguf();
        let mut req = DesktopEnvPatchRequest {
            model_path: Some(model.display().to_string()),
            device_index: None,
            context_tokens: 4096,
            kv_quant: Some("q5_1".to_string()),
            dry_run: false,
        };
        assert!(prepare_env_patch(&req)
            .unwrap_err()
            .contains("unknown KV cache type"));
        req.kv_quant = Some("q4_0".to_string());
        let q4 = prepare_env_patch(&req).unwrap();
        req.kv_quant = Some("f16".to_string());
        let f16 = prepare_env_patch(&req).unwrap();
        assert_ne!(
            q4.key, f16.key,
            "different KV types must not share a preview"
        );
    }

    #[test]
    fn rollbacks_are_gated_and_scoped() {
        let gate = ApplyGate::new();
        let all = prepare_lm_studio_rollback(&DesktopLmStudioRollbackRequest {
            model_path: None,
            dry_run: false,
        })
        .unwrap();
        assert!(enforce_gate(&gate, &all.1, &DeviceIdentity::none(), false).is_err());
        gate.record_preview(&all.1, &DeviceIdentity::none());
        assert!(enforce_gate(&gate, &all.1, &DeviceIdentity::none(), false).is_ok());

        let env_key = prepare_env_rollback();
        assert!(enforce_gate(&gate, &env_key, &DeviceIdentity::none(), false).is_err());
        // An LM Studio rollback preview never authorizes the registry rollback.
        gate.record_preview(&all.1, &DeviceIdentity::none());
        assert!(enforce_gate(&gate, &env_key, &DeviceIdentity::none(), false).is_err());

        assert!(prepare_lm_studio_rollback(&DesktopLmStudioRollbackRequest {
            model_path: Some("not-a-model.txt".into()),
            dry_run: false,
        })
        .is_err());
    }

    fn adapter(
        index: usize,
        name: &str,
        device: u32,
        bus: &str,
        discrete: bool,
    ) -> forge_core::GpuDeviceInfo {
        crate::apply_gate::test_support::gpu(index, name, 0x1002, device, Some(bus), discrete)
    }

    #[test]
    fn power_limit_and_reset_previews_are_bound_to_the_resolved_gpu() {
        let gpus = vec![
            adapter(0, "AMD Radeon RX 6900 XT", 0x73AF, "0000:03:00.0", true),
            adapter(1, "AMD Radeon RX 6800 XT", 0x73BF, "0000:06:00.0", true),
        ];
        let gate = ApplyGate::new();
        let power = |pci| DesktopPowerLimitRequest {
            pci_device_id: pci,
            limit_pct: 5,
            dry_run: false,
        };
        // Preview targets the 6900 XT.
        let key_a = prepare_power_limit(&power(Some(0x73AF))).unwrap();
        let dev_a = select_tuning_target(&gpus, Some(0x73AF)).unwrap();
        gate.record_preview(&key_a, &dev_a);

        // Same parameters, but the apply resolves a different PCI device: refused, and the
        // refusal comes from the identity binding, not from a key mismatch.
        let dev_b = select_tuning_target(&gpus, Some(0x73BF)).unwrap();
        let err = enforce_gate(&gate, &key_a, &dev_b, false).unwrap_err();
        assert!(err.contains("target GPU changed"), "{err}");

        // The reset command uses the same selector, so the same protection applies.
        let reset = DesktopResetTuningRequest {
            pci_device_id: Some(0x73AF),
            dry_run: false,
        };
        let reset_key = prepare_reset_tuning(&reset).unwrap();
        gate.record_preview(&reset_key, &dev_a);
        assert!(enforce_gate(&gate, &reset_key, &dev_b, false).is_err());
        gate.record_preview(&reset_key, &dev_a);
        assert!(enforce_gate(&gate, &reset_key, &dev_a, false).is_ok());
    }

    #[test]
    fn llm_config_previews_are_bound_to_the_resolved_gpu() {
        let model = temp_gguf();
        let igpu = adapter(0, "AMD Radeon(TM) Graphics", 0x164E, "0000:0e:00.0", false);
        let dgpu = adapter(1, "AMD Radeon RX 6900 XT", 0x73AF, "0000:03:00.0", true);
        let lm = DesktopLmStudioPatchRequest {
            model_path: Some(model.display().to_string()),
            device_index: Some(1),
            profile_id: None,
            context_tokens: 4096,
            safety_margin_mib: 512,
            dry_run: false,
        };
        let prepared = prepare_lm_studio_patch(&lm).unwrap();
        let env = DesktopEnvPatchRequest {
            model_path: Some(model.display().to_string()),
            device_index: Some(1),
            context_tokens: 4096,
            kv_quant: None,
            dry_run: false,
        };
        let env_prepared = prepare_env_patch(&env).unwrap();

        let before = vec![igpu.clone(), dgpu.clone()];
        let gate = ApplyGate::new();
        let at_preview = select_llm_target(&before, Some(1)).unwrap();
        gate.record_preview(&prepared.key, &at_preview);
        gate.record_preview(&env_prepared.key, &at_preview);

        // Unchanged system: both applies are authorized once.
        assert!(enforce_gate(&gate, &prepared.key, &at_preview, false).is_ok());
        assert!(enforce_gate(&gate, &prepared.key, &at_preview, false).is_err());

        // Index 1 now resolves to the integrated GPU: the environment patch is refused.
        let after = vec![dgpu, igpu];
        let at_apply = select_llm_target(&after, Some(1)).unwrap();
        assert!(enforce_gate(&gate, &env_prepared.key, &at_apply, false)
            .unwrap_err()
            .contains("target GPU changed"));

        // A removed device is an error before the gate is even consulted.
        assert!(select_llm_target(&after[..1], Some(1)).is_err());
    }

    #[test]
    fn rollbacks_use_an_explicit_no_device_identity() {
        // Rollbacks restore files / registry values and do not depend on a GPU, so their previews
        // carry the "none" identity and are never satisfied by (or satisfy) a GPU-bound preview.
        let gate = ApplyGate::new();
        let (_, key) = prepare_lm_studio_rollback(&DesktopLmStudioRollbackRequest {
            model_path: None,
            dry_run: false,
        })
        .unwrap();
        let gpu_identity = select_llm_target(
            &[adapter(
                0,
                "AMD Radeon RX 6900 XT",
                0x73AF,
                "0000:03:00.0",
                true,
            )],
            Some(0),
        )
        .unwrap();
        gate.record_preview(&key, &gpu_identity);
        assert!(enforce_gate(&gate, &key, &DeviceIdentity::none(), false).is_err());
        gate.record_preview(&key, &DeviceIdentity::none());
        assert!(enforce_gate(&gate, &key, &DeviceIdentity::none(), false).is_ok());
    }
}
