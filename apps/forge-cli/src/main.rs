use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use forge_benchmark::{
    apply_lm_studio_config, apply_windows_and_ollama_env_tweaks, autotune_llm_model,
    discover_local_llm_assets, generate_github_hardware_report, inspect_gguf_model,
    peak_process_rss_bytes, plan_llm_configuration_for_device, recommend_models_for_device,
    resolve_default_triton_script, rollback_lm_studio_config, rollback_windows_env_tweaks,
    run_llama_comparison, run_triton_suite, run_video_benchmark_suite,
    run_video_decode_benchmark_suite, run_vulkan_suite, EnvRollbackResult, KvCacheQuant,
    LlamaBenchConfig, LlmAutotuneConfig, LmStudioRollbackResult, ModelFitVerdict,
    TritonBenchConfig, VideoBenchSuiteConfig, VideoDecodeBenchSuiteConfig, VulkanBenchConfig,
    VulkanKernelSelection,
};
use forge_core::{
    builtin_profiles, match_profile_for_device, CapabilityResult, DiagnosticReport, GpuDeviceInfo,
    HardwareProfile, SupportStatus,
};
use forge_device::discover_all;
use forge_profiler::{
    check_competing_workload_warning, check_vram_pressure_warning, query_top_vram_processes,
    snapshot_all_gpus, GpuTelemetrySample, TelemetrySummary,
};
use forge_runtime::{
    default_untested_capabilities, probe_all_backends, probe_amf_capabilities,
    run_live_capability_checks, AmfAcceleration, AmfDecodeCodec, AmfEncodeCodec, AmfQualityPreset,
};
mod monitor_tune;

use monitor_tune::{
    cmd_monitor, cmd_tune_power_limit, cmd_tune_reset, cmd_tune_show, MonitorArgs, TuneSubcommand,
};
use serde::Serialize;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "forge",
    version,
    about = "RDNA Forge — AMD Radeon Performance & Capability Toolkit"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Inspect AMD GPUs, Windows drivers, compute backends, and runtime capabilities.
    Doctor(DoctorArgs),
    /// Run reproducible kernel and LLM inference benchmarks.
    Bench {
        #[command(subcommand)]
        target: BenchSubcommand,
    },
    /// Stream real-time ADLX GPU telemetry and WDDM VRAM alerts.
    Monitor(MonitorArgs),
    /// Inspect and safely adjust supported ADLX GPU tuning settings.
    Tune {
        #[command(subcommand)]
        action: TuneSubcommand,
    },
    /// Inspect hardware video encode/decode capabilities via AMD AMF.
    Video {
        #[command(subcommand)]
        action: VideoSubcommand,
    },
    /// Inspect GGUF models, calculate anti-WDDM-spill VRAM budgets, and autotune llama.cpp for AMD Radeon.
    Llm {
        #[command(subcommand)]
        action: LlmSubcommand,
    },
    /// Inspect and validate hardware target profiles.
    Profile {
        #[command(subcommand)]
        action: ProfileSubcommand,
    },
    /// Show Top-N Windows processes consuming dedicated and shared GPU VRAM.
    VramTop(VramTopArgs),
    /// Generate an anonymized Markdown hardware compatibility report for GitHub issues.
    Report(ReportArgs),
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// Run live GPU verification checks (Vulkan compute + HIP roundtrip) and embed results.
    #[arg(long)]
    verify: bool,
    /// Optional path to write the generated Markdown report.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Emit JSON payload (`issue_title`, `markdown`, `sanitized`) instead of raw Markdown.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct VramTopArgs {
    /// Maximum number of VRAM-consuming processes to display.
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// Emit machine-readable JSON instead of a human-readable table.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct DoctorArgs {
    /// Emit machine-readable JSON instead of human-readable text.
    #[arg(long)]
    json: bool,
    /// Execute live GPU verification kernels (Vulkan GEMM/RMSNorm/RoPE + HIP memory transfer).
    #[arg(long)]
    verify: bool,
    /// Optional Python executable path to probe for ROCm PyTorch and triton-windows.
    #[arg(long)]
    python: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum BenchSubcommand {
    /// Run Vulkan WGSL->SPIR-V compute benchmarks with statistical dispersion and correctness checks.
    Vulkan(VulkanBenchArgs),
    /// Compare llama.cpp Vulkan and HIP/ROCm backends on identical GGUF model and parameters.
    Llama(LlamaBenchArgs),
    /// Run PyTorch ROCm and Triton kernel benchmark suite via Python.
    Triton(TritonBenchArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KernelChoice {
    All,
    Gemm,
    GemmOpt,
    GemmI8,
    GemvW8,
    GemvQ4,
    #[value(name = "gemv-q4-k", alias = "gemv-q4k")]
    GemvQ4K,
    Rmsnorm,
    Rope,
    FusedRmsRope,
    FlashAttn,
}

impl From<KernelChoice> for VulkanKernelSelection {
    fn from(value: KernelChoice) -> Self {
        match value {
            KernelChoice::All => Self::All,
            KernelChoice::Gemm => Self::Gemm,
            KernelChoice::GemmOpt => Self::GemmOpt,
            KernelChoice::GemmI8 => Self::GemmI8,
            KernelChoice::GemvW8 => Self::GemvW8,
            KernelChoice::GemvQ4 => Self::GemvQ4,
            KernelChoice::GemvQ4K => Self::GemvQ4K,
            KernelChoice::Rmsnorm => Self::RmsNorm,
            KernelChoice::Rope => Self::Rope,
            KernelChoice::FusedRmsRope => Self::FusedRmsRope,
            KernelChoice::FlashAttn => Self::FlashAttn,
        }
    }
}

fn parse_bool_flag(s: &str) -> std::result::Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        other => Err(format!("expected 0/1 or true/false, got '{other}'")),
    }
}

#[derive(Debug, Args)]
struct VulkanBenchArgs {
    #[arg(long, value_enum, default_value_t = KernelChoice::All)]
    kernel: KernelChoice,
    #[arg(long)]
    size: Option<u32>,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 50)]
    iters: usize,
    #[arg(long, default_value_t = 50)]
    stability_runs: usize,
    /// Number of consecutive dispatches batched per command buffer submit for decode/elementwise kernels (1..=256, default 16).
    #[arg(long)]
    batch_dispatches: Option<usize>,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum VideoSubcommand {
    /// Inspect AMD AMF (amfrt64.dll) hardware video encode/decode codec capabilities.
    Caps {
        /// Emit machine-readable JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
    /// Benchmark AMD VCN hardware video encoders (H.264/AVC, HEVC, and AV1 on RDNA3/RDNA4).
    Bench(VideoBenchArgs),
    /// Benchmark AMD VCN hardware video decoders (H.264/AVC, HEVC 8-bit & 10-bit) using in-memory GOP streams.
    DecodeBench(VideoDecodeBenchArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum VideoCodecChoice {
    All,
    H264,
    Hevc,
    Hevc10,
    Av1,
    Av110,
}

impl VideoCodecChoice {
    fn to_codecs(self) -> Vec<AmfEncodeCodec> {
        match self {
            Self::All => vec![
                AmfEncodeCodec::H264Avc,
                AmfEncodeCodec::Hevc,
                AmfEncodeCodec::Hevc10Bit,
                AmfEncodeCodec::Av1,
            ],
            Self::H264 => vec![AmfEncodeCodec::H264Avc],
            Self::Hevc => vec![AmfEncodeCodec::Hevc],
            Self::Hevc10 => vec![AmfEncodeCodec::Hevc10Bit],
            Self::Av1 => vec![AmfEncodeCodec::Av1],
            Self::Av110 => vec![AmfEncodeCodec::Av110Bit],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum VideoDecodeCodecChoice {
    All,
    H264,
    Hevc,
    Hevc10,
}

impl VideoDecodeCodecChoice {
    fn to_codecs(self) -> Vec<AmfDecodeCodec> {
        match self {
            Self::All => vec![
                AmfDecodeCodec::H264Avc,
                AmfDecodeCodec::Hevc,
                AmfDecodeCodec::Hevc10Bit,
            ],
            Self::H264 => vec![AmfDecodeCodec::H264Avc],
            Self::Hevc => vec![AmfDecodeCodec::Hevc],
            Self::Hevc10 => vec![AmfDecodeCodec::Hevc10Bit],
        }
    }
}

#[derive(Debug, Args)]
struct VideoDecodeBenchArgs {
    /// Target hardware decoder codec(s) to benchmark.
    #[arg(long, value_enum, default_value_t = VideoDecodeCodecChoice::All)]
    codec: VideoDecodeCodecChoice,
    /// Frame width in pixels (even integer in 128..=7680).
    #[arg(long, default_value_t = 1920)]
    width: u32,
    /// Frame height in pixels (even integer in 128..=4320).
    #[arg(long, default_value_t = 1080)]
    height: u32,
    /// Nominal stream frame rate in FPS (1..=240).
    #[arg(long, default_value_t = 60)]
    fps: u32,
    /// Source GOP encode bitrate in megabits per second (1..=500).
    #[arg(long, default_value_t = 12)]
    bitrate_mbps: u32,
    /// Number of untimed warmup frames per codec.
    #[arg(long, default_value_t = 15)]
    warmup: u32,
    /// Number of timed frames per codec.
    #[arg(long, default_value_t = 120)]
    frames: u32,
    /// Emit machine-readable JSON instead of human-readable text.
    #[arg(long)]
    json: bool,
    /// Optional path to write JSON decode benchmark report.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum VideoPresetChoice {
    Speed,
    Balanced,
    Quality,
}

impl From<VideoPresetChoice> for AmfQualityPreset {
    fn from(value: VideoPresetChoice) -> Self {
        match value {
            VideoPresetChoice::Speed => Self::Speed,
            VideoPresetChoice::Balanced => Self::Balanced,
            VideoPresetChoice::Quality => Self::Quality,
        }
    }
}

#[derive(Debug, Args)]
struct VideoBenchArgs {
    /// Target hardware encoder codec(s) to benchmark.
    #[arg(long, value_enum, default_value_t = VideoCodecChoice::All)]
    codec: VideoCodecChoice,
    /// Frame width in pixels (even integer in 128..=7680).
    #[arg(long, default_value_t = 1920)]
    width: u32,
    /// Frame height in pixels (even integer in 128..=4320).
    #[arg(long, default_value_t = 1080)]
    height: u32,
    /// Nominal target stream frame rate in FPS (1..=240).
    #[arg(long, default_value_t = 60)]
    fps: u32,
    /// Target bitrate in megabits per second (1..=500).
    #[arg(long, default_value_t = 12)]
    bitrate_mbps: u32,
    /// Hardware encoder quality/speed preset.
    #[arg(long, value_enum, default_value_t = VideoPresetChoice::Speed)]
    preset: VideoPresetChoice,
    /// Number of untimed warmup frames per codec.
    #[arg(long, default_value_t = 15)]
    warmup: u32,
    /// Number of timed frames per codec.
    #[arg(long, default_value_t = 120)]
    frames: u32,
    /// Emit machine-readable JSON instead of human-readable text.
    #[arg(long)]
    json: bool,
    /// Optional path to write JSON benchmark report.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct LlamaBenchArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    vulkan_bin: Option<PathBuf>,
    #[arg(long)]
    hip_bin: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    prompt_tokens: u32,
    #[arg(long, default_value_t = 128)]
    gen_tokens: u32,
    #[arg(long, default_value_t = 5)]
    reps: u32,
    #[arg(long, default_value_t = 99)]
    ngl: u32,
    #[arg(long, default_value_t = 0)]
    vulkan_mg: i32,
    #[arg(long, default_value_t = 0)]
    hip_mg: i32,
    #[arg(long)]
    vulkan_device: Option<String>,
    #[arg(long)]
    hip_device: Option<String>,
    #[arg(long, value_parser = parse_bool_flag)]
    flash_attn: Option<bool>,
    #[arg(long)]
    hip_disable_graphs: bool,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct TritonBenchArgs {
    #[arg(long, default_value = "python")]
    python: PathBuf,
    #[arg(long)]
    script: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 50)]
    iters: usize,
    /// Run lightweight 256x256 accuracy verification without heavy full-size GPU sweeps.
    #[arg(long)]
    smoke: bool,
    /// Opt-in pre/post ADLX snapshot around Python ROCm/Triton execution.
    #[arg(long)]
    telemetry: bool,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum ProfileSubcommand {
    /// Load, validate, and display a hardware profile TOML file.
    Show {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// List all built-in AMD Radeon hardware profiles bundled with RDNA Forge.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Auto-match a discovered GPU adapter to its closest built-in AMD Radeon profile.
    Match {
        /// Discovered GPU device index to match (0 = preferred GPU).
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum LlmSubcommand {
    /// Discover local GGUF models and llama.cpp Vulkan/ROCm backends on this PC.
    Discover {
        #[arg(long)]
        json: bool,
    },
    /// Inspect a GGUF v2/v3 model header (architecture, layers, attention/SSM hybrid structure).
    Inspect {
        /// Path to the .gguf model file (or omit to inspect the first discovered local GGUF model).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Calculate anti-WDDM-spill VRAM budgets, KV cache quantization savings, and AMD Radeon tweaks.
    Plan(LlmPlanArgs),
    /// Run empirical llama-bench autotuning across Vulkan/ROCm AMD Radeon tweak profiles.
    Autotune(LlmAutotuneArgs),
    /// Preview (dry-run default) or patch LM Studio per-model config and backend preferences (with .bak backup).
    ApplyLmstudio(LlmApplyLmstudioArgs),
    /// Preview (dry-run default) or apply Windows/Ollama user environment variables (HKCU\Environment) & Modelfile.
    ApplyEnv(LlmApplyEnvArgs),
    /// Preview (dry-run default) or restore LM Studio .bak backups and reset RDNA Forge HKCU env vars.
    Rollback(LlmRollbackArgs),
    /// Recommend popular local LLM models & quantizations that fit on the selected GPU's VRAM.
    Recommend(LlmRecommendArgs),
}

#[derive(Debug, Args)]
struct LlmRecommendArgs {
    /// Target GPU device index (default: primary discrete GPU).
    #[arg(long)]
    device: Option<usize>,
    /// Target context window in tokens.
    #[arg(long, default_value_t = 4096)]
    context: u32,
    /// Safety VRAM margin in MiB reserved against Windows WDDM background spikes.
    #[arg(long, default_value_t = 512)]
    safety_margin_mib: u64,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct LlmRollbackArgs {
    /// Optional GGUF model path to restore only its specific LM Studio config backup (default: all backups).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Roll back LM Studio JSON configurations from .json.bak backups.
    #[arg(long)]
    lmstudio: bool,
    /// Roll back RDNA Forge-managed keys from HKCU\Environment.
    #[arg(long)]
    env: bool,
    /// Actually restore backup files and delete HKCU env keys. Defaults to dry-run preview.
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct LlmApplyLmstudioArgs {
    /// Path to the .gguf model file (or omit to use the first discovered local GGUF model).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Target GPU device index (default: primary discrete GPU).
    #[arg(long)]
    device: Option<usize>,
    /// Profile ID to apply (e.g. vulkan_wddm_priority, rocm_wmma_hipblaslt, rocm_q4_kv_long_ctx, vulkan_q4_kv_decode, moe_expert_offload, vulkan_strict_zero_spill).
    #[arg(long)]
    profile: Option<String>,
    /// Target context window in tokens.
    #[arg(long, default_value_t = 4096)]
    context: u32,
    /// Safety VRAM margin in MiB.
    #[arg(long, default_value_t = 512)]
    safety_margin_mib: u64,
    /// Actually write changes to LM Studio JSON configs (creates .bak backup first). Defaults to dry-run preview.
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum KvQuantChoice {
    F16,
    Q8_0,
    Q4_0,
}

impl From<KvQuantChoice> for KvCacheQuant {
    fn from(value: KvQuantChoice) -> Self {
        match value {
            KvQuantChoice::F16 => Self::F16,
            KvQuantChoice::Q8_0 => Self::Q8_0,
            KvQuantChoice::Q4_0 => Self::Q4_0,
        }
    }
}

#[derive(Debug, Args)]
struct LlmApplyEnvArgs {
    /// Path to the .gguf model file (or omit to use the first discovered local GGUF model).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Target GPU device index (default: primary discrete GPU).
    #[arg(long)]
    device: Option<usize>,
    /// Target context window in tokens for the generated Ollama Modelfile.
    #[arg(long, default_value_t = 4096)]
    context: u32,
    /// KV cache quantization type for OLLAMA_KV_CACHE_TYPE.
    #[arg(long, value_enum, default_value_t = KvQuantChoice::Q8_0)]
    kv_quant: KvQuantChoice,
    /// Optional path to write the generated Ollama Modelfile.
    #[arg(long)]
    modelfile_out: Option<PathBuf>,
    /// Actually commit environment variables to HKCU\Environment and write Modelfile. Defaults to dry-run preview.
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct LlmPlanArgs {
    /// Path to the .gguf model file (or omit to use the first discovered local GGUF model).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Target GPU device index (default: primary discrete GPU).
    #[arg(long)]
    device: Option<usize>,
    /// Target context window in tokens.
    #[arg(long, default_value_t = 4096)]
    context: u32,
    /// Safety VRAM margin in MiB reserved against Windows WDDM background spikes.
    #[arg(long, default_value_t = 512)]
    safety_margin_mib: u64,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct LlmAutotuneArgs {
    /// Path to the .gguf model file (or omit to use the first discovered local GGUF model).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Target GPU device index (default: primary discrete GPU).
    #[arg(long)]
    device: Option<usize>,
    /// Optional path to Vulkan llama-bench.exe.
    #[arg(long)]
    vulkan_bin: Option<PathBuf>,
    /// Optional path to ROCm/HIP llama-bench.exe.
    #[arg(long)]
    hip_bin: Option<PathBuf>,
    /// Optional path to ROCm runtime DLL directory containing hipblas.dll / rocblas.dll.
    #[arg(long)]
    rocm_lib_dir: Option<PathBuf>,
    /// Prompt tokens per probe pass.
    #[arg(long, default_value_t = 128)]
    prompt_tokens: u32,
    /// Generated tokens per probe pass.
    #[arg(long, default_value_t = 32)]
    gen_tokens: u32,
    /// Pre-filled KV cache context depth in tokens (-d) to test WDDM spill & attention scaling.
    #[arg(long, default_value_t = 2048)]
    context_depth: u32,
    /// Repetitions per candidate profile.
    #[arg(long, default_value_t = 2)]
    reps: u32,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor(args) => cmd_doctor(&args),
        Commands::Bench { target } => match target {
            BenchSubcommand::Vulkan(args) => cmd_bench_vulkan(&args),
            BenchSubcommand::Llama(args) => cmd_bench_llama(&args),
            BenchSubcommand::Triton(args) => cmd_bench_triton(&args),
        },
        Commands::Monitor(args) => cmd_monitor(&args),
        Commands::Tune { action } => match action {
            TuneSubcommand::Show { device_id, json } => cmd_tune_show(device_id.as_deref(), json),
            TuneSubcommand::PowerLimit {
                limit_pct,
                device_id,
                apply,
                json,
            } => cmd_tune_power_limit(limit_pct, device_id.as_deref(), apply, json),
            TuneSubcommand::Reset {
                device_id,
                apply,
                json,
            } => cmd_tune_reset(device_id.as_deref(), apply, json),
        },
        Commands::Video { action } => match action {
            VideoSubcommand::Caps { json } => cmd_video_caps(json),
            VideoSubcommand::Bench(args) => cmd_video_bench(&args),
            VideoSubcommand::DecodeBench(args) => cmd_video_decode_bench(&args),
        },
        Commands::Llm { action } => match action {
            LlmSubcommand::Discover { json } => cmd_llm_discover(json),
            LlmSubcommand::Inspect { model, json } => cmd_llm_inspect(model.as_deref(), json),
            LlmSubcommand::Plan(args) => cmd_llm_plan(&args),
            LlmSubcommand::Autotune(args) => cmd_llm_autotune(&args),
            LlmSubcommand::ApplyLmstudio(args) => cmd_llm_apply_lmstudio(&args),
            LlmSubcommand::ApplyEnv(args) => cmd_llm_apply_env(&args),
            LlmSubcommand::Rollback(args) => cmd_llm_rollback(&args),
            LlmSubcommand::Recommend(args) => cmd_llm_recommend(&args),
        },
        Commands::Profile { action } => match action {
            ProfileSubcommand::Show { path, json } => cmd_profile_show(&path, json),
            ProfileSubcommand::List { json } => cmd_profile_list(json),
            ProfileSubcommand::Match { device, json } => cmd_profile_match(device, json),
        },
        Commands::VramTop(args) => cmd_vram_top(&args),
        Commands::Report(args) => cmd_report(&args),
    }
}

#[derive(Serialize)]
struct DoctorCliReport<'a> {
    #[serde(flatten)]
    report: &'a DiagnosticReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    telemetry: Option<&'a [GpuTelemetrySample]>,
}

fn cmd_doctor(args: &DoctorArgs) -> Result<()> {
    let snapshot = discover_all();
    let (backends, missing_dependencies) = probe_all_backends(&snapshot, args.python.as_deref());
    let telemetry = snapshot_all_gpus().ok();

    let mut capabilities = vec![CapabilityResult {
        name: "Device discovery".to_string(),
        status: if snapshot.gpus.is_empty() {
            SupportStatus::Unsupported
        } else {
            SupportStatus::Verified
        },
        detail: Some(format!("{} GPU(s) enumerated", snapshot.gpus.len())),
    }];

    if args.verify {
        capabilities.extend(run_live_capability_checks(&snapshot.gpus));
    } else {
        capabilities.extend(default_untested_capabilities());
    }

    let report = DiagnosticReport {
        system: snapshot.system,
        gpus: snapshot.gpus,
        backends,
        capabilities,
        missing_dependencies,
        windows_audit: Some(snapshot.windows_audit),
    };

    if args.json {
        let cli_report = DoctorCliReport {
            report: &report,
            telemetry: telemetry.as_deref(),
        };
        println!("{}", serde_json::to_string_pretty(&cli_report)?);
    } else {
        print_doctor_report(&report, telemetry.as_deref());
    }

    Ok(())
}

fn print_doctor_report(report: &DiagnosticReport, telemetry: Option<&[GpuTelemetrySample]>) {
    println!("RDNA FORGE DIAGNOSTICS");
    println!();

    println!("GPU");
    if let Some(primary) = report.gpus.first() {
        print_gpu_block(primary, false);
        for secondary in report.gpus.iter().skip(1) {
            println!();
            println!("SECONDARY GPU [{}]", secondary.index);
            print_gpu_block(secondary, true);
        }
    } else {
        println!("  No GPU discovered");
    }

    println!();
    println!("SYSTEM");
    println!(
        "  OS: {} ({}, Build {})",
        report.system.os_name, report.system.os_version, report.system.os_build
    );
    println!(
        "  CPU: {} ({} logical cores)",
        report.system.cpu_name, report.system.logical_cores
    );
    println!("  RAM: {:.2} GiB", gib(report.system.total_ram_bytes));
    if let Some(primary) = report.gpus.first() {
        let driver = match (&primary.adrenalin_version, &primary.driver_version) {
            (Some(ad), Some(drv)) => format!("Adrenalin {ad} ({drv})"),
            (None, Some(drv)) => drv.clone(),
            (Some(ad), None) => format!("Adrenalin {ad}"),
            (None, None) => "Not detected".to_string(),
        };
        println!("  Driver: {driver}");
    }

    println!();
    println!("BACKENDS");
    for b in &report.backends {
        println!("  {}: [{}] {}", b.kind, b.status, b.summary);
        for detail in &b.details {
            println!("    - {detail}");
        }
    }

    println!();
    println!("TELEMETRY (ADLX)");
    match telemetry {
        Some(samples) if !samples.is_empty() => {
            for s in samples {
                let role = if s.discrete { "discrete" } else { "integrated" };
                let dev_hex = s
                    .pci_device_id
                    .map(|id| format!(" (0x{id:04X})"))
                    .unwrap_or_default();
                let clk = fmt_opt(s.gpu_clock_mhz, "MHz");
                let vclk = fmt_opt(s.vram_clock_mhz, "MHz");
                let temp = fmt_opt_f64(s.gpu_temp_c, "C");
                let hotspot = fmt_opt_f64(s.hotspot_temp_c, "C");
                let pwr = fmt_opt_f64(s.gpu_power_w, "W");
                let fan = fmt_opt(s.fan_rpm, "RPM");
                let volt = fmt_opt(s.voltage_mv, "mV");
                let vram_str = match (s.vram_used_mib, s.total_vram_mib) {
                    (Some(u), Some(t)) if t > 0 => {
                        let pct = (f64::from(u) / f64::from(t)) * 100.0;
                        format!("{u}/{t} MiB ({pct:.1}%)")
                    }
                    (Some(u), _) => format!("{u} MiB"),
                    _ => "n/a".to_string(),
                };
                println!("  {}{dev_hex} [{role}]:", s.gpu_name);
                println!(
                    "    Clock={clk} | VRAMClk={vclk} | Temp={temp} | Hotspot={hotspot} | Power={pwr} | Fan={fan} | Volt={volt} | VRAM={vram_str}"
                );
                if let (Some(u), Some(t)) = (s.vram_used_mib, s.total_vram_mib) {
                    if let Some(warn) = check_vram_pressure_warning(f64::from(u), t) {
                        println!("    WARNING: {warn}");
                    }
                }
                if let Some(warn) = check_competing_workload_warning(s) {
                    println!("    WARNING: {warn}");
                }
            }
        }
        _ => println!("  unavailable"),
    }

    println!();
    println!("CAPABILITIES");
    for cap in &report.capabilities {
        match &cap.detail {
            Some(detail) => println!("  {}: {} ({detail})", cap.name, cap.status),
            None => println!("  {}: {}", cap.name, cap.status),
        }
    }

    if let Some(ref audit) = report.windows_audit {
        println!();
        println!("WINDOWS GPU HEALTH AUDIT");
        let hags = match audit.hags_enabled {
            Some(true) => format!("ENABLED (HwSchMode={})", audit.hags_mode.unwrap_or(2)),
            Some(false) => format!("DISABLED (HwSchMode={})", audit.hags_mode.unwrap_or(1)),
            None => "Windows default (unset / disabled on RDNA2)".to_string(),
        };
        let tdr = match audit.tdr_delay_secs {
            Some(sec) => format!("{sec}s (custom registry override)"),
            None => format!(
                "{}s (Windows default; unset in HKLM)",
                audit.effective_tdr_delay_secs
            ),
        };
        let lp = match audit.long_paths_enabled {
            Some(true) => "ENABLED (LongPathsEnabled=1)",
            _ => "DISABLED (260-char MAX_PATH)",
        };
        println!("  HAGS (HwSchMode)   : {hags}");
        println!("  WDDM TdrDelay      : {tdr}");
        println!("  LongPathsEnabled   : {lp}");
        if audit.advisories.is_empty() {
            println!(
                "  Status             : All audited Windows GPU & filesystem settings healthy"
            );
        } else {
            for adv in &audit.advisories {
                println!("  ADVISORY           : {adv}");
            }
        }
    }

    if !report.missing_dependencies.is_empty() {
        println!();
        println!("NOTES / MISSING DEPENDENCIES");
        for note in &report.missing_dependencies {
            println!("  - {note}");
        }
    }
}

fn print_gpu_block(gpu: &GpuDeviceInfo, compact: bool) {
    println!("  {}", gpu.name);
    println!(
        "  Architecture: {} ({})",
        gpu.architecture,
        gpu.architecture.vcn_generation()
    );
    println!(
        "  Target: {}",
        gpu.gfx_target
            .as_ref()
            .map(|t| t.base.as_str())
            .unwrap_or("unknown")
    );
    if let Some(matched) = match_profile_for_device(gpu) {
        println!(
            "  Matched Profile: {} (id={})",
            matched.profile.device.name, matched.id
        );
    }
    println!(
        "  VRAM: {:.2} GiB ({} bytes)",
        gib(gpu.vram_bytes),
        gpu.vram_bytes
    );
    if !compact {
        println!(
            "  PCI: 0x{:04X}:0x{:04X}{}",
            gpu.vendor_id,
            gpu.device_id,
            gpu.pci_bus_id
                .as_deref()
                .map(|p| format!(" ({p})"))
                .unwrap_or_default()
        );
        if let (Some(cus), Some(wgps)) = (gpu.compute_units, gpu.wgp_count) {
            println!("  Compute Units: {cus} CUs ({wgps} WGPs)");
        }
        if let Some(rebar) = gpu.rebar_enabled {
            println!(
                "  Resizable BAR: {}",
                if rebar {
                    "ENABLED"
                } else {
                    "DISABLED (256 MiB BAR)"
                }
            );
        }
    }
}

fn cmd_bench_vulkan(args: &VulkanBenchArgs) -> Result<()> {
    let config = VulkanBenchConfig {
        device_index: args.device,
        kernel: args.kernel.into(),
        size_override: args.size,
        warmup_iters: args.warmup,
        measured_iters: args.iters,
        stability_runs: args.stability_runs,
        batch_dispatches: args.batch_dispatches,
    };

    let report = run_vulkan_suite(&config).context("Vulkan benchmark suite failed")?;
    let json = serde_json::to_string_pretty(&report)?;
    write_optional_output(args.output.as_deref(), &json)?;

    if args.json {
        println!("{json}");
    } else {
        println!("RDNA FORGE VULKAN BENCHMARK");
        println!(
            "Device: {} ({}, {})",
            report.device.name,
            report.device.architecture,
            report
                .device
                .gfx_target
                .as_ref()
                .map(|t| t.base.as_str())
                .unwrap_or("unknown")
        );
        println!(
            "Driver: {} | Vulkan: {}",
            report
                .device
                .adrenalin_version
                .as_deref()
                .unwrap_or("unknown"),
            report
                .device
                .vulkan_api_version
                .as_deref()
                .unwrap_or("unknown")
        );
        println!();
        for b in &report.benchmarks {
            let acc_str = b
                .accuracy
                .as_ref()
                .map(|a| {
                    format!(
                        "pass={} max_abs={:.2e} rmse={:.2e}",
                        a.passed, a.max_abs_err, a.rmse
                    )
                })
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  {:<24} {:<12} median={:.3} ms (min={:.3}, p95={:.3}, IQR={:.3}, MAD={:.3}) | {:.2} {} | {}",
                b.name,
                b.shape,
                b.latency.median_ms,
                b.latency.min_ms,
                b.latency.p95_ms,
                b.latency.iqr_ms,
                b.latency.mad_ms,
                b.throughput_median,
                b.throughput_unit,
                acc_str
            );
        }
        if let Some(ref st) = report.stability {
            println!();
            println!(
                "  Stability ({}): {}/{} passed, all_finite={}, all_correct={}, median={:.3} ms",
                st.workload,
                st.iterations - st.failures,
                st.iterations,
                st.all_finite,
                st.all_correct,
                st.latency.median_ms
            );
        }
        if let Some(rss) = report.peak_rss_bytes.or_else(peak_process_rss_bytes) {
            println!(
                "  Peak Process RSS: {:.2} MiB",
                rss as f64 / (1024.0 * 1024.0)
            );
        }
        if let Some(ref tel) = report.telemetry {
            println!();
            print_telemetry_summary("ADLX Telemetry (Vulkan suite)", tel);
        }
    }

    Ok(())
}

fn cmd_bench_llama(args: &LlamaBenchArgs) -> Result<()> {
    let config = LlamaBenchConfig {
        model_path: args.model.clone(),
        vulkan_bin: args.vulkan_bin.clone(),
        hip_bin: args.hip_bin.clone(),
        prompt_tokens: args.prompt_tokens,
        gen_tokens: args.gen_tokens,
        reps: args.reps,
        n_gpu_layers: args.ngl,
        vulkan_main_gpu: args.vulkan_mg,
        hip_main_gpu: args.hip_mg,
        vulkan_device: args.vulkan_device.clone(),
        hip_device: args.hip_device.clone(),
        flash_attn: args.flash_attn,
        hip_disable_graphs: args.hip_disable_graphs,
    };

    let report = run_llama_comparison(&config).context("llama-bench comparison failed")?;
    let json = serde_json::to_string_pretty(&report)?;
    write_optional_output(args.output.as_deref(), &json)?;

    if args.json {
        println!("{json}");
    } else {
        println!("RDNA FORGE LLAMA.CPP BENCHMARK COMPARISON");
        println!(
            "Model: {} | pp={} tg={} reps={} ngl={} fa={} hip_graphs={}",
            report.model_path,
            report.prompt_tokens,
            report.gen_tokens,
            report.reps,
            report.n_gpu_layers,
            args.flash_attn
                .map(|v| if v { "on" } else { "off" })
                .unwrap_or("default"),
            if args.hip_disable_graphs {
                "disabled"
            } else {
                "default"
            }
        );
        for run in &report.runs {
            println!();
            println!("Backend: {} ({})", run.backend, run.binary_path);
            if !run.succeeded {
                println!(
                    "  FAILED: {}",
                    run.error.as_deref().unwrap_or("unknown error")
                );
                if let Some(ref tel) = run.telemetry {
                    print_telemetry_summary(&format!("ADLX Telemetry ({})", run.backend), tel);
                }
                continue;
            }
            for e in &run.entries {
                println!(
                    "  {:<8} gpu='{}' backends='{}' | median={:.2} tok/s (avg={:.2}, stddev={:.2}, median_ms={:.2})",
                    e.test_kind,
                    e.gpu_info,
                    e.backends,
                    e.median_tok_s,
                    e.avg_tok_s,
                    e.stddev_tok_s,
                    e.latency.median_ms
                );
            }
            if let Some(ref tel) = run.telemetry {
                print_telemetry_summary(&format!("ADLX Telemetry ({})", run.backend), tel);
            }
        }
    }

    Ok(())
}

fn cmd_bench_triton(args: &TritonBenchArgs) -> Result<()> {
    let script_path = resolve_default_triton_script(args.script.as_deref())
        .context("could not locate modules/ai/triton/bench_suite.py; pass --script")?;

    let config = TritonBenchConfig {
        python_exe: args.python.clone(),
        script_path,
        warmup_iters: args.warmup,
        measured_iters: args.iters,
        smoke: args.smoke,
        collect_telemetry: args.telemetry,
    };

    let report = run_triton_suite(&config).context("Triton suite failed")?;
    let json = serde_json::to_string_pretty(&report)?;
    write_optional_output(args.output.as_deref(), &json)?;

    if args.json {
        println!("{json}");
    } else {
        println!("RDNA FORGE PYTORCH/TRITON BENCHMARK");
        println!(
            "PyTorch={} HIP={} Triton={} GPU={} ({})",
            report.provenance.torch_version.as_deref().unwrap_or("none"),
            report
                .provenance
                .torch_hip_version
                .as_deref()
                .unwrap_or("none"),
            report
                .provenance
                .triton_windows_version
                .as_deref()
                .or(report.provenance.triton_version.as_deref())
                .unwrap_or("none"),
            report.provenance.device_name.as_deref().unwrap_or("none"),
            report
                .provenance
                .gcn_arch_name
                .as_deref()
                .unwrap_or("unknown")
        );
        for b in &report.benchmarks {
            if b.status != "verified" {
                println!(
                    "  {:<32} {:<16} [{}] {}",
                    b.name,
                    b.shape,
                    b.status.to_uppercase(),
                    b.error.as_deref().unwrap_or("")
                );
                continue;
            }
            if let Some(ref lat) = b.latency {
                println!(
                    "  {:<32} {:<16} median={:.3} ms | {:.2} {} | pass={}",
                    b.name,
                    b.shape,
                    lat.median_ms,
                    b.throughput_median.unwrap_or(0.0),
                    b.throughput_unit.as_deref().unwrap_or(""),
                    b.accuracy.as_ref().map(|a| a.passed).unwrap_or(false)
                );
            }
        }
        if let Some(ref tel) = report.telemetry {
            println!();
            print_telemetry_summary("ADLX Telemetry (Triton suite)", tel);
        }
    }

    Ok(())
}

fn fmt_opt<T: std::fmt::Display>(val: Option<T>, unit: &str) -> String {
    val.map(|v| format!("{v} {unit}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_opt_f64(val: Option<f64>, unit: &str) -> String {
    val.map(|v| format!("{v:.1} {unit}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_telemetry_summary(label: &str, t: &TelemetrySummary) {
    let mut stdout = std::io::stdout().lock();
    let _ = print_telemetry_summary_to(&mut stdout, label, t);
}

pub(crate) fn print_telemetry_summary_to(
    w: &mut dyn std::io::Write,
    label: &str,
    t: &TelemetrySummary,
) -> std::io::Result<()> {
    writeln!(
        w,
        "  {label}: {} ({} samples @ {} ms)",
        t.gpu_name, t.sample_count, t.interval_ms
    )?;
    let mut parts = Vec::new();
    if let Some(ref m) = t.gpu_clock_mhz {
        parts.push(format!(
            "clk={:.0} ({:.0}-{:.0}) MHz",
            m.median, m.min, m.max
        ));
    }
    if let Some(ref m) = t.vram_clock_mhz {
        parts.push(format!("vram_clk={:.0} MHz", m.median));
    }
    if let Some(ref m) = t.gpu_temp_c {
        parts.push(format!(
            "temp={:.1} ({:.1}-{:.1}) C",
            m.median, m.min, m.max
        ));
    }
    if let Some(ref m) = t.hotspot_temp_c {
        parts.push(format!("hotspot={:.1} (max {:.1}) C", m.median, m.max));
    }
    if let Some(ref m) = t.gpu_power_w {
        parts.push(format!("pwr={:.1} (max {:.1}) W", m.median, m.max));
    }
    if let Some(ref m) = t.fan_rpm {
        parts.push(format!("fan={:.0} RPM", m.median));
    }
    if let Some(ref m) = t.vram_used_mib {
        let tot = t
            .total_vram_mib
            .map(|tot| format!("/{tot}"))
            .unwrap_or_default();
        parts.push(format!("vram={:.0}-{:.0}{tot} MiB", m.min, m.max));
    }
    if !parts.is_empty() {
        writeln!(w, "    {}", parts.join(" | "))?;
    }
    for warn in &t.warnings {
        writeln!(w, "    WARNING: {warn}")?;
    }
    Ok(())
}

fn cmd_video_caps(json: bool) -> Result<()> {
    let report =
        probe_amf_capabilities().context("failed to probe AMD AMF hardware video capabilities")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("RDNA FORGE AMF VIDEO CAPABILITIES");
    println!(
        "  AMF Runtime: {} | DX11 Context: {}",
        report.runtime_version,
        if report.dx11_initialized {
            "ready"
        } else {
            "not initialized"
        }
    );
    println!();
    println!("  Codec        Dir     Accel         Resolution Range      10bit  Interlaced Native Formats");
    for c in &report.codecs {
        let res = match (c.min_width, c.min_height, c.max_width, c.max_height) {
            (Some(w0), Some(h0), Some(w1), Some(h1)) => format!("{w0}x{h0}..{w1}x{h1}"),
            _ => "n/a".to_string(),
        };
        let ten_bit = if c
            .supported_formats
            .iter()
            .any(|f| matches!(f.as_str(), "P010" | "P012" | "P016" | "Y210" | "Y410"))
        {
            "yes"
        } else {
            "no"
        };
        let interlaced = c
            .interlaced_supported
            .map(|b| if b { "yes" } else { "no" })
            .unwrap_or("n/a");
        let extra = if c.acceleration == AmfAcceleration::NotSupported {
            c.detail.as_deref().unwrap_or("unsupported").to_string()
        } else if c.native_formats.is_empty() {
            "none".to_string()
        } else {
            c.native_formats.join(", ")
        };
        println!(
            "  {:<12} {:<7} {:<13} {:<21} {:<6} {:<10} {}",
            c.codec, c.direction, c.acceleration, res, ten_bit, interlaced, extra
        );
    }

    Ok(())
}

fn cmd_video_bench(args: &VideoBenchArgs) -> Result<()> {
    let suite_cfg = VideoBenchSuiteConfig {
        codecs: args.codec.to_codecs(),
        width: args.width,
        height: args.height,
        fps: args.fps,
        target_bitrate_bps: u64::from(args.bitrate_mbps).saturating_mul(1_000_000),
        preset: args.preset.into(),
        warmup_frames: args.warmup,
        measured_frames: args.frames,
        skip_unsupported: matches!(args.codec, VideoCodecChoice::All),
    };

    let report = run_video_benchmark_suite(&suite_cfg)
        .context("failed to run AMD AMF VCN hardware video encode benchmark")?;
    let json_text = serde_json::to_string_pretty(&report)?;
    write_optional_output(args.output.as_deref(), &json_text)?;

    if args.json {
        println!("{json_text}");
        return Ok(());
    }

    println!("RDNA FORGE AMF VCN HARDWARE ENCODE BENCHMARK");
    println!(
        "  AMF Runtime: {} | Resolution: {}x{} @ {} FPS | Preset: {} | Target: {} Mbps | Warmup/Frames: {}/{}",
        report.runtime_version,
        args.width,
        args.height,
        args.fps,
        suite_cfg.preset,
        args.bitrate_mbps,
        args.warmup,
        args.frames
    );
    println!();
    println!(
        "  {:<18} {:<6} {:<10} {:<10} {:<10} {:<11} {:<9} {:<10}",
        "Codec", "Format", "Encode FPS", "MPix/s", "Avg ms/f", "Eff. Mbps", "Packets", "Bitstream"
    );
    for r in &report.results {
        let valid = r.valid_bitstream || r.valid_annex_b;
        println!(
            "  {:<18} {:<6} {:<10.2} {:<10.2} {:<10.3} {:<11.2} {:<9} {:<10}",
            r.codec,
            r.surface_format,
            r.encode_fps,
            r.megapixels_per_sec,
            r.avg_frame_ms,
            r.effective_bitrate_mbps,
            format!("{}/{}", r.output_packets, r.submitted_frames),
            if valid {
                if r.codec.starts_with("AV1") {
                    "VALID OBU"
                } else {
                    "VALID"
                }
            } else {
                "INVALID"
            }
        );
    }
    if !report.skipped_codecs.is_empty() {
        println!();
        for skipped in &report.skipped_codecs {
            println!("  Skipped: {skipped}");
        }
    }

    if let Some(ref t) = report.telemetry {
        println!();
        print_telemetry_summary("VCN Telemetry", t);
    }

    Ok(())
}

fn cmd_video_decode_bench(args: &VideoDecodeBenchArgs) -> Result<()> {
    let suite_cfg = VideoDecodeBenchSuiteConfig {
        codecs: args.codec.to_codecs(),
        width: args.width,
        height: args.height,
        fps: args.fps,
        source_bitrate_bps: u64::from(args.bitrate_mbps).saturating_mul(1_000_000),
        warmup_frames: args.warmup,
        measured_frames: args.frames,
    };

    let report = run_video_decode_benchmark_suite(&suite_cfg)
        .context("failed to run AMD AMF VCN hardware video decode benchmark")?;
    let json_text = serde_json::to_string_pretty(&report)?;
    write_optional_output(args.output.as_deref(), &json_text)?;

    if args.json {
        println!("{json_text}");
        return Ok(());
    }

    println!("RDNA FORGE AMF VCN HARDWARE DECODE BENCHMARK");
    println!(
        "  AMF Runtime: {} | Resolution: {}x{} @ {} FPS | GOP Source: {} Mbps | Warmup/Frames: {}/{}",
        report.runtime_version,
        args.width,
        args.height,
        args.fps,
        args.bitrate_mbps,
        args.warmup,
        args.frames
    );
    println!();
    println!(
        "  {:<18} {:<6} {:<10} {:<10} {:<10} {:<11} {:<10}",
        "Codec", "Format", "Decode FPS", "MPix/s", "Avg ms/f", "Decoded", "DX11 Surf"
    );
    for r in &report.results {
        println!(
            "  {:<18} {:<6} {:<10.2} {:<10.2} {:<10.3} {:<11} {:<10}",
            r.codec,
            r.surface_format,
            r.decode_fps,
            r.megapixels_per_sec,
            r.avg_frame_ms,
            format!("{}/{}", r.decoded_frames, r.submitted_packets),
            if r.valid_surfaces { "VALID" } else { "INVALID" }
        );
    }

    if let Some(ref t) = report.telemetry {
        println!();
        print_telemetry_summary("VCN Decode Telemetry", t);
    }

    Ok(())
}

fn resolve_target_gguf_path(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    let assets = discover_local_llm_assets();
    if let Some(first) = assets.gguf_models.first() {
        Ok(PathBuf::from(first))
    } else {
        anyhow::bail!("no local .gguf model found automatically; pass --model <PATH_TO_MODEL.gguf>")
    }
}

fn cmd_llm_discover(json: bool) -> Result<()> {
    let assets = discover_local_llm_assets();
    if json {
        println!("{}", serde_json::to_string_pretty(&assets)?);
        return Ok(());
    }

    println!("RDNA FORGE LOCAL LLM ASSET DISCOVERY");
    println!("  Discovered GGUF Models ({}):", assets.gguf_models.len());
    if assets.gguf_models.is_empty() {
        println!("    (none found in .lmstudio/models or standard directories)");
    } else {
        for m in &assets.gguf_models {
            println!("    - {m}");
        }
    }
    println!(
        "  Vulkan llama-server: {}",
        assets.vulkan_server_bin.as_deref().unwrap_or("not found")
    );
    println!(
        "  ROCm/HIP llama-server: {}",
        assets.hip_server_bin.as_deref().unwrap_or("not found")
    );
    println!(
        "  ROCm Runtime DLL Dir (hipblas.dll): {}",
        assets.rocm_lib_dir.as_deref().unwrap_or("not found")
    );
    Ok(())
}

fn cmd_llm_inspect(model: Option<&Path>, json: bool) -> Result<()> {
    let path = resolve_target_gguf_path(model)?;
    let meta = inspect_gguf_model(&path)
        .with_context(|| format!("failed to inspect GGUF model '{}'", path.display()))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&meta)?);
        return Ok(());
    }

    println!("RDNA FORGE GGUF MODEL INSPECTOR");
    println!(
        "  File: {} ({:.2} GiB / {} MiB)",
        meta.file_name,
        gib(meta.file_size_bytes),
        meta.file_size_bytes.div_ceil(1024 * 1024)
    );
    println!("  Path: {}", meta.file_path);
    println!(
        "  Model: {} ({}) | Quant: {} (quantized by {})",
        meta.model_name.as_deref().unwrap_or("unknown"),
        meta.size_label.as_deref().unwrap_or("n/a"),
        meta.quantization_name,
        meta.quantized_by.as_deref().unwrap_or("unknown")
    );
    println!(
        "  GGUF: v{} | Tensors: {} | Architecture: {}",
        meta.gguf_version, meta.tensor_count, meta.architecture
    );
    println!(
        "  Layers: {} blocks (+{} MTP) | mean {:.1} MiB/layer [{}] | Declared Max Context: {} tokens",
        meta.block_count,
        meta.nextn_predict_layers,
        meta.bytes_per_layer as f64 / (1024.0 * 1024.0),
        meta.tensor_layout_basis(),
        meta.declared_context_length
    );
    if let Some(l) = &meta.tensor_layout {
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        println!(
            "  Tensors (aligned MiB): layers {:.1} | embedding {:.1} | output {:.1} | shared {:.1} | est. GPU weights at full offload {:.1}",
            mib(l.repeating_layer_bytes()),
            mib(l.embedding_bytes),
            mib(l.output_bytes),
            mib(l.shared_bytes),
            mib(l.full_offload_bytes())
        );
    }
    for note in meta.tensor_layout_notes() {
        println!("  NOTE: {note}");
    }
    println!(
        "  Dimensions: embd={}, ffn={}, heads={} (kv={}), head_dim=({}/{})",
        meta.embedding_length,
        meta.feed_forward_length
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        meta.head_count,
        meta.head_count_kv,
        meta.key_length,
        meta.value_length
    );
    if let Some(interval) = meta.full_attention_interval {
        println!(
            "  Hybrid SSM/Attention: full attention every {} layers (ssm_state={:?}, ssm_inner={:?})",
            interval, meta.ssm_state_size, meta.ssm_inner_size
        );
    }
    if meta.is_moe {
        println!(
            "  Mixture-of-Experts (MoE): {:?} routed experts ({:?} active/token, expert_ffn={:?})",
            meta.expert_count, meta.expert_used_count, meta.expert_feed_forward_length
        );
    }
    Ok(())
}

fn cmd_llm_plan(args: &LlmPlanArgs) -> Result<()> {
    let path = resolve_target_gguf_path(args.model.as_deref())?;
    let meta = inspect_gguf_model(&path)
        .with_context(|| format!("failed to inspect GGUF model '{}'", path.display()))?;
    let plan =
        plan_llm_configuration_for_device(&meta, args.device, args.context, args.safety_margin_mib)
            .context("failed to build LLM optimization plan")?;
    let json_text = serde_json::to_string_pretty(&plan)?;
    write_optional_output(args.output.as_deref(), &json_text)?;

    if args.json {
        println!("{json_text}");
        return Ok(());
    }

    println!("RDNA FORGE LLM OPTIMIZATION & ANTI-WDDM-SPILL PLAN");
    println!(
        "  Model: {} ({}, {:.2} GiB, {} layers, arch={})",
        plan.model.file_name,
        plan.model.quantization_name,
        gib(plan.model.file_size_bytes),
        plan.model.block_count,
        plan.model.architecture
    );
    let profile_note = plan
        .matched_profile_id
        .as_deref()
        .map(|id| format!(" | Profile={id}"))
        .unwrap_or_default();
    let vram_kind = if plan.is_apu_unified_memory {
        "Unified APU Memory"
    } else {
        "Live WDDM VRAM"
    };
    println!(
        "  Target GPU [{}]: {} ({}, {}){} | Physical CPU Cores: {}",
        plan.gpu_index,
        plan.gpu_name,
        plan.gpu_architecture,
        plan.gfx_target,
        profile_note,
        plan.physical_cpu_cores
    );
    println!(
        "  {vram_kind}: Total={} MiB | Used={} MiB | Free={} MiB | Safety Margin={} MiB",
        plan.total_vram_mib, plan.used_vram_mib, plan.free_vram_mib, plan.safety_margin_mib
    );
    if plan.recoverable_layers_if_apps_closed > 0 {
        println!(
            "  VRAM Recovery Potential: Closing background apps holding VRAM recovers up to +{} GPU layer(s).",
            plan.recoverable_layers_if_apps_closed
        );
    }

    if !plan.warnings.is_empty() {
        println!();
        println!("WARNINGS & HARDWARE TRAPS");
        for w in &plan.warnings {
            println!("  ! {w}");
        }
    }

    println!();
    println!(
        "KV CACHE & VRAM BUDGETS (@ {} context tokens)",
        plan.target_context_tokens
    );
    println!(
        "  {:<8} {:<8} {:<12} {:<11} {:<11} {:<13} {:<14} {:<12}",
        "KV Type",
        "Req FA",
        "Weights(MiB)",
        "KV+SSM(MiB)",
        "Total(MiB)",
        "FreeVRAM Fit",
        "WDDM Prio Fit",
        "Strict/Prio NGL"
    );
    for b in &plan.kv_budgets {
        println!(
            "  {:<8} {:<8} {:<12} {:<11} {:<11} {:<13} {:<14} {}/{}/{}",
            b.kv_quant,
            if b.requires_flash_attn { "yes" } else { "no" },
            b.model_weights_mib,
            b.kv_and_state_mib,
            b.total_required_mib,
            if b.fits_in_free_vram {
                "YES"
            } else {
                "OVERFLOW"
            },
            if b.fits_with_wddm_memory_priority {
                "YES (Prio=1)"
            } else {
                "PARTIAL"
            },
            b.strict_zero_spill_ngl,
            b.memory_priority_ngl,
            plan.model.block_count
        );
    }

    println!();
    println!(
        "RECOMMENDED {} ({}) EXECUTION PROFILES",
        if plan.gpu_architecture.is_empty() {
            "AMD RADEON"
        } else {
            plan.gpu_architecture.as_str()
        },
        plan.gfx_target
    );
    for (idx, p) in plan.recommended_profiles.iter().enumerate() {
        let ncmoe_note = if p.n_cpu_moe > 0 {
            format!(", ncmoe={}", p.n_cpu_moe)
        } else {
            String::new()
        };
        println!(
            "  [{}] {} (backend={}, dev={}, ngl={}{ncmoe_note}, fa={}, kv={}/{})",
            idx + 1,
            p.title,
            p.backend,
            p.device_selector,
            p.n_gpu_layers,
            p.flash_attn,
            p.cache_type_k,
            p.cache_type_v
        );
        println!("      Why: {}", p.rationale);
        println!("      Cmd: {}", p.llama_server_cmd);
    }

    if !plan.top_vram_processes.is_empty() {
        println!();
        println!("TOP VRAM CONSUMERS (WDDM GPU PROCESS MEMORY)");
        println!(
            "  {:<8} {:<30} {:>14} {:>12} {:>18}",
            "PID", "Process", "Dedicated(MiB)", "Shared(MiB)", "Recoverable Layers"
        );
        for p in &plan.top_vram_processes {
            println!(
                "  {:<8} {:<30} {:>14} {:>12} {:>18}",
                p.pid,
                p.process_name,
                p.dedicated_vram_mib,
                p.shared_vram_mib,
                format!("+{} layers", p.recoverable_gpu_layers)
            );
        }
    }

    println!();
    println!("LM STUDIO & LAUNCHER CHECKLIST");
    for tip in &plan.lm_studio_tips {
        println!("  {tip}");
    }

    Ok(())
}

fn cmd_llm_autotune(args: &LlmAutotuneArgs) -> Result<()> {
    let model_path = resolve_target_gguf_path(args.model.as_deref())?;
    let assets = discover_local_llm_assets();
    let vulkan_bin = args
        .vulkan_bin
        .clone()
        .or_else(|| assets.vulkan_bench_bin.map(PathBuf::from));
    let hip_bin = args
        .hip_bin
        .clone()
        .or_else(|| assets.hip_bench_bin.map(PathBuf::from));
    let rocm_lib_dir = args
        .rocm_lib_dir
        .clone()
        .or_else(|| assets.rocm_lib_dir.map(PathBuf::from));

    let cfg = LlmAutotuneConfig {
        model_path,
        device_index: args.device,
        vulkan_bin,
        hip_bin,
        rocm_lib_dir,
        prompt_tokens: args.prompt_tokens,
        gen_tokens: args.gen_tokens,
        context_depth: args.context_depth,
        reps: args.reps,
    };

    let report = autotune_llm_model(&cfg).context("failed to execute LLM empirical autotuner")?;
    let json_text = serde_json::to_string_pretty(&report)?;
    write_optional_output(args.output.as_deref(), &json_text)?;

    if args.json {
        println!("{json_text}");
        return Ok(());
    }

    println!("RDNA FORGE EMPIRICAL LLM AUTOTUNER REPORT");
    println!(
        "  Model: {} ({}, {:.2} GiB) | Probe: pp{} + tg{} @ depth d={}",
        report.plan.model.file_name,
        report.plan.model.quantization_name,
        gib(report.plan.model.file_size_bytes),
        report.prompt_tokens,
        report.gen_tokens,
        report.context_depth
    );
    println!();
    println!(
        "  {:<28} {:<7} {:<4} {:<10} {:<14} {:<14} {:<10}",
        "Profile ID", "Backend", "FA", "KV Cache", "Prefill(tok/s)", "Decode(tok/s)", "TG Speedup"
    );
    for c in &report.candidates {
        let pp = c
            .pp_tok_s
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "ERR".to_string());
        let tg = c
            .tg_tok_s
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "ERR".to_string());
        let spd = c
            .tg_speedup_vs_baseline
            .map(|v| format!("{v:.2}x"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "  {:<28} {:<7} {:<4} {:<10} {:<14} {:<14} {:<10}",
            c.profile_id,
            c.backend,
            if c.flash_attn { "on" } else { "off" },
            format!("{}/{}", c.cache_type_k, c.cache_type_v),
            pp,
            tg,
            spd
        );
    }

    println!();
    println!(
        "  Winner (Fastest Token Generation): {}",
        report.best_decode_profile_id.as_deref().unwrap_or("n/a")
    );
    println!(
        "  Winner (Fastest Prompt Ingestion): {}",
        report.best_prefill_profile_id.as_deref().unwrap_or("n/a")
    );
    println!(
        "  Winner (Best Overall Balanced):    {}",
        report.best_balanced_profile_id.as_deref().unwrap_or("n/a")
    );
    if let Some(best_id) = &report.best_balanced_profile_id {
        if let Some(best) = report.candidates.iter().find(|c| &c.profile_id == best_id) {
            println!("  Recommended Command:");
            println!("    {}", best.llama_server_cmd);
        }
    }

    Ok(())
}

fn cmd_llm_apply_lmstudio(args: &LlmApplyLmstudioArgs) -> Result<()> {
    let model_path = resolve_target_gguf_path(args.model.as_deref())?;
    let res = apply_lm_studio_config(
        &model_path,
        args.device,
        args.profile.as_deref(),
        args.context,
        args.safety_margin_mib,
        !args.apply,
    )
    .context("failed to build/apply LM Studio configuration")?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    let mode = if res.applied { "APPLIED" } else { "DRY-RUN" };
    println!(
        "[{mode}] LM STUDIO CONFIGURATION PATCHER — {}",
        res.profile_title
    );
    println!("  Profile ID: {}", res.profile_id);
    println!("  LM Studio Engine: {}", res.selected_lmstudio_backend);
    println!("  Model Config File: {}", res.model_config_path);
    if let Some(ref bak) = res.model_config_backup_path {
        println!("  Model Config Backup: {bak}");
    }
    println!("  Backend Prefs File: {}", res.backend_prefs_path);
    if let Some(ref bak) = res.backend_prefs_backup_path {
        println!("  Backend Prefs Backup: {bak}");
    }
    println!(
        "  Settings: context={}, offload={}, flash_attn={}, kv={}/{}, threads={}, disable_igpu={}",
        res.context_length,
        res.offload_ratio,
        res.flash_attention,
        res.cache_type_k,
        res.cache_type_v,
        res.cpu_threads,
        res.disabled_igpu
    );
    if !res.applied {
        println!("  No file was modified (dry-run preview). Pass --apply to write config files with .bak backup.");
    }
    Ok(())
}

fn cmd_llm_apply_env(args: &LlmApplyEnvArgs) -> Result<()> {
    let model_path = resolve_target_gguf_path(args.model.as_deref())?;
    let res = apply_windows_and_ollama_env_tweaks(
        &model_path,
        args.device,
        args.context,
        args.kv_quant.into(),
        args.modelfile_out.as_deref(),
        !args.apply,
    )
    .context("failed to build/apply Windows & Ollama environment tweaks")?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&res)?);
        return Ok(());
    }

    let mode = if res.applied { "APPLIED" } else { "DRY-RUN" };
    println!("[{mode}] WINDOWS USER ENVIRONMENT (HKCU\\Environment) & OLLAMA TWEAKS");
    for c in &res.env_changes {
        println!(
            "  {:<34} : {:<10} -> {} ({})",
            c.key,
            c.previous_value.as_deref().unwrap_or("(unset)"),
            c.new_value,
            c.purpose
        );
    }
    println!();
    println!("GENERATED OLLAMA MODELFILE");
    for line in res.ollama_modelfile_content.lines() {
        println!("  {line}");
    }
    println!("  Import Cmd: {}", res.ollama_create_cmd);
    if !res.applied {
        println!();
        println!("  No registry variable or file was modified (dry-run preview). Pass --apply to commit.");
    }
    Ok(())
}

#[derive(Serialize)]
struct CombinedRollbackReport {
    lmstudio: Option<LmStudioRollbackResult>,
    env: Option<EnvRollbackResult>,
}

fn cmd_llm_rollback(args: &LlmRollbackArgs) -> Result<()> {
    let do_lms = args.lmstudio || !args.env;
    let do_env = args.env || !args.lmstudio;
    let dry_run = !args.apply;

    let lmstudio = if do_lms {
        match rollback_lm_studio_config(args.model.as_deref(), dry_run) {
            Ok(res) => Some(res),
            Err(forge_core::ForgeError::RuntimeUnavailable { .. }) if !args.lmstudio => None,
            Err(e) => return Err(e).context("failed to inspect/restore LM Studio backups"),
        }
    } else {
        None
    };

    let env = if do_env {
        Some(
            rollback_windows_env_tweaks(dry_run)
                .context("failed to inspect/rollback HKCU\\Environment variables")?,
        )
    } else {
        None
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&CombinedRollbackReport { lmstudio, env })?
        );
        return Ok(());
    }

    let mode = if args.apply { "APPLIED" } else { "DRY-RUN" };
    println!("[{mode}] RDNA FORGE SAFE ROLLBACK (RESTORE / UNDO)");
    if let Some(ref lms) = lmstudio {
        println!();
        println!("LM STUDIO BACKUPS (.json.bak -> .json)");
        if lms.actions.is_empty() {
            println!("  No LM Studio .json.bak backup files found.");
        } else {
            for a in &lms.actions {
                println!(
                    "  [{}] {} <- {}",
                    if a.restored {
                        "RESTORED"
                    } else {
                        "WOULD RESTORE"
                    },
                    a.target_path,
                    a.backup_path
                );
            }
        }
    }
    if let Some(ref env_res) = env {
        println!();
        println!("WINDOWS USER ENVIRONMENT (HKCU\\Environment)");
        for k in &env_res.keys {
            let status = match (&k.current_value, k.removed) {
                (None, _) => "unset (no-op)".to_string(),
                (Some(val), true) => format!("DELETED (was '{val}')"),
                (Some(val), false) => format!("WOULD DELETE (currently '{val}')"),
            };
            println!("  {:<34} : {status}", k.key);
        }
    }
    if dry_run {
        println!();
        println!("  No file or registry variable was modified (dry-run preview). Pass --apply to restore/delete.");
    }
    Ok(())
}

fn cmd_llm_recommend(args: &LlmRecommendArgs) -> Result<()> {
    let report = recommend_models_for_device(args.device, args.context, args.safety_margin_mib)
        .context("failed to evaluate GPU model recommendations")?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let profile_note = report
        .matched_profile_id
        .as_deref()
        .map(|id| format!(" | Profile={id}"))
        .unwrap_or_default();
    let vram_kind = if report.is_apu_unified_memory {
        "Unified APU Memory"
    } else {
        "Physical WDDM VRAM"
    };
    println!("RDNA FORGE GPU MODEL RECOMMENDER (WHAT FITS ON MY GPU?)");
    println!(
        "  GPU [{}]: {} ({}, {}){profile_note}",
        report.gpu_index, report.gpu_name, report.gpu_architecture, report.gfx_target
    );
    println!(
        "  {vram_kind}: Total={} MiB | Free={} MiB | Priority=1 Budget={} MiB (@ {} context)",
        report.total_vram_mib,
        report.free_vram_mib,
        report.priority_budget_mib,
        report.target_context_tokens
    );
    println!();
    println!(
        "  {:<30} {:<15} {:<8} {:>9} {:>9} {:<22} {:<8} Notes",
        "Model", "Size", "Quant", "Req(MiB)", "MaxCtx", "Fit Verdict", "Backend"
    );
    for r in &report.recommendations {
        let verdict_str = match r.verdict {
            ModelFitVerdict::IdealFreeVram => "IDEAL (Free VRAM)",
            ModelFitVerdict::FitsWddmPriority => "WDDM Priority=1",
            ModelFitVerdict::FitsMoeOffload => "MoE (-ncmoe)",
            ModelFitVerdict::TooLarge => "TOO LARGE",
        };
        let ctx_str = if r.recommended_max_context > 0 {
            format!("{}K", r.recommended_max_context / 1024)
        } else {
            "--".to_string()
        };
        let short_backend = if r.suggested_backend.contains("hipBLASLt") {
            "ROCm/Vk"
        } else if r.suggested_backend.contains("ROCm") {
            "Vk/ROCm"
        } else {
            "Vulkan"
        };
        println!(
            "  {:<30} {:<15} {:<8} {:>9} {:>9} {:<22} {:<8} {}",
            r.name,
            r.params_label,
            r.quant,
            r.total_required_mib,
            ctx_str,
            verdict_str,
            short_backend,
            r.notes
        );
    }
    Ok(())
}

fn cmd_vram_top(args: &VramTopArgs) -> Result<()> {
    let limit = args.limit.clamp(1, 100);
    let rows =
        query_top_vram_processes(limit).context("failed to query Windows per-process GPU VRAM")?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    println!("RDNA FORGE TOP VRAM CONSUMERS (WDDM)");
    if rows.is_empty() {
        println!("  No processes currently report GPU VRAM usage.");
        return Ok(());
    }

    println!(
        "  {:<8} {:<32} {:>14} {:>12} {:<24}",
        "PID", "Process", "Dedicated(MiB)", "Shared(MiB)", "Adapter LUID"
    );
    let mut total_ded_mib = 0u64;
    for r in &rows {
        total_ded_mib = total_ded_mib.saturating_add(r.dedicated_vram_mib);
        println!(
            "  {:<8} {:<32} {:>14} {:>12} {:<24}",
            r.pid,
            r.process_name,
            r.dedicated_vram_mib,
            r.shared_vram_mib,
            r.adapter_luid.as_deref().unwrap_or("--")
        );
    }
    println!(
        "  Sum across top {} processes: {} MiB dedicated VRAM",
        rows.len(),
        total_ded_mib
    );
    Ok(())
}

fn cmd_report(args: &ReportArgs) -> Result<()> {
    let report = generate_github_hardware_report(args.verify)
        .context("failed to generate GitHub hardware report")?;

    if args.json {
        let json = serde_json::to_string_pretty(&report)?;
        write_optional_output(args.output.as_deref(), &json)?;
        println!("{json}");
    } else {
        write_optional_output(args.output.as_deref(), &report.markdown)?;
        println!("Issue Title: {}", report.issue_title);
        println!();
        println!("{}", report.markdown);
    }
    Ok(())
}

fn cmd_profile_show(path: &Path, json: bool) -> Result<()> {
    let profile = HardwareProfile::from_file(path)
        .with_context(|| format!("failed to load profile '{}'", path.display()))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&profile)?);
    } else {
        println!(
            "{} ({}, {})",
            profile.device.name, profile.device.architecture, profile.device.gfx_target
        );
        println!(
            "  CUs: {} ({} WGPs), Wave sizes: {:?}, LDS/CU: {} B, Vulkan shared mem: {} B",
            profile.compute.compute_units,
            profile.compute.wgp_count,
            profile.compute.wave_sizes,
            profile.compute.lds_bytes_per_cu,
            profile.compute.vulkan_max_shared_memory_bytes
        );
        println!(
            "  VRAM: {:.2} GiB, Bus: {}-bit, L2: {} MiB, Infinity Cache: {} MiB",
            gib(profile.memory.vram_bytes),
            profile.memory.bus_width_bits,
            profile.memory.l2_cache_bytes / (1024 * 1024),
            profile.memory.infinity_cache_bytes / (1024 * 1024)
        );
        println!(
            "  Capabilities: fp16={}, bf16={}, fp8={}, dp4a_i8={}, dot8_i4={}, wmma={}, mfma={}",
            profile.capabilities.fp16,
            profile.capabilities.bf16,
            profile.capabilities.fp8,
            profile.capabilities.dp4a_i8,
            profile.capabilities.dot8_i4,
            profile.capabilities.wmma,
            profile.capabilities.mfma
        );
    }
    Ok(())
}

fn cmd_profile_list(json: bool) -> Result<()> {
    let profiles = builtin_profiles();
    if json {
        println!("{}", serde_json::to_string_pretty(&profiles)?);
        return Ok(());
    }

    println!(
        "RDNA FORGE BUILT-IN HARDWARE PROFILES ({} total)",
        profiles.len()
    );
    println!(
        "  {:<13} {:<27} {:<8} {:<8} {:>5} {:>8} {:<6} {:<5}",
        "ID", "Name", "Arch", "Target", "CUs", "VRAM", "WMMA", "FP8"
    );
    for entry in &profiles {
        let p = &entry.profile;
        println!(
            "  {:<13} {:<27} {:<8} {:<8} {:>5} {:>6.1}G {:<6} {:<5}",
            entry.id,
            p.device.name,
            p.device.architecture,
            p.device.gfx_target,
            p.compute.compute_units,
            gib(p.memory.vram_bytes),
            if p.capabilities.wmma { "yes" } else { "no" },
            if p.capabilities.fp8 { "yes" } else { "no" },
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct ProfileMatchCliReport<'a> {
    device_index: usize,
    device_name: &'a str,
    architecture: String,
    gfx_target: Option<String>,
    pci_vendor_id: String,
    pci_device_id: String,
    matched_profile: Option<forge_core::BuiltinProfileInfo>,
}

fn cmd_profile_match(device_index: usize, json: bool) -> Result<()> {
    let snapshot = discover_all();
    let gpu = snapshot.gpus.get(device_index).ok_or_else(|| {
        anyhow::anyhow!(
            "GPU device index {} not found ({} device(s) discovered)",
            device_index,
            snapshot.gpus.len()
        )
    })?;
    let matched = match_profile_for_device(gpu);

    if json {
        let report = ProfileMatchCliReport {
            device_index: gpu.index,
            device_name: &gpu.name,
            architecture: gpu.architecture.to_string(),
            gfx_target: gpu.gfx_target.as_ref().map(|t| t.base.clone()),
            pci_vendor_id: format!("0x{:04X}", gpu.vendor_id),
            pci_device_id: format!("0x{:04X}", gpu.device_id),
            matched_profile: matched,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!(
        "Device [{}]: {} ({}, {}, PCI 0x{:04X}:0x{:04X})",
        gpu.index,
        gpu.name,
        gpu.architecture,
        gpu.gfx_target
            .as_ref()
            .map(|t| t.base.as_str())
            .unwrap_or("unknown"),
        gpu.vendor_id,
        gpu.device_id
    );
    match matched {
        Some(entry) => {
            println!(
                "  Matched Profile: {} (id={}, path={})",
                entry.profile.device.name, entry.id, entry.relative_path
            );
            println!(
                "  Specs: {} CUs ({} WGPs), {:.1} GiB, L2 {} MiB, Infinity Cache {} MiB",
                entry.profile.compute.compute_units,
                entry.profile.compute.wgp_count,
                gib(entry.profile.memory.vram_bytes),
                entry.profile.memory.l2_cache_bytes / (1024 * 1024),
                entry.profile.memory.infinity_cache_bytes / (1024 * 1024)
            );
        }
        None => {
            println!("  No built-in AMD Radeon hardware profile matched this adapter (generic architecture policy applies).");
        }
    }
    Ok(())
}

fn write_optional_output(path: Option<&Path>, content: &str) -> Result<()> {
    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(p, content)?;
    }
    Ok(())
}

fn gib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use monitor_tune::MonitorFormat;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_parses_negative_power_limit_and_monitor_formats() {
        let neg = Cli::try_parse_from(["forge", "tune", "power-limit", "-5"])
            .expect("negative power-limit must parse without '--'");
        match neg.command {
            Commands::Tune {
                action:
                    TuneSubcommand::PowerLimit {
                        limit_pct,
                        device_id,
                        apply,
                        json,
                    },
            } => {
                assert_eq!(limit_pct, -5);
                assert!(device_id.is_none());
                assert!(!apply);
                assert!(!json);
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let mon = Cli::try_parse_from([
            "forge",
            "monitor",
            "--format",
            "jsonl",
            "--count",
            "3",
            "--interval-ms",
            "250",
            "--device-id",
            "0x73AF",
        ])
        .expect("monitor args must parse");
        match mon.command {
            Commands::Monitor(args) => {
                assert_eq!(args.format, MonitorFormat::Jsonl);
                assert_eq!(args.count, Some(3));
                assert_eq!(args.interval_ms, 250);
                assert_eq!(args.device_id.as_deref(), Some("0x73AF"));
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let vid = Cli::try_parse_from(["forge", "video", "caps", "--json"])
            .expect("video caps args must parse");
        match vid.command {
            Commands::Video {
                action: VideoSubcommand::Caps { json },
            } => assert!(json),
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let bat = Cli::try_parse_from([
            "forge",
            "bench",
            "vulkan",
            "--kernel",
            "gemv-q4",
            "--batch-dispatches",
            "16",
        ])
        .expect("vulkan bench batch-dispatches args must parse");
        match bat.command {
            Commands::Bench {
                target: BenchSubcommand::Vulkan(args),
            } => assert_eq!(args.batch_dispatches, Some(16)),
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let q4k = Cli::try_parse_from(["forge", "bench", "vulkan", "--kernel", "gemv-q4-k"])
            .expect("gemv-q4-k kernel choice must parse");
        assert!(matches!(
            q4k.command,
            Commands::Bench {
                target: BenchSubcommand::Vulkan(VulkanBenchArgs {
                    kernel: KernelChoice::GemvQ4K,
                    ..
                })
            }
        ));

        let fa = Cli::try_parse_from(["forge", "bench", "vulkan", "--kernel", "flash-attn"])
            .expect("flash-attn kernel choice must parse");
        assert!(matches!(
            fa.command,
            Commands::Bench {
                target: BenchSubcommand::Vulkan(VulkanBenchArgs {
                    kernel: KernelChoice::FlashAttn,
                    ..
                })
            }
        ));

        let vb = Cli::try_parse_from([
            "forge", "video", "bench", "--codec", "hevc10", "--width", "3840", "--height", "2160",
            "--preset", "speed", "--frames", "120",
        ])
        .expect("video bench args must parse");
        match vb.command {
            Commands::Video {
                action: VideoSubcommand::Bench(args),
            } => {
                assert_eq!(args.codec, VideoCodecChoice::Hevc10);
                assert_eq!(args.width, 3840);
                assert_eq!(args.height, 2160);
                assert_eq!(args.preset, VideoPresetChoice::Speed);
                assert_eq!(args.frames, 120);
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let llm_plan =
            Cli::try_parse_from(["forge", "llm", "plan", "--device", "1", "--context", "8192"])
                .expect("llm plan args must parse");
        match llm_plan.command {
            Commands::Llm {
                action: LlmSubcommand::Plan(args),
            } => {
                assert_eq!(args.device, Some(1));
                assert_eq!(args.context, 8192);
                assert_eq!(args.safety_margin_mib, 512);
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let llm_lms = Cli::try_parse_from([
            "forge",
            "llm",
            "apply-lmstudio",
            "--device",
            "0",
            "--profile",
            "vulkan_wddm_priority",
            "--context",
            "4096",
        ])
        .expect("llm apply-lmstudio args must parse");
        match llm_lms.command {
            Commands::Llm {
                action: LlmSubcommand::ApplyLmstudio(args),
            } => {
                assert_eq!(args.device, Some(0));
                assert_eq!(args.profile.as_deref(), Some("vulkan_wddm_priority"));
                assert_eq!(args.context, 4096);
                assert!(!args.apply, "apply-lmstudio must default to dry-run");
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let llm_env = Cli::try_parse_from([
            "forge",
            "llm",
            "apply-env",
            "--device",
            "0",
            "--kv-quant",
            "q4-0",
            "--context",
            "8192",
        ])
        .expect("llm apply-env args must parse");
        match llm_env.command {
            Commands::Llm {
                action: LlmSubcommand::ApplyEnv(args),
            } => {
                assert_eq!(args.device, Some(0));
                assert_eq!(args.kv_quant, KvQuantChoice::Q4_0);
                assert_eq!(args.context, 8192);
                assert!(!args.apply, "apply-env must default to dry-run");
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let prof_list = Cli::try_parse_from(["forge", "profile", "list", "--json"])
            .expect("profile list --json must parse");
        assert!(matches!(
            prof_list.command,
            Commands::Profile {
                action: ProfileSubcommand::List { json: true }
            }
        ));

        let prof_match = Cli::try_parse_from(["forge", "profile", "match", "--device", "1"])
            .expect("profile match --device 1 must parse");
        assert!(matches!(
            prof_match.command,
            Commands::Profile {
                action: ProfileSubcommand::Match {
                    device: 1,
                    json: false
                }
            }
        ));

        let vram = Cli::try_parse_from(["forge", "vram-top", "--limit", "5", "--json"])
            .expect("vram-top args must parse");
        assert!(matches!(
            vram.command,
            Commands::VramTop(VramTopArgs {
                limit: 5,
                json: true
            })
        ));

        let rec = Cli::try_parse_from(["forge", "llm", "recommend", "--device", "0"])
            .expect("llm recommend must parse");
        assert!(matches!(
            rec.command,
            Commands::Llm {
                action: LlmSubcommand::Recommend(LlmRecommendArgs {
                    device: Some(0),
                    ..
                })
            }
        ));

        let rb = Cli::try_parse_from(["forge", "llm", "rollback", "--env"])
            .expect("llm rollback must parse");
        match rb.command {
            Commands::Llm {
                action: LlmSubcommand::Rollback(args),
            } => {
                assert!(args.env);
                assert!(!args.apply, "rollback must default to dry-run");
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }

        let rep = Cli::try_parse_from(["forge", "report", "--verify"])
            .expect("report --verify must parse");
        assert!(matches!(
            rep.command,
            Commands::Report(ReportArgs { verify: true, .. })
        ));

        let av1 = Cli::try_parse_from(["forge", "video", "bench", "--codec", "av1"])
            .expect("video bench --codec av1 must parse");
        assert!(matches!(
            av1.command,
            Commands::Video {
                action: VideoSubcommand::Bench(VideoBenchArgs {
                    codec: VideoCodecChoice::Av1,
                    ..
                })
            }
        ));

        let dec = Cli::try_parse_from([
            "forge",
            "video",
            "decode-bench",
            "--codec",
            "hevc",
            "--width",
            "3840",
            "--height",
            "2160",
            "--frames",
            "60",
        ])
        .expect("video decode-bench must parse");
        match dec.command {
            Commands::Video {
                action: VideoSubcommand::DecodeBench(args),
            } => {
                assert_eq!(args.codec, VideoDecodeCodecChoice::Hevc);
                assert_eq!(args.width, 3840);
                assert_eq!(args.height, 2160);
                assert_eq!(args.frames, 60);
            }
            other => panic!("unexpected command parsed: {other:?}"),
        }
    }
}
