use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use forge_benchmark::{
    peak_process_rss_bytes, resolve_default_triton_script, run_llama_comparison, run_triton_suite,
    run_vulkan_suite, LlamaBenchConfig, TritonBenchConfig, VulkanBenchConfig,
    VulkanKernelSelection,
};
use forge_core::{
    CapabilityResult, DiagnosticReport, GpuDeviceInfo, HardwareProfile, SupportStatus,
};
use forge_device::discover_all;
use forge_profiler::{
    check_competing_workload_warning, check_vram_pressure_warning, snapshot_all_gpus,
    GpuTelemetrySample, TelemetrySummary,
};
use forge_runtime::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    deterministic_q4_0_weights, pack_i8_b_transposed_k4, pack_i8_row_major_k4, probe_all_backends,
    ref_gemm_f32_exact_f64, ref_gemm_i8_to_i32, ref_gemv_q4_0_f32, ref_gemv_w8a32_f32,
    ref_rmsnorm_f32, ref_rmsnorm_rope_f32, ref_rope_f32, verify_hip_transfer, VulkanComputeEngine,
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
    /// Inspect and validate hardware target profiles.
    Profile {
        #[command(subcommand)]
        action: ProfileSubcommand,
    },
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
    Rmsnorm,
    Rope,
    FusedRmsRope,
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
            KernelChoice::Rmsnorm => Self::RmsNorm,
            KernelChoice::Rope => Self::Rope,
            KernelChoice::FusedRmsRope => Self::FusedRmsRope,
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
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long)]
    json: bool,
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
        Commands::Profile { action } => match action {
            ProfileSubcommand::Show { path, json } => cmd_profile_show(&path, json),
        },
    }
}

#[derive(Serialize)]
struct DoctorCliReport<'a> {
    #[serde(flatten)]
    report: &'a DiagnosticReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    telemetry: Option<&'a [GpuTelemetrySample]>,
}

const VULKAN_VERIFY_CAPABILITIES: &[&str] = &[
    "GEMM (Vulkan FP32)",
    "GEMM (Vulkan FP32 tiled vec4)",
    "GEMM (Vulkan INT8 DP4A)",
    "GEMV (Vulkan Wave32 W8A32)",
    "GEMV (Vulkan Wave32 Q4_0)",
    "RMSNorm (Vulkan FP32)",
    "RoPE (Vulkan FP32)",
    "Fused RMSNorm+RoPE (Vulkan Subgroup)",
];

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
        capabilities.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::NotTested,
            detail: Some("pass --verify to run live HIP round-trip test".to_string()),
        });
        for &name in VULKAN_VERIFY_CAPABILITIES {
            capabilities.push(CapabilityResult {
                name: name.to_string(),
                status: SupportStatus::NotTested,
                detail: Some("pass --verify to run live compute check".to_string()),
            });
        }
    }

    let report = DiagnosticReport {
        system: snapshot.system,
        gpus: snapshot.gpus,
        backends,
        capabilities,
        missing_dependencies,
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

fn cap_result(name: &str, res: Result<String>) -> CapabilityResult {
    match res {
        Ok(detail) => CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Verified,
            detail: Some(detail),
        },
        Err(err) => CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(err.to_string()),
        },
    }
}

fn push_unsupported_vulkan_caps(results: &mut Vec<CapabilityResult>, reason: &str) {
    for &name in VULKAN_VERIFY_CAPABILITIES {
        results.push(CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(reason.to_string()),
        });
    }
}

fn verify_f32_gemm_cap(
    name: &str,
    run_fn: impl FnOnce(
        &[f32],
        &[f32],
        u32,
    ) -> forge_core::ForgeResult<(Vec<f32>, forge_runtime::GpuTiming)>,
) -> CapabilityResult {
    let res = (|| -> Result<String> {
        let n = 64u32;
        let a = deterministic_f32_data((n * n) as usize, 11);
        let b = deterministic_f32_data((n * n) as usize, 22);
        let (actual, timing) = run_fn(&a, &b, n)?;
        let expected = ref_gemm_f32_exact_f64(&a, &b, n as usize, n as usize, n as usize)?;
        let acc = compare_slices(&actual, &expected, 1e-3, 1e-4)?;
        if !acc.passed {
            anyhow::bail!(
                "numerical mismatch (max_abs_err={:.3e}, rmse={:.3e})",
                acc.max_abs_err,
                acc.rmse
            );
        }
        Ok(format!(
            "64x64x64 passed (max_abs_err={:.2e}, {:.3} ms)",
            acc.max_abs_err,
            timing.gpu_ms.unwrap_or(timing.wall_ms)
        ))
    })();
    match res {
        Ok(detail) => CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Verified,
            detail: Some(detail),
        },
        Err(err) => CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(err.to_string()),
        },
    }
}

fn run_live_capability_checks(gpus: &[GpuDeviceInfo]) -> Vec<CapabilityResult> {
    let mut results = Vec::new();

    match verify_hip_transfer(0, 4 * 1024 * 1024) {
        Ok(res) if res.h2d_d2h_bit_exact => results.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::Verified,
            detail: Some(format!(
                "{} ({}) 4 MiB H2D+D2H bit-exact in {:.3} ms ({:.2} GB/s)",
                res.device_name,
                res.gfx_target.as_deref().unwrap_or("unknown"),
                res.elapsed_ms,
                res.bandwidth_gb_s
            )),
        }),
        Ok(res) => results.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(format!(
                "round-trip data mismatch on {} ({})",
                res.device_name,
                res.gfx_target.as_deref().unwrap_or("unknown")
            )),
        }),
        Err(err) => results.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(err.to_string()),
        }),
    }

    if gpus.is_empty() {
        push_unsupported_vulkan_caps(&mut results, "no Vulkan GPU discovered");
        return results;
    }

    match VulkanComputeEngine::new(0) {
        Ok(engine) => {
            results.push(verify_f32_gemm_cap("GEMM (Vulkan FP32)", |a, b, n| {
                engine.run_gemm_f32(a, b, n, n, n)
            }));
            results.push(verify_f32_gemm_cap(
                "GEMM (Vulkan FP32 tiled vec4)",
                |a, b, n| engine.run_gemm_f32_tiled_vec4(a, b, n, n, n),
            ));

            let i8_res = (|| -> Result<String> {
                let n = 64u32;
                let n_us = n as usize;
                let a = deterministic_i8_data(n_us * n_us, 71);
                let b = deterministic_i8_data(n_us * n_us, 82);
                let ap = pack_i8_row_major_k4(&a, n_us, n_us)?;
                let bp = pack_i8_b_transposed_k4(&b, n_us, n_us)?;
                let (actual, timing) = engine.run_gemm_i8_dp4a(&ap, &bp, n, n / 4, n)?;
                let expected = ref_gemm_i8_to_i32(&a, &b, n_us, n_us, n_us)?;
                let acc = compare_i32_exact(&actual, &expected)?;
                if !acc.passed {
                    anyhow::bail!("integer mismatch (max_abs_err={:.3e})", acc.max_abs_err);
                }
                Ok(format!(
                    "64x64x64 exact (OpSDot, max_abs_err={:.0}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(match i8_res {
                Ok(detail) => CapabilityResult {
                    name: "GEMM (Vulkan INT8 DP4A)".to_string(),
                    status: SupportStatus::Verified,
                    detail: Some(detail),
                },
                Err(err) => CapabilityResult {
                    name: "GEMM (Vulkan INT8 DP4A)".to_string(),
                    status: SupportStatus::Unsupported,
                    detail: Some(err.to_string()),
                },
            });

            // RMSNorm 128x256
            let rms_res = (|| -> Result<String> {
                let (rows, cols) = (128u32, 256u32);
                let x = deterministic_f32_data((rows * cols) as usize, 33);
                let w = deterministic_f32_data(cols as usize, 44);
                let (actual, timing) = engine.run_rmsnorm_f32(&x, &w, rows, cols, 1e-5)?;
                let expected = ref_rmsnorm_f32(&x, &w, rows as usize, cols as usize, 1e-5)?;
                let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;
                if !acc.passed {
                    anyhow::bail!("numerical mismatch (max_abs_err={:.3e})", acc.max_abs_err);
                }
                Ok(format!(
                    "128x256 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(match rms_res {
                Ok(detail) => CapabilityResult {
                    name: "RMSNorm (Vulkan FP32)".to_string(),
                    status: SupportStatus::Verified,
                    detail: Some(detail),
                },
                Err(err) => CapabilityResult {
                    name: "RMSNorm (Vulkan FP32)".to_string(),
                    status: SupportStatus::Unsupported,
                    detail: Some(err.to_string()),
                },
            });

            // RoPE 64x8x64
            let rope_res = (|| -> Result<String> {
                let (seq, heads, dim) = (64u32, 8u32, 64u32);
                let x = deterministic_f32_data((seq * heads * dim) as usize, 55);
                let (actual, timing) = engine.run_rope_f32(&x, seq, heads, dim, 10_000.0)?;
                let expected =
                    ref_rope_f32(&x, seq as usize, heads as usize, dim as usize, 10_000.0)?;
                let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;
                if !acc.passed {
                    anyhow::bail!("numerical mismatch (max_abs_err={:.3e})", acc.max_abs_err);
                }
                Ok(format!(
                    "64x8x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(match rope_res {
                Ok(detail) => CapabilityResult {
                    name: "RoPE (Vulkan FP32)".to_string(),
                    status: SupportStatus::Verified,
                    detail: Some(detail),
                },
                Err(err) => CapabilityResult {
                    name: "RoPE (Vulkan FP32)".to_string(),
                    status: SupportStatus::Unsupported,
                    detail: Some(err.to_string()),
                },
            });

            // GEMV Wave32 W8A32 (64x64)
            let w8_res = (|| -> Result<String> {
                let (n, k) = (64u32, 64u32);
                let w_i8 = deterministic_i8_data((n * k) as usize, 61);
                let w_packed = pack_i8_row_major_k4(&w_i8, n as usize, k as usize)?;
                let scales = deterministic_f32_data(n as usize, 62);
                let x = deterministic_f32_data(k as usize, 63);
                let (actual, timing) =
                    engine.run_gemv_w8a32_wave32(&w_packed, &scales, &x, n, k)?;
                let expected = ref_gemv_w8a32_f32(&w_i8, &scales, &x, n as usize, k as usize)?;
                let acc = compare_slices(&actual, &expected, 1e-3, 1e-4)?;
                if !acc.passed {
                    anyhow::bail!("numerical mismatch (max_abs_err={:.3e})", acc.max_abs_err);
                }
                Ok(format!(
                    "64x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("GEMV (Vulkan Wave32 W8A32)", w8_res));

            // GEMV Wave32 Q4_0 (64x64)
            let q4_res = (|| -> Result<String> {
                let (n, k) = (64u32, 64u32);
                let (w_q4, scales) = deterministic_q4_0_weights(n as usize, k as usize, 64)?;
                let x = deterministic_f32_data(k as usize, 65);
                let (actual, timing) = engine.run_gemv_q4_0_wave32(&w_q4, &scales, &x, n, k)?;
                let expected = ref_gemv_q4_0_f32(&w_q4, &scales, &x, n as usize, k as usize)?;
                let acc = compare_slices(&actual, &expected, 1e-3, 1e-4)?;
                if !acc.passed {
                    anyhow::bail!("numerical mismatch (max_abs_err={:.3e})", acc.max_abs_err);
                }
                Ok(format!(
                    "64x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("GEMV (Vulkan Wave32 Q4_0)", q4_res));

            // Fused subgroup RMSNorm+RoPE (64x8x64, pos_offset=17)
            let fused_res = (|| -> Result<String> {
                let (seq, heads, dim, pos_offset) = (64u32, 8u32, 64u32, 17u32);
                let cols = (heads * dim) as usize;
                let x = deterministic_f32_data((seq as usize) * cols, 66);
                let w = deterministic_f32_data(cols, 67);
                let (actual, timing) = engine.run_rmsnorm_rope_f32_subgroup(
                    &x, &w, seq, heads, dim, pos_offset, 1e-5, 10_000.0,
                )?;
                let expected = ref_rmsnorm_rope_f32(
                    &x,
                    &w,
                    seq as usize,
                    heads as usize,
                    dim as usize,
                    pos_offset as usize,
                    1e-5,
                    10_000.0,
                )?;
                let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;
                if !acc.passed {
                    anyhow::bail!("numerical mismatch (max_abs_err={:.3e})", acc.max_abs_err);
                }
                Ok(format!(
                    "64x8x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result(
                "Fused RMSNorm+RoPE (Vulkan Subgroup)",
                fused_res,
            ));
        }
        Err(err) => push_unsupported_vulkan_caps(&mut results, &err.to_string()),
    }

    results
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
    println!("  Architecture: {}", gpu.architecture);
    println!(
        "  Target: {}",
        gpu.gfx_target
            .as_ref()
            .map(|t| t.base.as_str())
            .unwrap_or("unknown")
    );
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
    }
}
