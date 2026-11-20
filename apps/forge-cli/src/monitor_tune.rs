use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use forge_profiler::{
    check_competing_workload_warning, check_vram_pressure_warning, format_sample_csv_row,
    format_sample_table_row, parse_pci_device_id, read_all_gpu_tuning, reset_gpu_tuning_to_factory,
    set_gpu_power_limit, summarize_samples, AdlxSession, GpuTelemetrySample, GpuTuningReport,
    MonitorRecord, TELEMETRY_CSV_HEADER, TELEMETRY_TABLE_HEADER,
};

use crate::print_telemetry_summary_to;

static MONITOR_INTERRUPTED: AtomicBool = AtomicBool::new(false);

struct CtrlCHandlerGuard {
    #[cfg(windows)]
    registered: bool,
}

#[cfg(windows)]
unsafe extern "system" fn console_ctrl_handler(ctrl_type: u32) -> i32 {
    use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_C_EVENT};
    if matches!(
        ctrl_type,
        CTRL_C_EVENT | CTRL_BREAK_EVENT | CTRL_CLOSE_EVENT
    ) {
        MONITOR_INTERRUPTED.store(true, Ordering::SeqCst);
        1
    } else {
        0
    }
}

impl CtrlCHandlerGuard {
    fn install() -> Self {
        MONITOR_INTERRUPTED.store(false, Ordering::SeqCst);
        #[cfg(windows)]
        let registered = unsafe {
            windows_sys::Win32::System::Console::SetConsoleCtrlHandler(
                Some(console_ctrl_handler),
                1,
            ) != 0
        };
        Self {
            #[cfg(windows)]
            registered,
        }
    }
}

impl Drop for CtrlCHandlerGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        if self.registered {
            unsafe {
                windows_sys::Win32::System::Console::SetConsoleCtrlHandler(
                    Some(console_ctrl_handler),
                    0,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MonitorFormat {
    Table,
    Jsonl,
    Csv,
}

#[derive(Debug, Args)]
pub struct MonitorArgs {
    /// Polling interval in milliseconds (minimum 50 ms).
    #[arg(long, default_value_t = 500)]
    pub interval_ms: u64,
    /// Stop automatically after collecting N telemetry samples.
    #[arg(long)]
    pub count: Option<usize>,
    /// Stop automatically after S seconds of monitoring.
    #[arg(long)]
    pub duration_secs: Option<f64>,
    /// Filter telemetry by hex PCI device ID (e.g. 0x73AF or 73AF).
    #[arg(long)]
    pub device_id: Option<String>,
    /// Stream output format.
    #[arg(long, value_enum, default_value_t = MonitorFormat::Table)]
    pub format: MonitorFormat,
    /// Optional path to also save streamed monitor records.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum TuneSubcommand {
    /// Inspect ADLX GPU tuning capabilities, ranges, defaults, and active settings (read-only).
    Show {
        /// Filter report to a specific hex PCI device ID (e.g. 0x73AF or 73AF).
        #[arg(long)]
        device_id: Option<String>,
        /// Emit machine-readable JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
    /// Validate (dry-run by default) or apply an ADLX manual power limit percentage change.
    PowerLimit {
        /// Signed percentage offset within the hardware ADLX power limit range (e.g. -5 or 10).
        #[arg(allow_hyphen_values = true)]
        limit_pct: i32,
        /// Target hex PCI device ID (required if multiple tunable discrete GPUs exist).
        #[arg(long)]
        device_id: Option<String>,
        /// Explicitly commit the power limit via ADLX; without --apply only a safe dry-run executes.
        #[arg(long)]
        apply: bool,
        /// Emit machine-readable JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
    /// Validate (dry-run by default) or reset GPU tuning to ADLX factory defaults.
    Reset {
        /// Target hex PCI device ID (required if multiple tunable discrete GPUs exist).
        #[arg(long)]
        device_id: Option<String>,
        /// Explicitly commit ResetToFactory via ADLX; without --apply only a safe dry-run executes.
        #[arg(long)]
        apply: bool,
        /// Emit machine-readable JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
}

pub fn parse_device_id_arg(raw: Option<&str>) -> Result<Option<u32>> {
    match raw {
        Some(text) => parse_pci_device_id(text)
            .map(Some)
            .with_context(|| format!("invalid hex PCI device ID '{text}' (expected e.g. 0x73AF)")),
        None => Ok(None),
    }
}

fn select_monitor_sample(
    samples: Vec<GpuTelemetrySample>,
    target_id: Option<u32>,
) -> Result<GpuTelemetrySample> {
    if let Some(id) = target_id {
        samples
            .into_iter()
            .find(|s| s.pci_device_id == Some(id))
            .with_context(|| format!("no AMD GPU matching PCI device ID 0x{id:04X} found in ADLX"))
    } else {
        samples
            .iter()
            .find(|s| s.discrete)
            .cloned()
            .or_else(|| samples.into_iter().next())
            .context("ADLX returned no GPU telemetry samples")
    }
}

fn emit_stream_line(file_writer: &mut Option<BufWriter<File>>, line: &str) -> Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{line}")?;
    stdout.flush()?;
    if let Some(writer) = file_writer.as_mut() {
        writeln!(writer, "{line}")?;
        writer.flush()?;
    }
    Ok(())
}

fn collect_sample_warnings(seq: usize, sample: &GpuTelemetrySample) -> Vec<String> {
    let mut warnings = Vec::new();
    if seq == 1 {
        if let Some(warn) = check_competing_workload_warning(sample) {
            warnings.push(warn);
        }
    }
    if let (Some(used), Some(total)) = (sample.vram_used_mib, sample.total_vram_mib) {
        if let Some(warn) = check_vram_pressure_warning(f64::from(used), total) {
            warnings.push(warn);
        }
    }
    warnings
}

pub fn cmd_monitor(args: &MonitorArgs) -> Result<()> {
    if args.interval_ms < 50 {
        bail!(
            "--interval-ms must be at least 50 ms (got {})",
            args.interval_ms
        );
    }
    if args.count == Some(0) {
        bail!("--count must be greater than 0");
    }
    let duration_limit = match args.duration_secs {
        Some(secs) => {
            if !secs.is_finite() || secs <= 0.0 {
                bail!("--duration-secs must be a positive finite number (got {secs})");
            }
            Some(Duration::from_secs_f64(secs))
        }
        None => None,
    };
    let target_id = parse_device_id_arg(args.device_id.as_deref())?;

    let session = AdlxSession::open().context("failed to initialize ADLX telemetry session")?;
    let first = select_monitor_sample(session.read_all_samples()?, target_id)?;

    let mut out_writer = if let Some(ref path) = args.output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)
            .with_context(|| format!("failed to create output file '{}'", path.display()))?;
        Some(BufWriter::new(file))
    } else {
        None
    };

    let _ctrl_c_guard = CtrlCHandlerGuard::install();
    let start = Instant::now();
    let interval = Duration::from_millis(args.interval_ms);
    let mut samples = Vec::new();
    let mut warned_competing = false;
    let mut warned_vram = false;

    match args.format {
        MonitorFormat::Table => {
            let dev_hex = first
                .pci_device_id
                .map(|id| format!(" (0x{id:04X})"))
                .unwrap_or_default();
            println!(
                "RDNA FORGE LIVE MONITOR — {}{} | interval={} ms",
                first.gpu_name, dev_hex, args.interval_ms
            );
            println!("{TELEMETRY_TABLE_HEADER}");
            if let Some(ref mut w) = out_writer {
                writeln!(w, "{TELEMETRY_TABLE_HEADER}")?;
            }
        }
        MonitorFormat::Csv => emit_stream_line(&mut out_writer, TELEMETRY_CSV_HEADER)?,
        MonitorFormat::Jsonl => {}
    }

    let mut current_opt = Some(first);
    loop {
        let sample = match current_opt.take() {
            Some(s) => s,
            None => select_monitor_sample(session.read_all_samples()?, target_id)?,
        };
        let elapsed = start.elapsed();
        let elapsed_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
        let seq = samples.len() + 1;
        let sample_warnings = collect_sample_warnings(seq, &sample);

        match args.format {
            MonitorFormat::Table => {
                let row = format_sample_table_row(seq, elapsed.as_secs_f64(), &sample);
                emit_stream_line(&mut out_writer, &row)?;
                for warn in &sample_warnings {
                    let is_competing = warn.starts_with("Competing GPU workload");
                    if (is_competing && !warned_competing) || (!is_competing && !warned_vram) {
                        println!("  WARNING: {warn}");
                        if is_competing {
                            warned_competing = true;
                        } else {
                            warned_vram = true;
                        }
                    }
                }
            }
            MonitorFormat::Csv => {
                let row = format_sample_csv_row(seq, elapsed_ms, &sample);
                emit_stream_line(&mut out_writer, &row)?;
            }
            MonitorFormat::Jsonl => {
                let record = MonitorRecord {
                    seq,
                    elapsed_ms,
                    sample: sample.clone(),
                    warnings: sample_warnings,
                };
                let json = serde_json::to_string(&record)?;
                emit_stream_line(&mut out_writer, &json)?;
            }
        }

        samples.push(sample);

        if MONITOR_INTERRUPTED.load(Ordering::Relaxed) {
            break;
        }
        if args.count.is_some_and(|max| samples.len() >= max) {
            break;
        }
        if duration_limit.is_some_and(|limit| start.elapsed() >= limit) {
            break;
        }

        let target_tick = start + interval.saturating_mul(samples.len() as u32);
        let effective_deadline = match duration_limit {
            Some(limit) => target_tick.min(start + limit),
            None => target_tick,
        };

        while Instant::now() < effective_deadline {
            if MONITOR_INTERRUPTED.load(Ordering::Relaxed) {
                break;
            }
            let remaining = effective_deadline.saturating_duration_since(Instant::now());
            std::thread::sleep(remaining.min(Duration::from_millis(50)));
        }

        if MONITOR_INTERRUPTED.load(Ordering::Relaxed) {
            break;
        }
        if duration_limit.is_some_and(|limit| start.elapsed() >= limit) {
            break;
        }
    }

    if let Some(ref mut writer) = out_writer {
        writer.flush()?;
    }

    if let Some(summary) = summarize_samples(&samples, args.interval_ms, samples.first()) {
        match args.format {
            MonitorFormat::Table => {
                println!();
                let mut stdout = io::stdout().lock();
                print_telemetry_summary_to(&mut stdout, "ADLX Monitor Summary", &summary)?;
            }
            MonitorFormat::Csv | MonitorFormat::Jsonl => {
                let mut stderr = io::stderr().lock();
                print_telemetry_summary_to(&mut stderr, "ADLX Monitor Summary", &summary)?;
            }
        }
    }

    Ok(())
}

pub fn cmd_tune_show(device_id: Option<&str>, json: bool) -> Result<()> {
    let target_id = parse_device_id_arg(device_id)?;
    let mut reports = read_all_gpu_tuning().context("failed to read ADLX GPU tuning state")?;
    if let Some(id) = target_id {
        reports.retain(|r| r.pci_device_id == Some(id));
        if reports.is_empty() {
            bail!("no AMD GPU matching PCI device ID 0x{id:04X} found in ADLX");
        }
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
        return Ok(());
    }

    println!("RDNA FORGE ADLX TUNING (READ-ONLY)");
    for report in &reports {
        print_tuning_report(report);
    }
    Ok(())
}

fn fmt_opt_range<T: std::fmt::Display>(
    val: Option<T>,
    unit: &str,
    range: Option<forge_profiler::IntRange>,
    default: Option<T>,
) -> String {
    let v = val
        .map(|x| format!("{x}{unit}"))
        .unwrap_or_else(|| "n/a".to_string());
    let mut extras = Vec::new();
    if let Some(r) = range {
        extras.push(format!("range {r}"));
    }
    if let Some(d) = default {
        extras.push(format!("default {d}{unit}"));
    }
    if extras.is_empty() {
        v
    } else {
        format!("{v} ({})", extras.join(", "))
    }
}

fn print_tuning_report(r: &GpuTuningReport) {
    let role = if r.discrete { "discrete" } else { "integrated" };
    let dev_hex = r
        .pci_device_id
        .map(|id| format!(" (0x{id:04X})"))
        .unwrap_or_default();
    let factory = r
        .at_factory
        .map(|b| if b { "true" } else { "false" })
        .unwrap_or("unknown");

    println!();
    println!("{}{} [{role}] | at_factory={factory}", r.gpu_name, dev_hex);
    println!(
        "  Capabilities: auto={}, preset={}, gfx={}, vram={}, fan={}, power={}",
        r.capabilities.auto_tuning,
        r.capabilities.preset_tuning,
        r.capabilities.manual_gfx,
        r.capabilities.manual_vram,
        r.capabilities.manual_fan,
        r.capabilities.manual_power
    );

    if let Some(ref preset) = r.preset {
        let supported = if preset.supported_presets.is_empty() {
            "none".to_string()
        } else {
            preset.supported_presets.join(", ")
        };
        let current = preset.current_preset.as_deref().unwrap_or("custom/none");
        println!("  Preset: active={current} | supported=[{supported}]");
    }

    if let Some(ref pwr) = r.power {
        println!(
            "  Power Limit: {}",
            fmt_opt_range(
                pwr.power_limit_pct,
                "%",
                pwr.power_limit_range,
                pwr.power_limit_default_pct
            )
        );
        if pwr.tdc_limit_supported {
            println!(
                "  TDC Limit: {}",
                fmt_opt_range(
                    pwr.tdc_limit_a,
                    " A",
                    pwr.tdc_limit_range,
                    pwr.tdc_limit_default_a
                )
            );
        }
    }

    if let Some(ref gfx) = r.gfx {
        println!(
            "  GFX Min Clock: {}",
            fmt_opt_range(
                gfx.min_freq_mhz,
                " MHz",
                gfx.min_freq_range,
                gfx.min_freq_default_mhz
            )
        );
        println!(
            "  GFX Max Clock: {}",
            fmt_opt_range(
                gfx.max_freq_mhz,
                " MHz",
                gfx.max_freq_range,
                gfx.max_freq_default_mhz
            )
        );
        println!(
            "  GFX Voltage: {}",
            fmt_opt_range(
                gfx.voltage_mv,
                " mV",
                gfx.voltage_range,
                gfx.voltage_default_mv
            )
        );
    }

    if let Some(ref vram) = r.vram {
        println!(
            "  VRAM Max Clock: {}",
            fmt_opt_range(
                vram.max_freq_mhz,
                " MHz",
                vram.max_freq_range,
                vram.max_freq_default_mhz
            )
        );
        if vram.memory_timing_supported {
            let cur = vram
                .memory_timing
                .map(|m| m.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let sup: Vec<String> = vram
                .supported_memory_timings
                .iter()
                .map(ToString::to_string)
                .collect();
            println!(
                "  VRAM Memory Timing: {cur} (supported=[{}])",
                sup.join(", ")
            );
        }
    }

    if let Some(ref fan) = r.fan {
        let spd = fan
            .speed_range_pct
            .map(|x| x.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        let tmp = fan
            .temp_range_c
            .map(|x| x.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        println!("  Fan Ranges: speed={spd} %, temp={tmp} C");
        if fan.zero_rpm_supported {
            println!(
                "  Zero RPM: enabled={} (default={})",
                fan.zero_rpm_enabled
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                fan.zero_rpm_default
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            );
        }
        if let Some(v) = fan.min_acoustic_limit_mhz {
            println!(
                "  Fan Min Acoustic Limit: {}",
                fmt_opt_range(
                    Some(v),
                    " MHz",
                    fan.min_acoustic_limit_range,
                    fan.min_acoustic_limit_default_mhz
                )
            );
        }
        if let Some(v) = fan.min_fan_speed_pct {
            println!(
                "  Fan Min Speed: {}",
                fmt_opt_range(
                    Some(v),
                    "%",
                    fan.min_fan_speed_range,
                    fan.min_fan_speed_default_pct
                )
            );
        }
        if let Some(v) = fan.target_fan_speed_pct {
            println!(
                "  Fan Target Speed: {}",
                fmt_opt_range(
                    Some(v),
                    "%",
                    fan.target_fan_speed_range,
                    fan.target_fan_speed_default_pct
                )
            );
        }
        if !fan.fan_curve.is_empty() {
            let pts: Vec<String> = fan
                .fan_curve
                .iter()
                .map(|p| format!("{}C->{}%", p.temp_c, p.speed_pct))
                .collect();
            println!("  Fan Curve (Active ADLX): {}", pts.join(", "));
        }
        if !fan.default_fan_curve.is_empty() {
            let dpts: Vec<String> = fan
                .default_fan_curve
                .iter()
                .map(|p| format!("{}C->{}%", p.temp_c, p.speed_pct))
                .collect();
            println!("  Fan Curve (Factory VBIOS): {}", dpts.join(", "));
        }
        if let Some(ref note) = fan.fan_note {
            println!("  Fan Diagnostic: {note}");
        }
    }
}

pub fn cmd_tune_power_limit(
    limit_pct: i32,
    device_id: Option<&str>,
    apply: bool,
    json: bool,
) -> Result<()> {
    let target_id = parse_device_id_arg(device_id)?;
    let res = set_gpu_power_limit(target_id, limit_pct, !apply)
        .context("ADLX power limit operation failed")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&res)?);
    } else {
        let mode = if apply {
            res.status.to_string()
        } else {
            "DRY-RUN".to_string()
        };
        let dev_hex = res
            .pci_device_id
            .map(|id| format!(" (0x{id:04X})"))
            .unwrap_or_default();
        println!("[{mode}] {}{} Power Limit", res.gpu_name, dev_hex);
        println!("  Previous: {}%", res.previous_limit_pct);
        println!("  Requested: {}%", res.requested_limit_pct);
        println!(
            "  Allowed Range: {} (default={})",
            res.allowed_range,
            res.default_limit_pct
                .map(|d| format!("{d}%"))
                .unwrap_or_else(|| "n/a".to_string())
        );
        if let Some(verified) = res.verified_limit_pct {
            println!("  Read-Back: {verified}%");
        }
        if !apply {
            println!("  No GPU setting was modified. Pass --apply to commit via ADLX.");
        } else {
            println!("  Operation: {}", res.status);
            println!("  Rollback:  {}", res.rollback);
            if let Some(v) = res.rollback_verified_limit_pct {
                println!("  Rollback read-back: {v}%");
            }
        }
        if let Some(err) = &res.error {
            println!("  Detail: {err}");
        }
    }

    // A requested change that is not confirmed is a failure, whatever the driver returned.
    if apply && !res.succeeded() {
        anyhow::bail!(
            "power limit change {} ({}); verify the GPU state before continuing",
            res.status,
            res.rollback
        );
    }
    Ok(())
}

pub fn cmd_tune_reset(device_id: Option<&str>, apply: bool, json: bool) -> Result<()> {
    let target_id = parse_device_id_arg(device_id)?;
    let res = reset_gpu_tuning_to_factory(target_id, !apply)
        .context("ADLX factory tuning reset failed")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&res)?);
    } else {
        let mode = if apply {
            res.status.to_string()
        } else {
            "DRY-RUN".to_string()
        };
        let dev_hex = res
            .pci_device_id
            .map(|id| format!(" (0x{id:04X})"))
            .unwrap_or_default();
        println!("[{mode}] {}{} ResetToFactory", res.gpu_name, dev_hex);
        println!(
            "  Was At Factory: {}",
            res.was_at_factory
                .map(|b| b.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        println!(
            "  Now At Factory: {}",
            res.now_at_factory
                .map(|b| b.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        if !apply {
            println!(
                "  No GPU setting was modified. Pass --apply to reset all ADLX tuning profiles."
            );
        } else {
            println!("  Operation: {}", res.status);
            println!(
                "  Rollback:  {} (a factory reset discards the previous tuning profile; it cannot be restored automatically)",
                res.rollback
            );
        }
        if let Some(err) = &res.error {
            println!("  Detail: {err}");
        }
    }

    if apply && !res.succeeded() {
        anyhow::bail!(
            "factory reset {}; verify the GPU state before continuing",
            res.status
        );
    }
    Ok(())
}
