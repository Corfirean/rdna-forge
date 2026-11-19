use serde::{Deserialize, Serialize};

/// Threshold percentage of total dedicated VRAM above which Windows WDDM contention is flagged.
pub const WDDM_VRAM_PRESSURE_PCT: f64 = 50.0;

/// Pre-run GPU utilization percentage above which competing GPU workloads are flagged.
pub const COMPETING_GPU_USAGE_PCT: f64 = 20.0;

/// Single point-in-time GPU telemetry snapshot from ADLX.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuTelemetrySample {
    pub gpu_name: String,
    pub pci_device_id: Option<u32>,
    pub discrete: bool,
    pub total_vram_mib: Option<u32>,
    pub timestamp_ms: Option<i64>,
    pub gpu_usage_pct: Option<f64>,
    pub gpu_clock_mhz: Option<i32>,
    pub vram_clock_mhz: Option<i32>,
    pub gpu_temp_c: Option<f64>,
    pub hotspot_temp_c: Option<f64>,
    pub gpu_power_w: Option<f64>,
    pub board_power_w: Option<f64>,
    pub fan_rpm: Option<i32>,
    pub vram_used_mib: Option<i32>,
    pub voltage_mv: Option<i32>,
}

/// Aggregated distribution summary for a single telemetry sensor over a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSummary {
    pub min: f64,
    pub median: f64,
    pub max: f64,
    pub samples: usize,
}

/// Aggregated ADLX telemetry and WDDM contention diagnostics collected over a workload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub gpu_name: String,
    pub pci_device_id: Option<u32>,
    pub total_vram_mib: Option<u32>,
    pub sample_count: usize,
    pub interval_ms: u64,
    pub gpu_usage_pct: Option<MetricSummary>,
    pub gpu_clock_mhz: Option<MetricSummary>,
    pub vram_clock_mhz: Option<MetricSummary>,
    pub gpu_temp_c: Option<MetricSummary>,
    pub hotspot_temp_c: Option<MetricSummary>,
    pub gpu_power_w: Option<MetricSummary>,
    pub board_power_w: Option<MetricSummary>,
    pub fan_rpm: Option<MetricSummary>,
    pub vram_used_mib: Option<MetricSummary>,
    pub voltage_mv: Option<MetricSummary>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Parses an ADLX PCI device ID string such as `"73AF"` or `"0x73AF"` into a `u32`.
pub fn parse_pci_device_id(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    let hex = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    if hex.is_empty() {
        return None;
    }
    u32::from_str_radix(hex, 16).ok()
}

/// Returns a Windows WDDM VRAM pressure warning if reported GPU VRAM usage exceeds `WDDM_VRAM_PRESSURE_PCT`.
pub fn check_vram_pressure_warning(used_mib: f64, total_mib: u32) -> Option<String> {
    if total_mib == 0 || !used_mib.is_finite() || used_mib <= 0.0 {
        return None;
    }
    let pct = (used_mib / f64::from(total_mib)) * 100.0;
    if pct > WDDM_VRAM_PRESSURE_PCT {
        Some(format!(
            "High WDDM VRAM occupancy ({used_mib:.0}/{total_mib} MiB, {pct:.1}%): \
             non-HIP desktop/compute allocations can force HIP WDDM eviction/thrashing even though hipMemGetInfo hides them"
        ))
    } else {
        None
    }
}

/// Returns a warning when a GPU already has significant competing activity prior to benchmarking.
pub fn check_competing_workload_warning(sample: &GpuTelemetrySample) -> Option<String> {
    let usage = sample.gpu_usage_pct?;
    if usage.is_finite() && usage > COMPETING_GPU_USAGE_PCT {
        Some(format!(
            "Competing GPU workload detected before benchmark start on {} (GPU usage {usage:.1}%)",
            sample.gpu_name
        ))
    } else {
        None
    }
}

fn summarize_metric<F>(samples: &[GpuTelemetrySample], extract: F) -> Option<MetricSummary>
where
    F: Fn(&GpuTelemetrySample) -> Option<f64>,
{
    let mut vals: Vec<f64> = samples
        .iter()
        .filter_map(extract)
        .filter(|v| v.is_finite())
        .collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    let median = if n.is_multiple_of(2) {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    };
    Some(MetricSummary {
        min: vals[0],
        median,
        max: vals[n - 1],
        samples: n,
    })
}

/// Computes per-sensor min/median/max statistics and WDDM health warnings from collected samples.
pub fn summarize_samples(
    samples: &[GpuTelemetrySample],
    interval_ms: u64,
    pre_run: Option<&GpuTelemetrySample>,
) -> Option<TelemetrySummary> {
    let first = samples.first()?;
    let vram_used_summary = summarize_metric(samples, |s| s.vram_used_mib.map(f64::from));

    let mut warnings = Vec::new();
    if let Some(pre) = pre_run {
        if let Some(w) = check_competing_workload_warning(pre) {
            warnings.push(w);
        }
    }

    if let Some(total_mib) = first.total_vram_mib {
        let peak_used = vram_used_summary
            .as_ref()
            .map(|m| m.max)
            .or_else(|| pre_run.and_then(|s| s.vram_used_mib.map(f64::from)));
        if let Some(used_mib) = peak_used {
            if let Some(w) = check_vram_pressure_warning(used_mib, total_mib) {
                warnings.push(w);
            }
        }
    }

    Some(TelemetrySummary {
        gpu_name: first.gpu_name.clone(),
        pci_device_id: first.pci_device_id,
        total_vram_mib: first.total_vram_mib,
        sample_count: samples.len(),
        interval_ms,
        gpu_usage_pct: summarize_metric(samples, |s| s.gpu_usage_pct),
        gpu_clock_mhz: summarize_metric(samples, |s| s.gpu_clock_mhz.map(f64::from)),
        vram_clock_mhz: summarize_metric(samples, |s| s.vram_clock_mhz.map(f64::from)),
        gpu_temp_c: summarize_metric(samples, |s| s.gpu_temp_c),
        hotspot_temp_c: summarize_metric(samples, |s| s.hotspot_temp_c),
        gpu_power_w: summarize_metric(samples, |s| s.gpu_power_w),
        board_power_w: summarize_metric(samples, |s| s.board_power_w),
        fan_rpm: summarize_metric(samples, |s| s.fan_rpm.map(f64::from)),
        vram_used_mib: vram_used_summary,
        voltage_mv: summarize_metric(samples, |s| s.voltage_mv.map(f64::from)),
        warnings,
    })
}

/// CSV column header emitted by `forge monitor --format csv`.
pub const TELEMETRY_CSV_HEADER: &str = "seq,elapsed_ms,timestamp_ms,gpu_name,pci_device_id,gpu_usage_pct,gpu_clock_mhz,vram_clock_mhz,gpu_temp_c,hotspot_temp_c,gpu_power_w,board_power_w,fan_rpm,vram_used_mib,total_vram_mib,voltage_mv";

/// Fixed-width table header emitted by `forge monitor --format table`.
pub const TELEMETRY_TABLE_HEADER: &str = " SEQ  TIME(s)   UTIL%  GFX(MHz)  MEM(MHz)  EDGE(C)   HOT(C)   PWR(W)  FAN(RPM)  VOLT(mV)        VRAM(MiB)";

/// Single JSONL record emitted by `forge monitor --format jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitorRecord {
    pub seq: usize,
    pub elapsed_ms: u64,
    #[serde(flatten)]
    pub sample: GpuTelemetrySample,
    pub warnings: Vec<String>,
}

fn escape_csv_field(raw: &str) -> String {
    if raw.contains([',', '"', '\n', '\r']) {
        let escaped = raw.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        raw.to_string()
    }
}

fn csv_opt<T: std::fmt::Display>(val: Option<T>) -> String {
    val.map(|v| v.to_string()).unwrap_or_default()
}

fn csv_opt_f64(val: Option<f64>) -> String {
    val.map(|v| format!("{v:.1}")).unwrap_or_default()
}

/// Formats a single telemetry sample as an RFC 4180 CSV line matching `TELEMETRY_CSV_HEADER`.
pub fn format_sample_csv_row(seq: usize, elapsed_ms: u64, sample: &GpuTelemetrySample) -> String {
    let dev_hex = sample
        .pci_device_id
        .map(|id| format!("0x{id:04X}"))
        .unwrap_or_default();
    [
        seq.to_string(),
        elapsed_ms.to_string(),
        csv_opt(sample.timestamp_ms),
        escape_csv_field(&sample.gpu_name),
        dev_hex,
        csv_opt_f64(sample.gpu_usage_pct),
        csv_opt(sample.gpu_clock_mhz),
        csv_opt(sample.vram_clock_mhz),
        csv_opt_f64(sample.gpu_temp_c),
        csv_opt_f64(sample.hotspot_temp_c),
        csv_opt_f64(sample.gpu_power_w),
        csv_opt_f64(sample.board_power_w),
        csv_opt(sample.fan_rpm),
        csv_opt(sample.vram_used_mib),
        csv_opt(sample.total_vram_mib),
        csv_opt(sample.voltage_mv),
    ]
    .join(",")
}

/// Formats a single telemetry sample as a fixed-width row matching `TELEMETRY_TABLE_HEADER`.
pub fn format_sample_table_row(
    seq: usize,
    elapsed_secs: f64,
    sample: &GpuTelemetrySample,
) -> String {
    let util = sample
        .gpu_usage_pct
        .map(|v| format!("{v:.1}%"))
        .unwrap_or_else(|| "n/a".to_string());
    let gfx = sample
        .gpu_clock_mhz
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let mem = sample
        .vram_clock_mhz
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let edge = sample
        .gpu_temp_c
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "n/a".to_string());
    let hot = sample
        .hotspot_temp_c
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "n/a".to_string());
    let pwr = sample
        .gpu_power_w
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "n/a".to_string());
    let fan = sample
        .fan_rpm
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let volt = sample
        .voltage_mv
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let vram = match (sample.vram_used_mib, sample.total_vram_mib) {
        (Some(u), Some(t)) => format!("{u}/{t}"),
        (Some(u), None) => format!("{u}"),
        _ => "n/a".to_string(),
    };

    format!(
        "{seq:>4}  {elapsed_secs:>7.2}  {util:>6}  {gfx:>8}  {mem:>8}  {edge:>7}  {hot:>7}  {pwr:>7}  {fan:>8}  {volt:>8}  {vram:>15}"
    )
}
