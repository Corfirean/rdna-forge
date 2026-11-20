use std::fmt;

use forge_core::{ForgeError, ForgeResult};
use serde::{Deserialize, Serialize};

/// Discrete integer tuning range reported by ADLX (`min..=max` with allowed `step`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntRange {
    pub min: i32,
    pub max: i32,
    pub step: i32,
}

impl IntRange {
    /// Returns `Some(IntRange)` only when `min <= max` and `step > 0`.
    pub fn from_adlx(min: i32, max: i32, step: i32) -> Option<Self> {
        let range = Self { min, max, step };
        range.is_valid().then_some(range)
    }

    /// Returns `true` if this range defines at least one valid step-aligned value.
    pub fn is_valid(&self) -> bool {
        self.min <= self.max && self.step > 0
    }

    /// Validates that `value` lies within `min..=max` and aligns to `(value - min) % step == 0`.
    pub fn validate(&self, value: i32, name: &str) -> ForgeResult<()> {
        if !self.is_valid() {
            return Err(ForgeError::InvalidParameter(format!(
                "{name} hardware range is invalid (min={}, max={}, step={})",
                self.min, self.max, self.step
            )));
        }
        if value < self.min || value > self.max {
            return Err(ForgeError::InvalidParameter(format!(
                "{name}={value} is outside hardware range {}..={} (step {})",
                self.min, self.max, self.step
            )));
        }
        let offset = i64::from(value) - i64::from(self.min);
        let step = i64::from(self.step);
        if offset % step != 0 {
            return Err(ForgeError::InvalidParameter(format!(
                "{name}={value} is not aligned to step {} from minimum {} (allowed {}..={})",
                self.step, self.min, self.min, self.max
            )));
        }
        Ok(())
    }
}

impl fmt::Display for IntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..={} (step {})", self.min, self.max, self.step)
    }
}

/// ADLX `ADLX_MEMORYTIMING_DESCRIPTION` enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTimingMode {
    Default,
    FastTiming,
    FastTimingLevel2,
    Automatic,
    Level1,
    Level2,
    Unknown(i32),
}

impl MemoryTimingMode {
    /// Maps a raw ADLX `ADLX_MEMORYTIMING_DESCRIPTION` integer discriminant.
    pub fn from_adlx(code: i32) -> Self {
        match code {
            0 => Self::Default,
            1 => Self::FastTiming,
            2 => Self::FastTimingLevel2,
            3 => Self::Automatic,
            4 => Self::Level1,
            5 => Self::Level2,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for MemoryTimingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "Default"),
            Self::FastTiming => write!(f, "Fast Timing"),
            Self::FastTimingLevel2 => write!(f, "Fast Timing Level 2"),
            Self::Automatic => write!(f, "Automatic"),
            Self::Level1 => write!(f, "Memory Timing Level 1"),
            Self::Level2 => write!(f, "Memory Timing Level 2"),
            Self::Unknown(code) => write!(f, "Unknown({code})"),
        }
    }
}

/// GPU tuning feature flags queried from `IADLXGPUTuningServices`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningCapabilities {
    pub auto_tuning: bool,
    pub preset_tuning: bool,
    pub manual_gfx: bool,
    pub manual_vram: bool,
    pub manual_fan: bool,
    pub manual_power: bool,
}

impl TuningCapabilities {
    /// Returns `true` if ADLX reports at least one supported tuning mode on this GPU.
    pub fn any_supported(&self) -> bool {
        self.auto_tuning
            || self.preset_tuning
            || self.manual_gfx
            || self.manual_vram
            || self.manual_fan
            || self.manual_power
    }
}

/// ADLX GPU preset tuning support and active preset selection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetTuningInfo {
    pub supported_presets: Vec<String>,
    pub current_preset: Option<String>,
}

/// Read-only power tuning state from `IADLXManualPowerTuning` / `IADLXManualPowerTuning1`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerTuningInfo {
    pub power_limit_pct: Option<i32>,
    pub power_limit_range: Option<IntRange>,
    pub power_limit_default_pct: Option<i32>,
    pub tdc_limit_supported: bool,
    pub tdc_limit_a: Option<i32>,
    pub tdc_limit_range: Option<IntRange>,
    pub tdc_limit_default_a: Option<i32>,
}

/// Read-only manual GFX frequency and voltage tuning state (`IADLXManualGraphicsTuning2(_1)`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GfxTuningInfo {
    pub min_freq_mhz: Option<i32>,
    pub min_freq_range: Option<IntRange>,
    pub min_freq_default_mhz: Option<i32>,
    pub max_freq_mhz: Option<i32>,
    pub max_freq_range: Option<IntRange>,
    pub max_freq_default_mhz: Option<i32>,
    pub voltage_mv: Option<i32>,
    pub voltage_range: Option<IntRange>,
    pub voltage_default_mv: Option<i32>,
}

/// Read-only manual VRAM tuning state (`IADLXManualVRAMTuning2(_1)`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramTuningInfo {
    pub max_freq_mhz: Option<i32>,
    pub max_freq_range: Option<IntRange>,
    pub max_freq_default_mhz: Option<i32>,
    pub memory_timing_supported: bool,
    pub memory_timing: Option<MemoryTimingMode>,
    pub supported_memory_timings: Vec<MemoryTimingMode>,
}

/// Single point on the manual fan temperature-vs-speed curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FanCurvePoint {
    pub temp_c: i32,
    pub speed_pct: i32,
}

/// Read-only manual fan tuning state (`IADLXManualFanTuning` / `IADLXManualFanTuning1`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FanTuningInfo {
    pub speed_range_pct: Option<IntRange>,
    pub temp_range_c: Option<IntRange>,
    pub zero_rpm_supported: bool,
    pub zero_rpm_enabled: Option<bool>,
    pub zero_rpm_default: Option<bool>,
    pub min_acoustic_limit_mhz: Option<i32>,
    pub min_acoustic_limit_range: Option<IntRange>,
    pub min_acoustic_limit_default_mhz: Option<i32>,
    pub min_fan_speed_pct: Option<i32>,
    pub min_fan_speed_range: Option<IntRange>,
    pub min_fan_speed_default_pct: Option<i32>,
    pub target_fan_speed_pct: Option<i32>,
    pub target_fan_speed_range: Option<IntRange>,
    pub target_fan_speed_default_pct: Option<i32>,
    pub fan_curve: Vec<FanCurvePoint>,
    #[serde(default)]
    pub default_fan_curve: Vec<FanCurvePoint>,
    #[serde(default)]
    pub flat_pwm_override_pct: Option<i32>,
    #[serde(default)]
    pub external_controller: Option<String>,
    #[serde(default)]
    pub fan_note: Option<String>,
}

/// Complete read-only ADLX tuning inspection report for a single AMD GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTuningReport {
    pub gpu_name: String,
    pub pci_device_id: Option<u32>,
    pub discrete: bool,
    pub at_factory: Option<bool>,
    pub capabilities: TuningCapabilities,
    pub preset: Option<PresetTuningInfo>,
    pub power: Option<PowerTuningInfo>,
    pub gfx: Option<GfxTuningInfo>,
    pub vram: Option<VramTuningInfo>,
    pub fan: Option<FanTuningInfo>,
}

/// Outcome of validating (dry-run) or applying a GPU power limit change via ADLX.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerLimitChangeResult {
    pub gpu_name: String,
    pub pci_device_id: Option<u32>,
    pub previous_limit_pct: i32,
    pub requested_limit_pct: i32,
    pub allowed_range: IntRange,
    pub default_limit_pct: Option<i32>,
    /// The driver accepted the change request (`SetPowerLimit` returned OK). Not proof of state.
    pub applied: bool,
    /// Value read back from the driver after the request, if the read succeeded.
    pub verified_limit_pct: Option<i32>,
    /// True only when the read-back value equals the requested value.
    #[serde(default)]
    pub confirmed: bool,
}

/// Outcome of validating (dry-run) or executing an ADLX factory tuning reset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningResetResult {
    pub gpu_name: String,
    pub pci_device_id: Option<u32>,
    pub was_at_factory: Option<bool>,
    /// The driver accepted the reset request (`ResetToFactory` returned OK). Not proof of state.
    pub applied: bool,
    /// State read back after the request, if the read succeeded.
    pub now_at_factory: Option<bool>,
    /// True only when the read-back reports the GPU is at factory settings.
    #[serde(default)]
    pub confirmed: bool,
}

/// Interprets the read-back after a power-limit request.
///
/// `Ok(true)` = confirmed, `Ok(false)` = read-back unavailable (outcome unknown, must be shown
/// as unconfirmed), `Err` = the driver reports a different value, i.e. the change did not take
/// effect as requested.
pub fn check_power_limit_readback(
    requested_pct: i32,
    previous_pct: i32,
    readback: Option<i32>,
) -> ForgeResult<bool> {
    match readback {
        Some(v) if v == requested_pct => Ok(true),
        Some(v) => Err(ForgeError::ExecutionFailed(format!(
            "SetPowerLimit({requested_pct}%) was accepted but the driver now reports {v}% (previous value was {previous_pct}%); the GPU power limit is not as requested"
        ))),
        None => Ok(false),
    }
}

/// Interprets the read-back after a factory-reset request (see [`check_power_limit_readback`]).
pub fn check_reset_readback(now_at_factory: Option<bool>) -> ForgeResult<bool> {
    match now_at_factory {
        Some(true) => Ok(true),
        Some(false) => Err(ForgeError::ExecutionFailed(
            "ResetToFactory was accepted but ADLX still reports non-factory tuning; the reset is not confirmed"
                .to_string(),
        )),
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_limit_readback_distinguishes_confirmed_mismatch_and_unknown() {
        assert!(check_power_limit_readback(10, 0, Some(10)).unwrap());
        assert!(!check_power_limit_readback(10, 0, None).unwrap());
        let err = check_power_limit_readback(10, 0, Some(0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not as requested") && err.contains("previous value was 0%"));
    }

    #[test]
    fn reset_readback_distinguishes_confirmed_mismatch_and_unknown() {
        assert!(check_reset_readback(Some(true)).unwrap());
        assert!(!check_reset_readback(None).unwrap());
        assert!(check_reset_readback(Some(false)).is_err());
    }

    #[test]
    fn range_validation_rejects_out_of_range_and_misaligned_values() {
        let r = IntRange::from_adlx(-10, 10, 2).unwrap();
        assert!(r.validate(4, "x").is_ok());
        assert!(r.validate(11, "x").is_err());
        assert!(r.validate(-11, "x").is_err());
        assert!(r.validate(3, "x").is_err());
        assert!(IntRange::from_adlx(5, 1, 1).is_none());
        assert!(IntRange::from_adlx(0, 10, 0).is_none());
        // Extreme bounds must not overflow the alignment arithmetic.
        let wide = IntRange::from_adlx(i32::MIN, i32::MAX, 1).unwrap();
        assert!(wide.validate(i32::MAX, "x").is_ok());
    }
}
