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

// ---------------------------------------------------------------------------------------------
// ADLX result codes (ADLXDefines.h `ADLX_RESULT`, a plain C enum, 32-bit on MSVC).
// Values 0..=10 are explicit in the header; 11..=18 follow by C sequential numbering.
// ---------------------------------------------------------------------------------------------

/// `ADLX_SUCCEEDED(x)`: `ADLX_OK`, `ADLX_ALREADY_ENABLED` or `ADLX_ALREADY_INITIALIZED`.
pub fn adlx_succeeded(rc: i32) -> bool {
    matches!(rc, 0..=2)
}

/// Human-readable `ADLX_RESULT` name, e.g. `ADLX_RESET_NEEDED (18)`.
pub fn adlx_result_name(rc: i32) -> String {
    const NAMES: [&str; 19] = [
        "ADLX_OK",
        "ADLX_ALREADY_ENABLED",
        "ADLX_ALREADY_INITIALIZED",
        "ADLX_FAIL",
        "ADLX_INVALID_ARGS",
        "ADLX_BAD_VER",
        "ADLX_UNKNOWN_INTERFACE",
        "ADLX_TERMINATED",
        "ADLX_ADL_INIT_ERROR",
        "ADLX_NOT_FOUND",
        "ADLX_INVALID_OBJECT",
        "ADLX_ORPHAN_OBJECTS",
        "ADLX_NOT_SUPPORTED",
        "ADLX_PENDING_OPERATION",
        "ADLX_GPU_INACTIVE",
        "ADLX_GPU_IN_USE",
        "ADLX_TIMEOUT_OPERATION",
        "ADLX_NOT_ACTIVE",
        "ADLX_RESET_NEEDED",
    ];
    match usize::try_from(rc).ok().and_then(|i| NAMES.get(i)) {
        Some(name) => format!("{name} ({rc})"),
        None => format!("ADLX_RESULT({rc})"),
    }
}

// ---------------------------------------------------------------------------------------------
// Change / rollback state machine
// ---------------------------------------------------------------------------------------------

/// What happened to a requested change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    /// No change was made (dry run, nothing to do, or the driver rejected the request).
    #[default]
    NotApplied,
    /// The driver accepted the request but the new state could not be read back.
    Applied,
    /// The driver accepted the request and the read-back matches it.
    Confirmed,
    /// The driver accepted the request but the read-back contradicts it.
    NotConfirmed,
}

impl fmt::Display for OperationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotApplied => "NOT APPLIED",
            Self::Applied => "APPLIED (UNVERIFIED)",
            Self::Confirmed => "CONFIRMED",
            Self::NotConfirmed => "NOT CONFIRMED",
        })
    }
}

/// What happened to the attempt to restore the previous value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollbackStatus {
    /// No restore was needed (nothing changed, or the change is not known to have failed).
    #[default]
    NotNeeded,
    /// This operation cannot be restored automatically.
    Unsupported,
    /// A restore request was issued but its result could not be read back.
    Attempted,
    /// The previous value was restored and read back.
    Confirmed,
    /// The restore request failed or the read-back still differs from the previous value.
    Failed,
}

impl fmt::Display for RollbackStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotNeeded => "ROLLBACK NOT NEEDED",
            Self::Unsupported => "ROLLBACK UNSUPPORTED",
            Self::Attempted => "ROLLBACK ATTEMPTED (UNVERIFIED)",
            Self::Confirmed => "ROLLBACK CONFIRMED",
            Self::Failed => "ROLLBACK FAILED",
        })
    }
}

/// Minimal hardware surface needed to change and restore a power limit. The real implementation
/// talks to ADLX; tests use a scripted fake, so the state machine is verified without a GPU.
pub trait PowerLimitControl {
    /// Reads the current limit (percent), or `None` if it cannot be read.
    fn read(&mut self) -> Option<i32>;
    /// Requests a new limit. `Err` carries the raw `ADLX_RESULT`.
    fn write(&mut self, value_pct: i32) -> Result<(), i32>;
}

/// Result of [`change_power_limit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerLimitOutcome {
    pub previous_pct: i32,
    pub requested_pct: i32,
    pub status: OperationStatus,
    pub verified_pct: Option<i32>,
    pub rollback: RollbackStatus,
    pub rollback_verified_pct: Option<i32>,
    pub error: Option<String>,
}

/// Validates, applies, verifies and (on a contradicting result) restores a power limit.
///
/// Order: read the original value -> validate the request against the hardware range ->
/// (dry run stops here) -> write -> read back -> compare -> if the read-back contradicts the
/// request, or a rejected write left the value changed or unknown, write the original value back
/// and read it back again. A pre-validation failure returns `Err` and the hardware is never
/// written. An unreadable read-back after an accepted write is reported as
/// [`OperationStatus::Applied`] (unverified) and is *not* rolled back, because nothing shows that
/// the change failed.
pub fn change_power_limit(
    ctl: &mut dyn PowerLimitControl,
    range: &IntRange,
    requested_pct: i32,
    dry_run: bool,
) -> ForgeResult<PowerLimitOutcome> {
    let previous = ctl.read().ok_or_else(|| {
        ForgeError::ExecutionFailed(
            "cannot read the current power limit; refusing to change it without a restore point"
                .to_string(),
        )
    })?;
    range.validate(requested_pct, "power_limit_pct")?;

    let mut out = PowerLimitOutcome {
        previous_pct: previous,
        requested_pct,
        status: OperationStatus::NotApplied,
        verified_pct: None,
        rollback: RollbackStatus::NotNeeded,
        rollback_verified_pct: None,
        error: None,
    };
    if dry_run || requested_pct == previous {
        return Ok(out);
    }

    let needs_rollback = match ctl.write(requested_pct) {
        Err(rc) => {
            let after = ctl.read();
            out.verified_pct = after;
            out.error = Some(format!(
                "SetPowerLimit({requested_pct}%) was rejected: {}",
                adlx_result_name(rc)
            ));
            if after == Some(previous) {
                return Ok(out); // rejected, state unchanged: NOT APPLIED
            }
            out.status = OperationStatus::NotConfirmed;
            true // value changed despite the error, or its state is unknown
        }
        Ok(()) => match ctl.read() {
            Some(v) if v == requested_pct => {
                out.status = OperationStatus::Confirmed;
                out.verified_pct = Some(v);
                false
            }
            Some(v) => {
                out.status = OperationStatus::NotConfirmed;
                out.verified_pct = Some(v);
                out.error = Some(format!(
                    "SetPowerLimit({requested_pct}%) was accepted but the driver reports {v}%"
                ));
                true
            }
            None => {
                out.status = OperationStatus::Applied;
                out.error = Some(
                    "the driver accepted the request but the value could not be read back"
                        .to_string(),
                );
                false
            }
        },
    };

    if needs_rollback {
        match ctl.write(previous) {
            Err(rc) => {
                let after = ctl.read();
                out.rollback_verified_pct = after;
                out.rollback = if after == Some(previous) {
                    RollbackStatus::Confirmed
                } else {
                    RollbackStatus::Failed
                };
                if out.rollback == RollbackStatus::Failed {
                    append_error(
                        &mut out.error,
                        format!(
                            "restoring {previous}% failed: {}; the GPU may still be at a changed power limit",
                            adlx_result_name(rc)
                        ),
                    );
                }
            }
            Ok(()) => {
                let after = ctl.read();
                out.rollback_verified_pct = after;
                out.rollback = match after {
                    Some(v) if v == previous => RollbackStatus::Confirmed,
                    Some(v) => {
                        append_error(
                            &mut out.error,
                            format!(
                                "restore to {previous}% was accepted but the driver reports {v}%"
                            ),
                        );
                        RollbackStatus::Failed
                    }
                    None => RollbackStatus::Attempted,
                };
            }
        }
    }
    Ok(out)
}

fn append_error(slot: &mut Option<String>, msg: String) {
    *slot = Some(match slot.take() {
        Some(existing) => format!("{existing}; {msg}"),
        None => msg,
    });
}

/// Minimal hardware surface for a factory reset.
pub trait FactoryResetControl {
    fn is_at_factory(&mut self) -> Option<bool>;
    /// Requests the reset. `Err` carries the raw `ADLX_RESULT`.
    fn reset(&mut self) -> Result<(), i32>;
}

/// Result of [`reset_to_factory`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResetOutcome {
    pub was_at_factory: Option<bool>,
    pub now_at_factory: Option<bool>,
    pub status: OperationStatus,
    pub rollback: RollbackStatus,
    pub error: Option<String>,
}

/// Requests a factory reset and verifies it by reading `IsAtFactory` back.
///
/// A reset discards the user's tuning profile and the previous profile is not captured, so there
/// is nothing to restore: `rollback` is always [`RollbackStatus::Unsupported`] when a reset was
/// requested. Dry runs and "already at factory" never write.
///
/// The state is read back after the call whether it returned success or an error, because an
/// error does not prove that nothing changed. An error with a factory read-back is reported as
/// `Confirmed` (with the error text), an error with a non-factory or unreadable state as
/// `NotConfirmed`; `NotApplied` is reserved for dry runs and "already at factory".
pub fn reset_to_factory(
    ctl: &mut dyn FactoryResetControl,
    dry_run: bool,
) -> ForgeResult<ResetOutcome> {
    let was = ctl.is_at_factory();
    let mut out = ResetOutcome {
        was_at_factory: was,
        now_at_factory: was,
        status: OperationStatus::NotApplied,
        rollback: RollbackStatus::NotNeeded,
        error: None,
    };
    if dry_run || was == Some(true) {
        return Ok(out);
    }
    out.rollback = RollbackStatus::Unsupported;

    // An error return does not prove the driver left the state untouched (a partial or late
    // reset is possible), so the state is read back after the call whether it succeeded or not and
    // the result reflects what ADLX reports.
    let call = ctl.reset();
    out.now_at_factory = ctl.is_at_factory();
    let rejected = call
        .err()
        .map(|rc| format!("ResetToFactory returned {}", adlx_result_name(rc)));
    match (out.now_at_factory, rejected) {
        (Some(true), None) => out.status = OperationStatus::Confirmed,
        (Some(true), Some(why)) => {
            // The call reported an error but ADLX now reports factory tuning: report the state.
            out.status = OperationStatus::Confirmed;
            out.error = Some(format!(
                "{why}, but ADLX reports the GPU is now at factory settings"
            ));
        }
        (Some(false), None) => {
            out.status = OperationStatus::NotConfirmed;
            out.error = Some(
                "ResetToFactory was accepted but ADLX still reports non-factory tuning".to_string(),
            );
        }
        (Some(false), Some(why)) => {
            out.status = OperationStatus::NotConfirmed;
            out.error = Some(format!(
                "{why} and ADLX still reports non-factory tuning; the reset is not confirmed"
            ));
        }
        (None, None) => {
            out.status = OperationStatus::Applied;
            out.error = Some(
                "the reset was accepted but the factory state could not be read back".to_string(),
            );
        }
        (None, Some(why)) => {
            // Neither "applied" nor "not applied" can be claimed: the state is unknown.
            out.status = OperationStatus::NotConfirmed;
            out.error = Some(format!(
                "{why} and the factory state could not be read back; whether the reset took effect is unknown"
            ));
        }
    }
    Ok(out)
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
    /// The driver accepted a change request (`status` is not `NotApplied`).
    pub applied: bool,
    /// Value read back after the request, if the read succeeded.
    pub verified_limit_pct: Option<i32>,
    /// True only when the read-back equals the requested value.
    #[serde(default)]
    pub confirmed: bool,
    #[serde(default)]
    pub status: OperationStatus,
    #[serde(default)]
    pub rollback: RollbackStatus,
    /// Value read back after a rollback attempt, if any.
    #[serde(default)]
    pub rollback_verified_limit_pct: Option<i32>,
    #[serde(default)]
    pub error: Option<String>,
}

impl PowerLimitChangeResult {
    /// True when the outcome is what the caller asked for: a confirmed change, or a dry run /
    /// no-op that did not fail. Everything else must be reported as a failure.
    pub fn succeeded(&self) -> bool {
        match self.status {
            OperationStatus::Confirmed => true,
            OperationStatus::NotApplied => self.error.is_none(),
            OperationStatus::Applied | OperationStatus::NotConfirmed => false,
        }
    }
}

/// Outcome of validating (dry-run) or executing an ADLX factory tuning reset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningResetResult {
    pub gpu_name: String,
    pub pci_device_id: Option<u32>,
    pub was_at_factory: Option<bool>,
    /// The driver accepted a reset request (`status` is not `NotApplied`).
    pub applied: bool,
    /// State read back after the request, if the read succeeded.
    pub now_at_factory: Option<bool>,
    /// True only when the read-back reports the GPU is at factory settings.
    #[serde(default)]
    pub confirmed: bool,
    #[serde(default)]
    pub status: OperationStatus,
    #[serde(default)]
    pub rollback: RollbackStatus,
    #[serde(default)]
    pub error: Option<String>,
}

impl TuningResetResult {
    /// See [`PowerLimitChangeResult::succeeded`].
    pub fn succeeded(&self) -> bool {
        match self.status {
            OperationStatus::Confirmed => true,
            OperationStatus::NotApplied => self.error.is_none(),
            OperationStatus::Applied | OperationStatus::NotConfirmed => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted fake: models a power-limit register with configurable misbehavior.
    struct FakePower {
        value: Option<i32>,
        /// Number of `write` calls and the values written, in order.
        writes: Vec<i32>,
        /// Per-write behavior, consumed in order; `Honor` once exhausted.
        script: Vec<Write>,
        /// When true, `read` returns None (sensor/read-back unavailable).
        unreadable: bool,
        /// Reads after this many `write` calls are unavailable (None); `usize::MAX` = never.
        unreadable_after_writes: usize,
    }

    #[derive(Clone, Copy)]
    enum Write {
        /// Accept and store the value.
        Honor,
        /// Return OK but silently keep the old value (driver ignored it).
        Ignore,
        /// Return OK and store a different value.
        StoreOther(i32),
        /// Return this error and keep the old value.
        Reject(i32),
        /// Return this error but store the value anyway.
        RejectButStore(i32),
    }

    impl FakePower {
        fn new(value: i32, script: Vec<Write>) -> Self {
            Self {
                value: Some(value),
                writes: Vec::new(),
                script,
                unreadable: false,
                unreadable_after_writes: usize::MAX,
            }
        }
    }

    impl PowerLimitControl for FakePower {
        fn read(&mut self) -> Option<i32> {
            if self.unreadable || self.writes.len() >= self.unreadable_after_writes {
                return None;
            }
            self.value
        }
        fn write(&mut self, v: i32) -> Result<(), i32> {
            let step = if self.script.is_empty() {
                Write::Honor
            } else {
                self.script.remove(0)
            };
            self.writes.push(v);
            match step {
                Write::Honor => {
                    self.value = Some(v);
                    Ok(())
                }
                Write::Ignore => Ok(()),
                Write::StoreOther(o) => {
                    self.value = Some(o);
                    Ok(())
                }
                Write::Reject(rc) => Err(rc),
                Write::RejectButStore(rc) => {
                    self.value = Some(v);
                    Err(rc)
                }
            }
        }
    }

    fn range() -> IntRange {
        IntRange::from_adlx(-10, 12, 1).unwrap()
    }

    #[test]
    fn dry_run_never_writes_even_when_the_request_is_valid() {
        let mut f = FakePower::new(0, vec![]);
        let o = change_power_limit(&mut f, &range(), 5, true).unwrap();
        assert!(f.writes.is_empty());
        assert_eq!(o.status, OperationStatus::NotApplied);
        assert_eq!(o.rollback, RollbackStatus::NotNeeded);
        assert_eq!(f.value, Some(0));
    }

    #[test]
    fn invalid_or_unrestorable_requests_fail_before_any_write() {
        let mut f = FakePower::new(0, vec![]);
        assert!(change_power_limit(&mut f, &range(), 13, false).is_err()); // out of range
        assert!(change_power_limit(&mut f, &range(), -11, false).is_err());
        assert!(f.writes.is_empty());
        // Without a readable original value there is no restore point: refuse.
        let mut g = FakePower::new(0, vec![]);
        g.unreadable = true;
        let err = change_power_limit(&mut g, &range(), 5, false).unwrap_err();
        assert!(err.to_string().contains("restore point"));
        assert!(g.writes.is_empty());
    }

    #[test]
    fn confirmed_change_needs_no_rollback() {
        let mut f = FakePower::new(0, vec![]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(f.writes, vec![5]);
        assert_eq!(o.status, OperationStatus::Confirmed);
        assert_eq!(o.verified_pct, Some(5));
        assert_eq!(o.rollback, RollbackStatus::NotNeeded);
        assert!(o.error.is_none());
    }

    #[test]
    fn same_value_is_a_no_op_without_a_write() {
        let mut f = FakePower::new(3, vec![]);
        let o = change_power_limit(&mut f, &range(), 3, false).unwrap();
        assert!(f.writes.is_empty());
        assert_eq!(o.status, OperationStatus::NotApplied);
        assert!(o.error.is_none());
    }

    #[test]
    fn rejected_write_with_unchanged_state_is_not_applied_and_not_rolled_back() {
        let mut f = FakePower::new(0, vec![Write::Reject(3)]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(f.writes, vec![5]); // no restore write: state never changed
        assert_eq!(o.status, OperationStatus::NotApplied);
        assert_eq!(o.rollback, RollbackStatus::NotNeeded);
        assert!(o.error.as_deref().unwrap().contains("ADLX_FAIL (3)"));
    }

    #[test]
    fn contradicting_readback_triggers_a_confirmed_rollback() {
        // Driver says OK but keeps the old value (ignored) -> NotConfirmed, restore is a no-op write.
        let mut f = FakePower::new(0, vec![Write::Ignore]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(o.verified_pct, Some(0));
        assert_eq!(f.writes, vec![5, 0]);
        assert_eq!(o.rollback, RollbackStatus::Confirmed);
        assert_eq!(f.value, Some(0));

        // Driver stores a different value than requested -> restored to the original.
        let mut f = FakePower::new(0, vec![Write::StoreOther(9)]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(o.verified_pct, Some(9));
        assert_eq!(o.rollback, RollbackStatus::Confirmed);
        assert_eq!(o.rollback_verified_pct, Some(0));
        assert_eq!(f.value, Some(0));
    }

    #[test]
    fn write_error_that_still_changed_the_value_is_rolled_back() {
        let mut f = FakePower::new(0, vec![Write::RejectButStore(3)]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(f.writes, vec![5, 0]);
        assert_eq!(o.rollback, RollbackStatus::Confirmed);
        assert_eq!(f.value, Some(0));
        assert!(o.error.as_deref().unwrap().contains("rejected"));
    }

    #[test]
    fn failed_rollback_is_reported_as_failed_not_hidden() {
        // Change is ignored AND the restore write is rejected while the value is wrong.
        let mut f = FakePower::new(0, vec![Write::StoreOther(9), Write::Reject(12)]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(o.rollback, RollbackStatus::Failed);
        assert_eq!(o.rollback_verified_pct, Some(9));
        let e = o.error.unwrap();
        assert!(e.contains("restoring 0% failed") && e.contains("ADLX_NOT_SUPPORTED (12)"));

        // Restore accepted but the value stays wrong -> also FAILED.
        let mut f = FakePower::new(0, vec![Write::StoreOther(9), Write::StoreOther(7)]);
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.rollback, RollbackStatus::Failed);
        assert_eq!(o.rollback_verified_pct, Some(7));
    }

    #[test]
    fn rollback_that_cannot_be_read_back_is_attempted_not_confirmed() {
        // First read-back (after write 1) contradicts; reads become unavailable after the restore write.
        let mut f = FakePower::new(0, vec![Write::StoreOther(9)]);
        f.unreadable_after_writes = 2;
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(f.writes, vec![5, 0]);
        assert_eq!(o.rollback, RollbackStatus::Attempted);
        assert_eq!(o.rollback_verified_pct, None);
    }

    #[test]
    fn unreadable_readback_after_an_accepted_write_is_applied_unverified_without_rollback() {
        let mut f = FakePower::new(0, vec![]);
        f.unreadable_after_writes = 1;
        let o = change_power_limit(&mut f, &range(), 5, false).unwrap();
        assert_eq!(o.status, OperationStatus::Applied);
        assert_eq!(f.writes, vec![5]);
        assert_eq!(o.rollback, RollbackStatus::NotNeeded);
        assert!(o.error.unwrap().contains("could not be read back"));
    }

    /// Scripted fake factory-reset backend.
    struct FakeReset {
        at_factory: Option<bool>,
        resets: usize,
        /// What `reset()` returns.
        call: Result<(), i32>,
        /// Whether a `reset()` call (even an erroring one) actually moves the GPU to factory.
        applies: bool,
        /// When false, `is_at_factory()` is unreadable once `reset()` has been called.
        readable_after: bool,
    }

    impl FakeReset {
        fn new(call: Result<(), i32>, applies: bool, readable_after: bool) -> Self {
            Self {
                at_factory: Some(false),
                resets: 0,
                call,
                applies,
                readable_after,
            }
        }
    }

    impl FactoryResetControl for FakeReset {
        fn is_at_factory(&mut self) -> Option<bool> {
            if self.resets > 0 && !self.readable_after {
                return None;
            }
            self.at_factory
        }
        fn reset(&mut self) -> Result<(), i32> {
            self.resets += 1;
            if self.applies {
                self.at_factory = Some(true);
            }
            self.call
        }
    }

    #[test]
    fn reset_dry_run_and_already_at_factory_never_reset() {
        let mut f = FakeReset::new(Ok(()), true, true);
        let o = reset_to_factory(&mut f, true).unwrap();
        assert_eq!((f.resets, o.status), (0, OperationStatus::NotApplied));
        assert_eq!(o.rollback, RollbackStatus::NotNeeded);

        let mut g = FakeReset::new(Ok(()), true, true);
        g.at_factory = Some(true);
        let o = reset_to_factory(&mut g, false).unwrap();
        assert_eq!((g.resets, o.status), (0, OperationStatus::NotApplied));
        assert!(o.error.is_none());
    }

    #[test]
    fn reset_success_with_factory_readback_is_confirmed() {
        let mut f = FakeReset::new(Ok(()), true, true);
        let o = reset_to_factory(&mut f, false).unwrap();
        assert_eq!(f.resets, 1);
        assert_eq!(o.status, OperationStatus::Confirmed);
        assert_eq!(o.now_at_factory, Some(true));
        assert_eq!(o.rollback, RollbackStatus::Unsupported);
        assert!(o.error.is_none());
    }

    #[test]
    fn reset_success_but_state_contradicts_is_not_confirmed() {
        // The driver says OK but the GPU still reports non-factory tuning.
        let mut f = FakeReset::new(Ok(()), false, true);
        let o = reset_to_factory(&mut f, false).unwrap();
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(o.now_at_factory, Some(false));
        assert_eq!(o.rollback, RollbackStatus::Unsupported);
        assert!(o.error.unwrap().contains("still reports non-factory"));
    }

    #[test]
    fn reset_success_with_unreadable_state_is_applied_unverified() {
        let mut f = FakeReset::new(Ok(()), true, false);
        let o = reset_to_factory(&mut f, false).unwrap();
        assert_eq!(o.status, OperationStatus::Applied);
        assert_eq!(o.now_at_factory, None);
        assert_eq!(o.rollback, RollbackStatus::Unsupported);
    }

    #[test]
    fn reset_error_but_state_became_factory_reports_the_actual_state() {
        // ResetToFactory returned ADLX_FAIL, yet IsAtFactory now says true.
        let mut f = FakeReset::new(Err(3), true, true);
        let o = reset_to_factory(&mut f, false).unwrap();
        assert_eq!(f.resets, 1);
        assert_eq!(o.status, OperationStatus::Confirmed);
        assert_eq!(o.now_at_factory, Some(true));
        assert_eq!(o.rollback, RollbackStatus::Unsupported);
        let e = o.error.unwrap();
        assert!(e.contains("ADLX_FAIL (3)") && e.contains("now at factory settings"));
    }

    #[test]
    fn reset_error_with_unchanged_state_is_not_confirmed_not_not_applied() {
        // The error does not prove nothing changed: the result must say "not confirmed".
        let mut f = FakeReset::new(Err(18), false, true);
        let o = reset_to_factory(&mut f, false).unwrap();
        assert_eq!(f.resets, 1);
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(o.now_at_factory, Some(false));
        assert_eq!(o.rollback, RollbackStatus::Unsupported);
        let e = o.error.unwrap();
        assert!(e.contains("ADLX_RESET_NEEDED (18)") && e.contains("not confirmed"));
    }

    #[test]
    fn reset_error_with_unreadable_state_does_not_claim_it_was_not_applied() {
        let mut f = FakeReset::new(Err(3), true, false);
        let o = reset_to_factory(&mut f, false).unwrap();
        assert_ne!(o.status, OperationStatus::NotApplied);
        assert_eq!(o.status, OperationStatus::NotConfirmed);
        assert_eq!(o.now_at_factory, None);
        assert_eq!(o.rollback, RollbackStatus::Unsupported);
        assert!(o.error.unwrap().contains("unknown"));
    }

    #[test]
    fn reset_results_succeed_only_when_confirmed() {
        let result = |status, error: Option<&str>| TuningResetResult {
            gpu_name: "g".into(),
            pci_device_id: None,
            was_at_factory: Some(false),
            applied: status != OperationStatus::NotApplied,
            now_at_factory: None,
            confirmed: status == OperationStatus::Confirmed,
            status,
            rollback: RollbackStatus::Unsupported,
            error: error.map(str::to_string),
        };
        assert!(result(OperationStatus::Confirmed, None).succeeded());
        assert!(result(
            OperationStatus::Confirmed,
            Some("call errored but state is factory")
        )
        .succeeded());
        assert!(!result(OperationStatus::NotConfirmed, Some("x")).succeeded());
        assert!(!result(OperationStatus::Applied, None).succeeded());
    }

    #[test]
    fn result_helpers_follow_the_adlx_header() {
        assert!(adlx_succeeded(0) && adlx_succeeded(1) && adlx_succeeded(2));
        assert!(!adlx_succeeded(3) && !adlx_succeeded(-1) && !adlx_succeeded(18));
        assert_eq!(adlx_result_name(3), "ADLX_FAIL (3)");
        assert_eq!(adlx_result_name(18), "ADLX_RESET_NEEDED (18)");
        assert_eq!(adlx_result_name(99), "ADLX_RESULT(99)");
        assert_eq!(adlx_result_name(-5), "ADLX_RESULT(-5)");
    }

    #[test]
    fn succeeded_requires_confirmation_for_applied_changes() {
        let base = |status, error: Option<&str>| PowerLimitChangeResult {
            gpu_name: "g".into(),
            pci_device_id: None,
            previous_limit_pct: 0,
            requested_limit_pct: 5,
            allowed_range: range(),
            default_limit_pct: None,
            applied: status != OperationStatus::NotApplied,
            verified_limit_pct: None,
            confirmed: status == OperationStatus::Confirmed,
            status,
            rollback: RollbackStatus::NotNeeded,
            rollback_verified_limit_pct: None,
            error: error.map(str::to_string),
        };
        assert!(base(OperationStatus::Confirmed, None).succeeded());
        assert!(base(OperationStatus::NotApplied, None).succeeded()); // dry run
        assert!(!base(OperationStatus::NotApplied, Some("rejected")).succeeded());
        assert!(!base(OperationStatus::Applied, None).succeeded());
        assert!(!base(OperationStatus::NotConfirmed, None).succeeded());
    }

    #[test]
    fn memory_timing_codes_match_the_adlx_enum() {
        // ADLXDefines.h ADLX_MEMORYTIMING_DESCRIPTION: DEFAULT=0 .. MEMORYTIMING_LEVEL_2=5
        assert_eq!(MemoryTimingMode::from_adlx(0), MemoryTimingMode::Default);
        assert_eq!(MemoryTimingMode::from_adlx(1), MemoryTimingMode::FastTiming);
        assert_eq!(
            MemoryTimingMode::from_adlx(2),
            MemoryTimingMode::FastTimingLevel2
        );
        assert_eq!(MemoryTimingMode::from_adlx(3), MemoryTimingMode::Automatic);
        assert_eq!(MemoryTimingMode::from_adlx(4), MemoryTimingMode::Level1);
        assert_eq!(MemoryTimingMode::from_adlx(5), MemoryTimingMode::Level2);
        assert_eq!(MemoryTimingMode::from_adlx(6), MemoryTimingMode::Unknown(6));
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
