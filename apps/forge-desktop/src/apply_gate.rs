//! Backend enforcement for every command that can change GPU or application configuration.
//!
//! The webview is a separate trust domain, so the frontend's `dry_run: false` is only a *request*.
//! The backend independently requires that:
//!
//! 1. all inputs are valid and unambiguous (this module's validators), and
//! 2. an identical dry-run preview succeeded recently in this process (the [`ApplyGate`]).
//!
//! A preview authorizes exactly one apply with exactly the same parameters and expires after
//! [`PREVIEW_TTL`]. This also guarantees the user was shown which configuration would change.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use forge_benchmark::KvCacheQuant;

/// How long a successful dry-run preview stays valid for a matching apply.
pub const PREVIEW_TTL: Duration = Duration::from_secs(15 * 60);

/// Hardware power limits are percentage offsets; anything beyond this is not a real request.
pub const MAX_POWER_LIMIT_ABS_PCT: i32 = 100;

/// Highest GPU index the desktop accepts (an obvious sanity bound, not a hardware limit).
pub const MAX_DEVICE_INDEX: usize = 63;

/// Remembers which exact operations were previewed.
#[derive(Debug, Default)]
pub struct ApplyGate {
    previews: Mutex<HashMap<String, Instant>>,
}

impl ApplyGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a dry run with this key completed successfully.
    pub fn record_preview(&self, key: &str) {
        self.record_preview_at(key, Instant::now());
    }

    /// Consumes the preview for `key`. Fails if none exists, it expired, or it was already used.
    pub fn authorize_apply(&self, key: &str) -> Result<(), String> {
        self.authorize_apply_at(key, Instant::now())
    }

    pub(crate) fn record_preview_at(&self, key: &str, now: Instant) {
        let mut map = self.previews.lock().unwrap_or_else(|p| p.into_inner());
        map.retain(|_, t| now.saturating_duration_since(*t) <= PREVIEW_TTL);
        map.insert(key.to_string(), now);
    }

    pub(crate) fn authorize_apply_at(&self, key: &str, now: Instant) -> Result<(), String> {
        let mut map = self.previews.lock().unwrap_or_else(|p| p.into_inner());
        match map.remove(key) {
            Some(t) if now.saturating_duration_since(t) <= PREVIEW_TTL => Ok(()),
            Some(_) => Err(
                "apply refused: the dry-run preview for these parameters has expired; run the preview again"
                    .to_string(),
            ),
            None => Err(
                "apply refused: no matching dry-run preview. Run the dry-run preview with exactly these parameters first, review it, then apply"
                    .to_string(),
            ),
        }
    }
}

/// Validates a power-limit request before anything touches the GPU.
pub fn validate_power_limit(limit_pct: i32, pci_device_id: Option<u32>) -> Result<(), String> {
    if !(-MAX_POWER_LIMIT_ABS_PCT..=MAX_POWER_LIMIT_ABS_PCT).contains(&limit_pct) {
        return Err(format!(
            "power limit {limit_pct}% is outside the plausible range of +/-{MAX_POWER_LIMIT_ABS_PCT}%; the hardware range is checked again by ADLX"
        ));
    }
    validate_pci_device_id(pci_device_id)
}

pub fn validate_pci_device_id(pci_device_id: Option<u32>) -> Result<(), String> {
    match pci_device_id {
        Some(id) if id == 0 || id > 0xFFFF => {
            Err(format!("PCI device id 0x{id:X} is not a valid 16-bit id"))
        }
        _ => Ok(()),
    }
}

pub fn validate_device_index(index: Option<usize>) -> Result<(), String> {
    match index {
        Some(i) if i > MAX_DEVICE_INDEX => Err(format!(
            "GPU device index {i} is out of range (max {MAX_DEVICE_INDEX})"
        )),
        _ => Ok(()),
    }
}

/// Accepts only a plain profile id (`[a-z0-9_]{1,64}`), never a path or free text.
pub fn validate_profile_id(profile_id: Option<&str>) -> Result<Option<String>, String> {
    let Some(raw) = profile_id.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if raw.len() > 64
        || !raw
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(format!(
            "profile id '{raw}' is invalid (expected 1-64 characters of a-z, 0-9, '_')"
        ));
    }
    Ok(Some(raw.to_string()))
}

/// Strict KV-cache type: only `f16`, `q8_0` or `q4_0`. Absent means the documented default
/// (`q8_0`); an unrecognized value is an error instead of silently becoming the default.
pub fn parse_kv_quant_strict(kv_quant: Option<&str>) -> Result<KvCacheQuant, String> {
    match kv_quant.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(KvCacheQuant::Q8_0),
        Some("f16") => Ok(KvCacheQuant::F16),
        Some("q8_0") => Ok(KvCacheQuant::Q8_0),
        Some("q4_0") => Ok(KvCacheQuant::Q4_0),
        Some(other) => Err(format!(
            "unknown KV cache type '{other}' (expected f16, q8_0 or q4_0)"
        )),
    }
}

/// Validates an explicit model path: bounded, no NUL, `.gguf` extension, existing regular file.
/// `None` / blank means "use the first discovered model" and is resolved by the caller.
pub fn validate_model_path(model_path: Option<&str>) -> Result<Option<PathBuf>, String> {
    let Some(raw) = model_path.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if raw.len() > 4096 || raw.contains('\0') {
        return Err("model path is too long or contains a NUL character".to_string());
    }
    let path = PathBuf::from(raw);
    let is_gguf = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if !is_gguf {
        return Err(format!("model path '{raw}' must point to a .gguf file"));
    }
    if !path.is_file() {
        return Err(format!(
            "model file '{raw}' does not exist or is not a regular file"
        ));
    }
    Ok(Some(path))
}

/// Canonical, parameter-complete key identifying one operation for the gate. Every field that
/// influences what gets written must be part of it so a preview cannot authorize a different apply.
pub fn gate_key(operation: &str, parts: &[String]) -> String {
    let mut key = String::from(operation);
    for p in parts {
        key.push('\u{1f}');
        key.push_str(p);
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rdna_forge_gate_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, b"x").unwrap();
        p
    }

    #[test]
    fn apply_without_a_preview_is_refused() {
        let gate = ApplyGate::new();
        let err = gate.authorize_apply("power|5").unwrap_err();
        assert!(err.contains("no matching dry-run preview"));
    }

    #[test]
    fn a_matching_preview_authorizes_exactly_one_apply() {
        let gate = ApplyGate::new();
        gate.record_preview("power|5");
        assert!(gate.authorize_apply("power|5").is_ok());
        // Single use: a second apply needs a fresh preview.
        assert!(gate.authorize_apply("power|5").is_err());
    }

    #[test]
    fn a_preview_does_not_authorize_different_parameters() {
        let gate = ApplyGate::new();
        gate.record_preview(&gate_key("power_limit", &["0x73AF".into(), "5".into()]));
        assert!(gate
            .authorize_apply(&gate_key("power_limit", &["0x73AF".into(), "10".into()]))
            .is_err());
        assert!(gate
            .authorize_apply(&gate_key("reset", &["0x73AF".into(), "5".into()]))
            .is_err());
        // The original preview is still intact for its own parameters.
        assert!(gate
            .authorize_apply(&gate_key("power_limit", &["0x73AF".into(), "5".into()]))
            .is_ok());
    }

    #[test]
    fn previews_expire() {
        let gate = ApplyGate::new();
        let t0 = Instant::now();
        gate.record_preview_at("k", t0);
        let err = gate
            .authorize_apply_at("k", t0 + PREVIEW_TTL + Duration::from_secs(1))
            .unwrap_err();
        assert!(err.contains("expired"));

        gate.record_preview_at("k2", t0);
        assert!(gate.authorize_apply_at("k2", t0 + PREVIEW_TTL).is_ok());
    }

    #[test]
    fn key_parts_cannot_be_confused_by_concatenation() {
        let a = gate_key("op", &["ab".into(), "c".into()]);
        let b = gate_key("op", &["a".into(), "bc".into()]);
        assert_ne!(a, b);
    }

    #[test]
    fn power_limit_inputs_are_validated() {
        assert!(validate_power_limit(0, None).is_ok());
        assert!(validate_power_limit(-100, Some(0x73AF)).is_ok());
        assert!(validate_power_limit(101, None).is_err());
        assert!(validate_power_limit(i32::MIN, None).is_err());
        assert!(validate_power_limit(i32::MAX, None).is_err());
        assert!(validate_power_limit(5, Some(0)).is_err());
        assert!(validate_power_limit(5, Some(0x1_0000)).is_err());
    }

    #[test]
    fn ambiguous_kv_quant_is_rejected_not_defaulted() {
        assert_eq!(parse_kv_quant_strict(None).unwrap(), KvCacheQuant::Q8_0);
        assert_eq!(
            parse_kv_quant_strict(Some("  ")).unwrap(),
            KvCacheQuant::Q8_0
        );
        assert_eq!(
            parse_kv_quant_strict(Some("f16")).unwrap(),
            KvCacheQuant::F16
        );
        assert_eq!(
            parse_kv_quant_strict(Some("q4_0")).unwrap(),
            KvCacheQuant::Q4_0
        );
        for bad in ["q5_0", "Q8_0", "q8_0;rm", "../x", "f32"] {
            assert!(
                parse_kv_quant_strict(Some(bad)).is_err(),
                "accepted {bad:?}"
            );
        }
    }

    #[test]
    fn profile_ids_must_be_plain_identifiers() {
        assert_eq!(validate_profile_id(None).unwrap(), None);
        assert_eq!(validate_profile_id(Some("  ")).unwrap(), None);
        assert_eq!(
            validate_profile_id(Some("vulkan_wddm_priority"))
                .unwrap()
                .as_deref(),
            Some("vulkan_wddm_priority")
        );
        for bad in ["../evil", "a b", "UPPER", "x;y", "a\\b", &"a".repeat(65)] {
            assert!(validate_profile_id(Some(bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn model_paths_must_be_existing_gguf_files() {
        assert_eq!(validate_model_path(None).unwrap(), None);
        assert_eq!(validate_model_path(Some("   ")).unwrap(), None);

        let ok = temp_file("model.GGUF");
        assert_eq!(
            validate_model_path(Some(&ok.display().to_string())).unwrap(),
            Some(ok.clone())
        );

        let not_gguf = temp_file("model.bin");
        assert!(validate_model_path(Some(&not_gguf.display().to_string())).is_err());
        let dir = ok.parent().unwrap().display().to_string();
        assert!(validate_model_path(Some(&dir)).is_err());
        assert!(validate_model_path(Some("C:\\definitely\\missing\\m.gguf")).is_err());
        assert!(validate_model_path(Some("m\0.gguf")).is_err());
        assert!(validate_model_path(Some(&"a".repeat(5000))).is_err());
    }

    #[test]
    fn device_indices_are_bounded() {
        assert!(validate_device_index(None).is_ok());
        assert!(validate_device_index(Some(0)).is_ok());
        assert!(validate_device_index(Some(MAX_DEVICE_INDEX)).is_ok());
        assert!(validate_device_index(Some(MAX_DEVICE_INDEX + 1)).is_err());
        assert!(validate_device_index(Some(usize::MAX)).is_err());
    }
}
