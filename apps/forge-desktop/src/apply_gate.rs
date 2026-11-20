//! Backend enforcement for every command that can change GPU or application configuration.
//!
//! The webview is a separate trust domain, so the frontend's `dry_run: false` is only a *request*.
//! The backend independently requires that:
//!
//! 1. all inputs are valid and unambiguous (this module's validators),
//! 2. an identical dry-run preview succeeded recently in this process (the [`ApplyGate`]), and
//! 3. the preview was made for the **same physical GPU** that the apply would now target.
//!
//! A preview authorizes exactly one apply with exactly the same parameters, expires after
//! [`PREVIEW_TTL`], and is bound to a [`DeviceIdentity`] resolved by the backend at preview time
//! and resolved again at apply time. This also guarantees the user was shown which configuration
//! would change, and on which GPU.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use forge_benchmark::KvCacheQuant;
use forge_core::{GpuDeviceInfo, AMD_PCI_VENDOR_ID};

/// How long a successful dry-run preview stays valid for a matching apply.
pub const PREVIEW_TTL: Duration = Duration::from_secs(15 * 60);

/// Hardware power limits are percentage offsets; anything beyond this is not a real request.
pub const MAX_POWER_LIMIT_ABS_PCT: i32 = 100;

/// Highest GPU index the desktop accepts (an obvious sanity bound, not a hardware limit).
pub const MAX_DEVICE_INDEX: usize = 63;

/// Identity of the physical GPU an operation targets, resolved by the backend (never supplied by
/// the frontend).
///
/// The token combines vendor id, device id and the PCI bus id (the slot), plus the adapter name.
/// The bus id is what distinguishes two identical cards and what survives a change in enumeration
/// order. The enumeration index is *not* part of the token, except as a clearly labelled fallback
/// when the platform reports no bus id for an adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceIdentity(String);

impl DeviceIdentity {
    /// For operations that do not target a GPU (restoring files or registry values).
    pub fn none() -> Self {
        Self("none".to_string())
    }

    pub fn token(&self) -> &str {
        &self.0
    }
}

/// Builds the identity token of one discovered adapter.
pub fn identity_of(gpu: &GpuDeviceInfo) -> DeviceIdentity {
    let bus = match gpu
        .pci_bus_id
        .as_deref()
        .map(str::trim)
        .filter(|b| !b.is_empty())
    {
        Some(bus) => format!("bus={bus}"),
        // No stable slot id: fall back to the index, and say so in the token.
        None => format!("bus=unknown(index={})", gpu.index),
    };
    DeviceIdentity(format!(
        "gpu:{:04x}:{:04x}/{}/{}",
        gpu.vendor_id,
        gpu.device_id,
        bus,
        gpu.name.trim()
    ))
}

/// Selects the GPU a tuning operation (power limit, factory reset) would act on and returns its
/// identity.
///
/// With a PCI device id the target is the AMD adapter with that id; with none it is the single
/// discrete AMD adapter. If the choice is ambiguous (two identical cards share a device id, or
/// several discrete cards exist) the request is refused: the backend cannot tell which one the
/// driver would pick, so it must not authorize either.
pub fn select_tuning_target(
    gpus: &[GpuDeviceInfo],
    pci_device_id: Option<u32>,
) -> Result<DeviceIdentity, String> {
    let amd = gpus.iter().filter(|g| g.vendor_id == AMD_PCI_VENDOR_ID);
    let candidates: Vec<&GpuDeviceInfo> = match pci_device_id {
        Some(id) => amd.filter(|g| g.device_id == id).collect(),
        None => amd.filter(|g| g.discrete).collect(),
    };
    match candidates.as_slice() {
        [one] => Ok(identity_of(one)),
        [] => Err(match pci_device_id {
            Some(id) => format!("no AMD GPU with PCI device id 0x{id:04X} is present"),
            None => "no discrete AMD GPU is present".to_string(),
        }),
        many => Err(format!(
            "the target GPU is ambiguous ({} candidates: {}); the apply would not be bound to one physical device",
            many.len(),
            many.iter()
                .map(|g| g.name.trim())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Selects the GPU an LLM configuration patch targets, mirroring the planner: an explicit
/// position in the discovery list, otherwise the first discrete adapter, otherwise the first one.
pub fn select_llm_target(
    gpus: &[GpuDeviceInfo],
    device_index: Option<usize>,
) -> Result<DeviceIdentity, String> {
    let gpu = match device_index {
        Some(i) => gpus
            .get(i)
            .ok_or_else(|| format!("GPU device index {i} is not present"))?,
        None => gpus
            .iter()
            .find(|g| g.discrete)
            .or_else(|| gpus.first())
            .ok_or_else(|| "no GPU devices were discovered".to_string())?,
    };
    Ok(identity_of(gpu))
}

#[derive(Debug)]
struct Preview {
    at: Instant,
    identity: DeviceIdentity,
}

/// Remembers which exact operations were previewed, and for which GPU.
#[derive(Debug, Default)]
pub struct ApplyGate {
    previews: Mutex<HashMap<String, Preview>>,
}

impl ApplyGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a dry run with this key, targeting `identity`, completed successfully.
    pub fn record_preview(&self, key: &str, identity: &DeviceIdentity) {
        self.record_preview_at(key, identity, Instant::now());
    }

    /// Consumes the preview for `key`. Fails if none exists, it expired, it was already used, or
    /// it was made for a different GPU than `identity`. A refused attempt still consumes the
    /// preview, so a new preview is required.
    pub fn authorize_apply(&self, key: &str, identity: &DeviceIdentity) -> Result<(), String> {
        self.authorize_apply_at(key, identity, Instant::now())
    }

    pub(crate) fn record_preview_at(&self, key: &str, identity: &DeviceIdentity, now: Instant) {
        let mut map = self.previews.lock().unwrap_or_else(|p| p.into_inner());
        map.retain(|_, p| now.saturating_duration_since(p.at) <= PREVIEW_TTL);
        map.insert(
            key.to_string(),
            Preview {
                at: now,
                identity: identity.clone(),
            },
        );
    }

    pub(crate) fn authorize_apply_at(
        &self,
        key: &str,
        identity: &DeviceIdentity,
        now: Instant,
    ) -> Result<(), String> {
        let mut map = self.previews.lock().unwrap_or_else(|p| p.into_inner());
        let Some(preview) = map.remove(key) else {
            return Err(
                "apply refused: no matching dry-run preview. Run the dry-run preview with exactly these parameters first, review it, then apply"
                    .to_string(),
            );
        };
        if now.saturating_duration_since(preview.at) > PREVIEW_TTL {
            return Err(
                "apply refused: the dry-run preview for these parameters has expired; run the preview again"
                    .to_string(),
            );
        }
        if preview.identity != *identity {
            return Err(format!(
                "apply refused: the target GPU changed since the preview (previewed '{}', now '{}'); run the preview again",
                preview.identity.token(),
                identity.token()
            ));
        }
        Ok(())
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
pub(crate) mod test_support {
    use forge_core::{GpuArchitecture, GpuDeviceInfo};

    /// Builds a minimal adapter record for gate tests.
    pub fn gpu(
        index: usize,
        name: &str,
        vendor_id: u32,
        device_id: u32,
        pci_bus_id: Option<&str>,
        discrete: bool,
    ) -> GpuDeviceInfo {
        GpuDeviceInfo {
            index,
            name: name.to_string(),
            vendor_id,
            device_id,
            pci_bus_id: pci_bus_id.map(str::to_string),
            architecture: GpuArchitecture::Rdna2,
            gfx_target: None,
            discrete,
            vram_bytes: 0,
            shared_memory_per_block_bytes: None,
            vulkan_shared_memory_bytes: None,
            compute_units: None,
            wgp_count: None,
            warp_size: None,
            max_buffer_size_bytes: None,
            max_memory_allocation_bytes: None,
            rebar_enabled: None,
            driver_version: None,
            adrenalin_version: None,
            vulkan_api_version: None,
            vulkan_driver_info: None,
            vulkan_cooperative_matrix: false,
            vulkan_fp16: false,
            vulkan_int8: false,
            vulkan_int8_dot_product: false,
            hip_available: false,
            hip_runtime_version: None,
            core_clock_mhz: None,
            memory_clock_mhz: None,
            memory_bus_width_bits: None,
            l2_cache_bytes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::gpu;
    use super::*;

    const AMD: u32 = 0x1002;

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

    fn id(token: &str) -> DeviceIdentity {
        DeviceIdentity(token.to_string())
    }

    // ------------------------------------------------------------------ gate behavior

    #[test]
    fn apply_without_a_preview_is_refused() {
        let gate = ApplyGate::new();
        let err = gate.authorize_apply("power|5", &id("a")).unwrap_err();
        assert!(err.contains("no matching dry-run preview"));
    }

    #[test]
    fn matching_device_and_parameters_authorize_exactly_one_apply() {
        let gate = ApplyGate::new();
        gate.record_preview("power|5", &id("gpu-a"));
        assert!(gate.authorize_apply("power|5", &id("gpu-a")).is_ok());
        // Single use: a second apply needs a fresh preview.
        assert!(gate.authorize_apply("power|5", &id("gpu-a")).is_err());
    }

    #[test]
    fn a_preview_for_one_gpu_never_authorizes_another() {
        let gate = ApplyGate::new();
        gate.record_preview("power|5", &id("gpu-a"));
        let err = gate.authorize_apply("power|5", &id("gpu-b")).unwrap_err();
        assert!(err.contains("target GPU changed"));
        assert!(err.contains("gpu-a") && err.contains("gpu-b"));
        // The refused attempt consumed the preview: even the original GPU needs a new one.
        assert!(gate.authorize_apply("power|5", &id("gpu-a")).is_err());
    }

    #[test]
    fn a_preview_does_not_authorize_different_parameters() {
        let gate = ApplyGate::new();
        let dev = id("gpu-a");
        gate.record_preview(
            &gate_key("power_limit", &["0x73AF".into(), "5".into()]),
            &dev,
        );
        assert!(gate
            .authorize_apply(
                &gate_key("power_limit", &["0x73AF".into(), "10".into()]),
                &dev
            )
            .is_err());
        assert!(gate
            .authorize_apply(&gate_key("reset", &["0x73AF".into(), "5".into()]), &dev)
            .is_err());
        // The original preview is still intact for its own parameters.
        assert!(gate
            .authorize_apply(
                &gate_key("power_limit", &["0x73AF".into(), "5".into()]),
                &dev
            )
            .is_ok());
    }

    #[test]
    fn previews_expire() {
        let gate = ApplyGate::new();
        let dev = id("gpu-a");
        let t0 = Instant::now();
        gate.record_preview_at("k", &dev, t0);
        let err = gate
            .authorize_apply_at("k", &dev, t0 + PREVIEW_TTL + Duration::from_secs(1))
            .unwrap_err();
        assert!(err.contains("expired"));

        gate.record_preview_at("k2", &dev, t0);
        assert!(gate
            .authorize_apply_at("k2", &dev, t0 + PREVIEW_TTL)
            .is_ok());
    }

    #[test]
    fn operations_without_a_gpu_use_an_explicit_none_identity() {
        let gate = ApplyGate::new();
        gate.record_preview("env_rollback", &DeviceIdentity::none());
        assert!(gate
            .authorize_apply("env_rollback", &DeviceIdentity::none())
            .is_ok());
        // A GPU-bound preview can never satisfy a "none" apply or the reverse.
        gate.record_preview("x", &id("gpu-a"));
        assert!(gate.authorize_apply("x", &DeviceIdentity::none()).is_err());
    }

    #[test]
    fn key_parts_cannot_be_confused_by_concatenation() {
        let a = gate_key("op", &["ab".into(), "c".into()]);
        let b = gate_key("op", &["a".into(), "bc".into()]);
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------ device identity selection

    #[test]
    fn identity_uses_the_bus_id_and_survives_reordering() {
        let a = gpu(
            0,
            "AMD Radeon RX 6900 XT",
            AMD,
            0x73AF,
            Some("0000:03:00.0"),
            true,
        );
        let a_moved = gpu(
            1,
            "AMD Radeon RX 6900 XT",
            AMD,
            0x73AF,
            Some("0000:03:00.0"),
            true,
        );
        // Same physical card at a different enumeration index: same identity.
        assert_eq!(identity_of(&a), identity_of(&a_moved));
        let token = identity_of(&a).token().to_string();
        assert!(token.contains("1002:73af") && token.contains("0000:03:00.0"));
        assert!(!token.contains("index="));

        // Two identical cards in different slots differ.
        let b = gpu(
            1,
            "AMD Radeon RX 6900 XT",
            AMD,
            0x73AF,
            Some("0000:06:00.0"),
            true,
        );
        assert_ne!(identity_of(&a), identity_of(&b));

        // Without a bus id the index is the labelled fallback.
        let no_bus = gpu(2, "AMD Radeon RX 6900 XT", AMD, 0x73AF, None, true);
        assert!(identity_of(&no_bus)
            .token()
            .contains("bus=unknown(index=2)"));
    }

    #[test]
    fn tuning_target_matching_device_and_parameters_is_authorized() {
        let gpus = vec![
            gpu(
                0,
                "AMD Radeon RX 6900 XT",
                AMD,
                0x73AF,
                Some("0000:03:00.0"),
                true,
            ),
            gpu(
                1,
                "AMD Radeon(TM) Graphics",
                AMD,
                0x164E,
                Some("0000:0e:00.0"),
                false,
            ),
        ];
        let gate = ApplyGate::new();
        let at_preview = select_tuning_target(&gpus, Some(0x73AF)).unwrap();
        gate.record_preview("power|5", &at_preview);
        let at_apply = select_tuning_target(&gpus, Some(0x73AF)).unwrap();
        assert!(gate.authorize_apply("power|5", &at_apply).is_ok());

        // No PCI id: the single discrete AMD adapter is the target, and it is the same card.
        assert_eq!(select_tuning_target(&gpus, None).unwrap(), at_preview);
    }

    #[test]
    fn tuning_preview_for_one_pci_device_does_not_authorize_another() {
        let gpus = vec![
            gpu(
                0,
                "AMD Radeon RX 6900 XT",
                AMD,
                0x73AF,
                Some("0000:03:00.0"),
                true,
            ),
            gpu(
                1,
                "AMD Radeon RX 6800 XT",
                AMD,
                0x73BF,
                Some("0000:06:00.0"),
                true,
            ),
        ];
        let gate = ApplyGate::new();
        gate.record_preview(
            "power|5",
            &select_tuning_target(&gpus, Some(0x73AF)).unwrap(),
        );
        let other = select_tuning_target(&gpus, Some(0x73BF)).unwrap();
        assert!(gate
            .authorize_apply("power|5", &other)
            .unwrap_err()
            .contains("target GPU changed"));
    }

    #[test]
    fn identical_cards_sharing_a_device_id_are_refused_not_guessed() {
        let gpus = vec![
            gpu(
                0,
                "AMD Radeon RX 6900 XT",
                AMD,
                0x73AF,
                Some("0000:03:00.0"),
                true,
            ),
            gpu(
                1,
                "AMD Radeon RX 6900 XT",
                AMD,
                0x73AF,
                Some("0000:06:00.0"),
                true,
            ),
        ];
        let err = select_tuning_target(&gpus, Some(0x73AF)).unwrap_err();
        assert!(err.contains("ambiguous") && err.contains("2 candidates"));
        // Several discrete cards and no PCI id is equally ambiguous.
        assert!(select_tuning_target(&gpus, None)
            .unwrap_err()
            .contains("ambiguous"));
    }

    #[test]
    fn absent_devices_are_errors() {
        let gpus = vec![gpu(
            0,
            "AMD Radeon RX 6900 XT",
            AMD,
            0x73AF,
            Some("0000:03:00.0"),
            true,
        )];
        assert!(select_tuning_target(&gpus, Some(0x7551))
            .unwrap_err()
            .contains("0x7551"));
        assert!(select_tuning_target(&[], None).is_err());
        // A non-AMD adapter with a matching device id is not a tuning target.
        let nvidia = vec![gpu(
            0,
            "Other GPU",
            0x10DE,
            0x73AF,
            Some("0000:01:00.0"),
            true,
        )];
        assert!(select_tuning_target(&nvidia, Some(0x73AF)).is_err());
        assert!(select_llm_target(&gpus, Some(3))
            .unwrap_err()
            .contains("index 3"));
        assert!(select_llm_target(&[], None).is_err());
        assert!(select_llm_target(&[], Some(0)).is_err());
    }

    #[test]
    fn a_changed_device_index_cannot_reuse_a_preview_for_another_gpu() {
        // Preview with device_index=1 while the list is [iGPU, dGPU].
        let igpu = gpu(
            0,
            "AMD Radeon(TM) Graphics",
            AMD,
            0x164E,
            Some("0000:0e:00.0"),
            false,
        );
        let dgpu = gpu(
            1,
            "AMD Radeon RX 6900 XT",
            AMD,
            0x73AF,
            Some("0000:03:00.0"),
            true,
        );
        let before = vec![igpu.clone(), dgpu.clone()];
        let gate = ApplyGate::new();
        gate.record_preview("lms|1", &select_llm_target(&before, Some(1)).unwrap());

        // Enumeration order changed (driver update, device re-ordering): index 1 is now the iGPU.
        let after = vec![dgpu, igpu];
        let now = select_llm_target(&after, Some(1)).unwrap();
        assert!(gate
            .authorize_apply("lms|1", &now)
            .unwrap_err()
            .contains("target GPU changed"));
    }

    #[test]
    fn llm_target_defaults_to_the_first_discrete_adapter_like_the_planner() {
        let igpu = gpu(
            0,
            "AMD Radeon(TM) Graphics",
            AMD,
            0x164E,
            Some("0000:0e:00.0"),
            false,
        );
        let dgpu = gpu(
            1,
            "AMD Radeon RX 6900 XT",
            AMD,
            0x73AF,
            Some("0000:03:00.0"),
            true,
        );
        let gpus = vec![igpu.clone(), dgpu.clone()];
        assert_eq!(select_llm_target(&gpus, None).unwrap(), identity_of(&dgpu));
        assert_eq!(
            select_llm_target(&gpus, Some(0)).unwrap(),
            identity_of(&igpu)
        );
        // Only an iGPU: it is the fallback.
        assert_eq!(
            select_llm_target(std::slice::from_ref(&igpu), None).unwrap(),
            identity_of(&igpu)
        );
    }

    /// Hardware check (read-only discovery, no GPU state is changed): on a real AMD system the
    /// identity must come from the PCI bus id, not from the index fallback.
    /// Run with: `cargo test -p forge-desktop -- --ignored --nocapture live_identity`
    #[test]
    #[ignore = "requires a discovered AMD GPU; run explicitly"]
    fn live_identity_resolves_from_the_pci_bus_id() {
        let gpus = forge_device::discover_all().gpus;
        assert!(!gpus.is_empty(), "no GPUs discovered");
        for g in &gpus {
            let token = identity_of(g).token().to_string();
            println!("index {} -> {token}", g.index);
            assert!(
                !token.contains("bus=unknown"),
                "no PCI bus id for '{}'",
                g.name
            );
        }
        let llm = select_llm_target(&gpus, None).unwrap();
        println!("default LLM target: {}", llm.token());
        // Stable across two independent discovery runs (same physical GPUs, same identities).
        let again = forge_device::discover_all().gpus;
        assert_eq!(select_llm_target(&again, None).unwrap(), llm);
        if let Ok(t) = select_tuning_target(&gpus, None) {
            println!("default tuning target: {}", t.token());
            assert_eq!(select_tuning_target(&again, None).unwrap(), t);
        }
    }

    // ------------------------------------------------------------------ input validation

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
