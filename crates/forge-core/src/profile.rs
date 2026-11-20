use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::architecture::{GfxTarget, AMD_PCI_VENDOR_ID};
use crate::error::{ForgeError, ForgeResult};
use crate::status::GpuDeviceInfo;

/// Declarative hardware target profile stored under `profiles/<gfx>/<device>.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub device: ProfileDeviceSection,
    pub compute: ProfileComputeSection,
    pub memory: ProfileMemorySection,
    pub capabilities: ProfileCapabilitiesSection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDeviceSection {
    pub name: String,
    pub gfx_target: String,
    pub architecture: String,
    pub pci_vendor_id: String,
    pub pci_device_ids: Vec<String>,
    pub discrete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileComputeSection {
    pub compute_units: u32,
    pub wgp_count: u32,
    pub simds_per_cu: u32,
    pub wave_sizes: Vec<u32>,
    pub lds_bytes_per_cu: u32,
    pub vulkan_max_shared_memory_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileMemorySection {
    pub vram_bytes: u64,
    pub bus_width_bits: u32,
    pub l2_cache_bytes: u64,
    pub infinity_cache_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileCapabilitiesSection {
    pub fp16: bool,
    pub bf16: bool,
    pub fp8: bool,
    pub dp4a_i8: bool,
    pub dot8_i4: bool,
    pub wmma: bool,
    pub mfma: bool,
}

impl HardwareProfile {
    /// Parses a hardware profile from a TOML string and validates invariants.
    pub fn from_toml_str(input: &str) -> ForgeResult<Self> {
        let profile: Self =
            toml::from_str(input).map_err(|e| ForgeError::InvalidProfile(e.to_string()))?;
        profile.validate()?;
        Ok(profile)
    }

    /// Loads and validates a hardware profile from disk.
    pub fn from_file(path: &Path) -> ForgeResult<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_toml_str(&content)
    }

    /// Validates all profile fields so invalid profiles are rejected early.
    pub fn validate(&self) -> ForgeResult<GfxTarget> {
        if self.device.name.trim().is_empty() {
            return Err(ForgeError::InvalidProfile(
                "device.name must not be empty".to_string(),
            ));
        }

        let target = GfxTarget::parse(&self.device.gfx_target)
            .map_err(|e| ForgeError::InvalidProfile(e.to_string()))?;

        if self.device.pci_device_ids.is_empty() {
            return Err(ForgeError::InvalidProfile(
                "device.pci_device_ids must contain at least one ID".to_string(),
            ));
        }
        for id in &self.device.pci_device_ids {
            let trimmed = id.trim().trim_start_matches("0x").trim_start_matches("0X");
            if trimmed.is_empty() || u32::from_str_radix(trimmed, 16).is_err() {
                return Err(ForgeError::InvalidProfile(format!(
                    "invalid PCI device ID '{id}'"
                )));
            }
        }

        if self.compute.compute_units == 0 || self.compute.wgp_count == 0 {
            return Err(ForgeError::InvalidProfile(
                "compute_units and wgp_count must be > 0".to_string(),
            ));
        }
        if self.compute.wave_sizes.is_empty()
            || self.compute.wave_sizes.iter().any(|&w| w != 32 && w != 64)
        {
            return Err(ForgeError::InvalidProfile(
                "compute.wave_sizes must be non-empty and contain only 32 or 64".to_string(),
            ));
        }
        if self.compute.lds_bytes_per_cu == 0 || self.compute.vulkan_max_shared_memory_bytes == 0 {
            return Err(ForgeError::InvalidProfile(
                "LDS and shared memory limits must be > 0".to_string(),
            ));
        }
        if self.memory.vram_bytes == 0 {
            return Err(ForgeError::InvalidProfile(
                "memory.vram_bytes must be > 0".to_string(),
            ));
        }

        Ok(target)
    }
}

/// Built-in hardware profile metadata embedded in `forge-core`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltinProfileInfo {
    pub id: String,
    pub relative_path: String,
    pub profile: HardwareProfile,
}

const BUILTIN_PROFILE_SPECS: &[(&str, &str, &str)] = &[
    (
        "rx6900xt",
        "profiles/gfx1030/rx6900xt.toml",
        include_str!("../../../profiles/gfx1030/rx6900xt.toml"),
    ),
    (
        "rx6950xt",
        "profiles/gfx1030/rx6950xt.toml",
        include_str!("../../../profiles/gfx1030/rx6950xt.toml"),
    ),
    (
        "rx6800xt",
        "profiles/gfx1030/rx6800xt.toml",
        include_str!("../../../profiles/gfx1030/rx6800xt.toml"),
    ),
    (
        "rx6700xt",
        "profiles/gfx1031/rx6700xt.toml",
        include_str!("../../../profiles/gfx1031/rx6700xt.toml"),
    ),
    (
        "rx6600xt",
        "profiles/gfx1032/rx6600xt.toml",
        include_str!("../../../profiles/gfx1032/rx6600xt.toml"),
    ),
    (
        "rx7900xtx",
        "profiles/gfx1100/rx7900xtx.toml",
        include_str!("../../../profiles/gfx1100/rx7900xtx.toml"),
    ),
    (
        "rx7900xt",
        "profiles/gfx1100/rx7900xt.toml",
        include_str!("../../../profiles/gfx1100/rx7900xt.toml"),
    ),
    (
        "pro_w7900",
        "profiles/gfx1100/pro_w7900.toml",
        include_str!("../../../profiles/gfx1100/pro_w7900.toml"),
    ),
    (
        "rx7800xt",
        "profiles/gfx1101/rx7800xt.toml",
        include_str!("../../../profiles/gfx1101/rx7800xt.toml"),
    ),
    (
        "rx7700xt",
        "profiles/gfx1101/rx7700xt.toml",
        include_str!("../../../profiles/gfx1101/rx7700xt.toml"),
    ),
    (
        "rx7600xt",
        "profiles/gfx1102/rx7600xt.toml",
        include_str!("../../../profiles/gfx1102/rx7600xt.toml"),
    ),
    (
        "radeon780m",
        "profiles/gfx1103/radeon780m.toml",
        include_str!("../../../profiles/gfx1103/radeon780m.toml"),
    ),
    (
        "radeon890m",
        "profiles/gfx1150/radeon890m.toml",
        include_str!("../../../profiles/gfx1150/radeon890m.toml"),
    ),
    (
        "radeon8060s",
        "profiles/gfx1151/radeon8060s.toml",
        include_str!("../../../profiles/gfx1151/radeon8060s.toml"),
    ),
    (
        "rx9070xt",
        "profiles/gfx1201/rx9070xt.toml",
        include_str!("../../../profiles/gfx1201/rx9070xt.toml"),
    ),
    (
        "rx9070",
        "profiles/gfx1201/rx9070.toml",
        include_str!("../../../profiles/gfx1201/rx9070.toml"),
    ),
    (
        "ai_pro_r9700",
        "profiles/gfx1201/ai_pro_r9700.toml",
        include_str!("../../../profiles/gfx1201/ai_pro_r9700.toml"),
    ),
    (
        "rx9060xt",
        "profiles/gfx1200/rx9060xt.toml",
        include_str!("../../../profiles/gfx1200/rx9060xt.toml"),
    ),
];

/// Returns all validated built-in AMD Radeon hardware profiles bundled with `forge-core`.
pub fn builtin_profiles() -> Vec<BuiltinProfileInfo> {
    BUILTIN_PROFILE_SPECS
        .iter()
        .filter_map(|&(id, relative_path, raw)| {
            HardwareProfile::from_toml_str(raw)
                .ok()
                .map(|profile| BuiltinProfileInfo {
                    id: id.to_string(),
                    relative_path: relative_path.to_string(),
                    profile,
                })
        })
        .collect()
}

/// Finds a built-in hardware profile by case-insensitive identifier (e.g. `"rx7900xtx"`).
pub fn builtin_profile_by_id(id: &str) -> Option<BuiltinProfileInfo> {
    let needle = id.trim();
    builtin_profiles()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case(needle))
}

fn parse_hex_u32(s: &str) -> Option<u32> {
    let trimmed = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    if trimmed.is_empty() {
        return None;
    }
    u32::from_str_radix(trimmed, 16).ok()
}

fn model_tokens(name: &str) -> Vec<String> {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|tok| !tok.is_empty())
        .map(|tok| tok.to_ascii_lowercase())
        .filter(|tok| {
            !matches!(
                tok.as_str(),
                "amd" | "radeon" | "tm" | "r" | "graphics" | "series"
            )
        })
        .collect()
}

/// Automatically matches a discovered GPU adapter to the closest built-in AMD hardware profile,
/// disambiguating shared PCI device IDs using model name tokens, CU counts, and VRAM capacity.
pub fn match_profile_for_device(device: &GpuDeviceInfo) -> Option<BuiltinProfileInfo> {
    if device.vendor_id != AMD_PCI_VENDOR_ID {
        return None;
    }

    let dev_tokens = model_tokens(&device.name);
    let dev_gfx = device.gfx_target.as_ref().map(|t| t.base.as_str());

    let mut best: Option<(i32, BuiltinProfileInfo)> = None;
    for candidate in builtin_profiles() {
        let p = &candidate.profile;
        let pci_match = p
            .device
            .pci_device_ids
            .iter()
            .filter_map(|id| parse_hex_u32(id))
            .any(|id| id == device.device_id);
        let gfx_match = dev_gfx.is_some_and(|g| g == p.device.gfx_target);

        if !pci_match && !gfx_match {
            continue;
        }

        let mut score: i32 = if pci_match { 20 } else { 10 };

        if device.discrete == p.device.discrete {
            score += 15;
        } else {
            score -= 30;
        }

        let prof_tokens = model_tokens(&p.device.name);
        if !prof_tokens.is_empty() && !dev_tokens.is_empty() {
            if prof_tokens == dev_tokens {
                score += 100;
            } else if prof_tokens.iter().all(|t| dev_tokens.contains(t)) {
                score += 50 + (prof_tokens.len() as i32 * 5);
            }
        }

        if device.compute_units == Some(p.compute.compute_units) {
            score += 40;
        }

        if device.discrete && device.vram_bytes > 0 {
            let diff = device.vram_bytes.abs_diff(p.memory.vram_bytes);
            if diff <= p.memory.vram_bytes / 6 {
                score += 25;
            }
        }

        if score >= 30 {
            let replace = match &best {
                None => true,
                Some((best_score, _)) => score > *best_score,
            };
            if replace {
                best = Some((score, candidate));
            }
        }
    }

    best.map(|(_, info)| info)
}
