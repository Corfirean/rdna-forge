use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::architecture::GfxTarget;
use crate::error::{ForgeError, ForgeResult};

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
