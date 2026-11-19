use std::fmt;

use serde::{Deserialize, Serialize};

use crate::architecture::{GfxTarget, GpuArchitecture};

/// Verification status for backends, capabilities, and compatibility matrix entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportStatus {
    Verified,
    PartiallyVerified,
    Experimental,
    Unsupported,
    NotTested,
}

impl fmt::Display for SupportStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verified => write!(f, "VERIFIED"),
            Self::PartiallyVerified => write!(f, "PARTIALLY VERIFIED"),
            Self::Experimental => write!(f, "EXPERIMENTAL"),
            Self::Unsupported => write!(f, "UNSUPPORTED"),
            Self::NotTested => write!(f, "NOT TESTED"),
        }
    }
}

/// Supported compute, telemetry, or inference backend identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Vulkan,
    Hip,
    Triton,
    DirectMl,
    OpenCl,
    Adlx,
    Amf,
    LlamaCpp,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Hip => write!(f, "HIP"),
            Self::Triton => write!(f, "Triton"),
            Self::DirectMl => write!(f, "DirectML"),
            Self::OpenCl => write!(f, "OpenCL"),
            Self::Adlx => write!(f, "ADLX"),
            Self::Amf => write!(f, "AMF"),
            Self::LlamaCpp => write!(f, "llama.cpp"),
        }
    }
}

/// Detected runtime/backend status reported by `forge doctor`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub kind: BackendKind,
    pub status: SupportStatus,
    pub version: Option<String>,
    pub summary: String,
    pub details: Vec<String>,
}

/// Outcome of an individual hardware or kernel capability check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityResult {
    pub name: String,
    pub status: SupportStatus,
    pub detail: Option<String>,
}

/// Unified description of a discovered GPU adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub pci_bus_id: Option<String>,
    pub architecture: GpuArchitecture,
    pub gfx_target: Option<GfxTarget>,
    pub discrete: bool,
    pub vram_bytes: u64,
    pub shared_memory_per_block_bytes: Option<u64>,
    pub vulkan_shared_memory_bytes: Option<u32>,
    pub compute_units: Option<u32>,
    pub wgp_count: Option<u32>,
    pub warp_size: Option<u32>,
    pub max_buffer_size_bytes: Option<u64>,
    pub max_memory_allocation_bytes: Option<u64>,
    pub rebar_enabled: Option<bool>,
    pub driver_version: Option<String>,
    pub adrenalin_version: Option<String>,
    pub vulkan_api_version: Option<String>,
    pub vulkan_driver_info: Option<String>,
    pub vulkan_cooperative_matrix: bool,
    pub vulkan_fp16: bool,
    pub vulkan_int8: bool,
    pub hip_available: bool,
    pub hip_runtime_version: Option<String>,
    pub core_clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
    pub memory_bus_width_bits: Option<u32>,
    pub l2_cache_bytes: Option<u64>,
}

/// Detected host OS, CPU, and system memory metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os_name: String,
    pub os_version: String,
    pub os_build: u32,
    pub architecture: String,
    pub cpu_name: String,
    pub logical_cores: u32,
    pub total_ram_bytes: u64,
}

/// Complete diagnostic output emitted by `forge doctor`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub system: SystemInfo,
    pub gpus: Vec<GpuDeviceInfo>,
    pub backends: Vec<BackendStatus>,
    pub capabilities: Vec<CapabilityResult>,
    pub missing_dependencies: Vec<String>,
}
