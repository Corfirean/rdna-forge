pub mod adlx;
pub mod hip;
pub mod os;
pub mod registry;
pub mod vulkan;

use forge_core::{
    lookup_amd_pci_device, select_preferred_device, GfxTarget, GpuArchitecture, GpuDeviceInfo,
    SystemInfo, WindowsSystemAudit, AMD_PCI_VENDOR_ID,
};

pub use adlx::{probe_adlx, AdlxProbeReport};
pub use hip::{probe_hip, HipDeviceProbe, HipDevicePropR0600, HipProbeReport};
pub use os::{build_windows_system_audit, detect_system_info, detect_windows_system_audit};
pub use registry::{query_amd_registry_adapters, RegistryAdapterInfo};
pub use vulkan::{probe_vulkan, VulkanDeviceProbe, VulkanProbeReport};

/// Complete raw discovery snapshot across OS, Vulkan, HIP, Registry, and ADLX.
#[derive(Debug, Clone)]
pub struct DiscoverySnapshot {
    pub system: SystemInfo,
    pub gpus: Vec<GpuDeviceInfo>,
    pub vulkan: VulkanProbeReport,
    pub hip: HipProbeReport,
    pub adlx: AdlxProbeReport,
    pub registry_adapters: Vec<RegistryAdapterInfo>,
    pub windows_audit: WindowsSystemAudit,
}

/// Discovers host OS/CPU/RAM metadata.
pub fn discover_system() -> SystemInfo {
    detect_system_info()
}

/// Discovers and correlates GPUs across Vulkan, HIP, and the Windows Registry.
pub fn discover_gpus() -> Vec<GpuDeviceInfo> {
    discover_all().gpus
}

/// Runs full multi-source discovery and returns both raw probe reports and correlated GPU records.
pub fn discover_all() -> DiscoverySnapshot {
    let system = detect_system_info();
    let vulkan = probe_vulkan();
    let hip = probe_hip();
    let adlx = probe_adlx();
    let registry_adapters = query_amd_registry_adapters();

    let gpus = correlate_devices(&vulkan.devices, &hip, &registry_adapters);
    let primary_rebar = gpus.first().and_then(|g| g.rebar_enabled);
    let windows_audit = detect_windows_system_audit(primary_rebar);

    DiscoverySnapshot {
        system,
        gpus,
        vulkan,
        hip,
        adlx,
        registry_adapters,
        windows_audit,
    }
}

/// Correlates Vulkan, HIP, and Windows Registry probes into canonical GPU records.
///
/// Exposed for deterministic unit testing with synthetic probe inputs.
pub fn correlate_devices(
    vulkan_devices: &[VulkanDeviceProbe],
    hip_report: &HipProbeReport,
    registry_adapters: &[RegistryAdapterInfo],
) -> Vec<GpuDeviceInfo> {
    let mut used_hip = vec![false; hip_report.devices.len()];
    let mut used_reg = vec![false; registry_adapters.len()];
    let mut devices = Vec::new();

    for vk_dev in vulkan_devices {
        let hip_idx = find_matching_hip(vk_dev, &hip_report.devices, &used_hip);
        let hip_dev = hip_idx.map(|i| {
            used_hip[i] = true;
            &hip_report.devices[i]
        });

        let reg_idx = find_matching_registry(vk_dev, registry_adapters, &used_reg);
        let reg_dev = reg_idx.map(|i| {
            used_reg[i] = true;
            &registry_adapters[i]
        });

        devices.push(build_device_info(
            devices.len(),
            Some(vk_dev),
            hip_dev,
            hip_report.runtime_version.clone(),
            reg_dev,
        ));
    }

    // Add any HIP-only devices that were not exposed via Vulkan.
    for (idx, hip_dev) in hip_report.devices.iter().enumerate() {
        if used_hip[idx] {
            continue;
        }
        let reg_dev = registry_adapters
            .iter()
            .enumerate()
            .find(|(ri, r)| !used_reg[*ri] && names_roughly_match(&hip_dev.name, &r.description))
            .map(|(ri, r)| {
                used_reg[ri] = true;
                r
            });

        devices.push(build_device_info(
            devices.len(),
            None,
            Some(hip_dev),
            hip_report.runtime_version.clone(),
            reg_dev,
        ));
    }

    // Sort with preferred GPU first (discrete AMD first) and re-index deterministically.
    devices.sort_by_key(|d| {
        let is_amd = u8::from(d.vendor_id == AMD_PCI_VENDOR_ID);
        let is_discrete = u8::from(d.discrete);
        let known_arch = u8::from(d.architecture != GpuArchitecture::Unknown);
        (
            std::cmp::Reverse(is_amd),
            std::cmp::Reverse(is_discrete),
            std::cmp::Reverse(known_arch),
            std::cmp::Reverse(d.vram_bytes),
        )
    });

    for (idx, dev) in devices.iter_mut().enumerate() {
        dev.index = idx;
    }

    // Verify preferred selector aligns with index 0 when non-empty.
    let _ = select_preferred_device(&devices);

    devices
}

fn find_matching_hip(
    vk_dev: &VulkanDeviceProbe,
    hip_devices: &[HipDeviceProbe],
    used: &[bool],
) -> Option<usize> {
    if vk_dev.vendor_id != AMD_PCI_VENDOR_ID {
        return None;
    }

    // 1. Exact PCI bus ID match if available on both sides.
    if let Some(ref vk_pci) = vk_dev.pci_bus_id {
        if let Some((idx, _)) = hip_devices
            .iter()
            .enumerate()
            .find(|(i, h)| !used[*i] && h.pci_bus_id.as_deref() == Some(vk_pci.as_str()))
        {
            return Some(idx);
        }
    }

    // 2. Match by name + discrete/integrated role.
    if let Some((idx, _)) = hip_devices.iter().enumerate().find(|(i, h)| {
        !used[*i]
            && names_roughly_match(&vk_dev.name, &h.name)
            && h.integrated
                .map(|int| int != vk_dev.discrete)
                .unwrap_or(true)
    }) {
        return Some(idx);
    }

    // 3. Fallback: match by name only.
    hip_devices
        .iter()
        .enumerate()
        .find(|(i, h)| !used[*i] && names_roughly_match(&vk_dev.name, &h.name))
        .map(|(i, _)| i)
}

fn find_matching_registry(
    vk_dev: &VulkanDeviceProbe,
    reg_adapters: &[RegistryAdapterInfo],
    used: &[bool],
) -> Option<usize> {
    if vk_dev.vendor_id != AMD_PCI_VENDOR_ID {
        return None;
    }

    if let Some((idx, _)) = reg_adapters.iter().enumerate().find(|(i, r)| {
        !used[*i] && r.vendor_id == Some(vk_dev.vendor_id) && r.device_id == Some(vk_dev.device_id)
    }) {
        return Some(idx);
    }

    reg_adapters
        .iter()
        .enumerate()
        .find(|(i, r)| !used[*i] && names_roughly_match(&vk_dev.name, &r.description))
        .map(|(i, _)| i)
}

fn names_roughly_match(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.to_ascii_lowercase()
            .replace("(tm)", "")
            .replace("(r)", "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let na = norm(a);
    let nb = norm(b);
    na == nb || na.contains(&nb) || nb.contains(&na)
}

fn build_device_info(
    index: usize,
    vk_dev: Option<&VulkanDeviceProbe>,
    hip_dev: Option<&HipDeviceProbe>,
    hip_runtime_version: Option<String>,
    reg_dev: Option<&RegistryAdapterInfo>,
) -> GpuDeviceInfo {
    let vendor_id = vk_dev
        .map(|v| v.vendor_id)
        .or_else(|| reg_dev.and_then(|r| r.vendor_id))
        .unwrap_or(if hip_dev.is_some() {
            AMD_PCI_VENDOR_ID
        } else {
            0
        });
    let device_id = vk_dev
        .map(|v| v.device_id)
        .or_else(|| reg_dev.and_then(|r| r.device_id))
        .unwrap_or(0);

    let known = (vendor_id == AMD_PCI_VENDOR_ID)
        .then(|| lookup_amd_pci_device(device_id))
        .flatten();

    let gfx_target = hip_dev
        .and_then(|h| h.gcn_arch_name.as_deref())
        .and_then(|raw| GfxTarget::parse(raw).ok())
        .or_else(|| known.and_then(|k| GfxTarget::parse(k.gfx_target).ok()));

    let architecture = gfx_target
        .as_ref()
        .map(|g| g.architecture)
        .or_else(|| known.map(|k| k.architecture))
        .unwrap_or(GpuArchitecture::Unknown);

    let discrete = vk_dev
        .map(|v| v.discrete)
        .or_else(|| hip_dev.and_then(|h| h.integrated.map(|i| !i)))
        .or_else(|| known.map(|k| k.discrete))
        .unwrap_or(false);

    // Note: For discrete GPUs, HIP reports true total physical VRAM (whereas Vulkan
    // DEVICE_LOCAL heap subtracts reserved/BAR slices, and Windows WMI caps at 4 GiB).
    // For APUs/integrated GPUs, prefer registry dedicated VRAM or Vulkan device_local if HIP reports GTT pool.
    let vram_bytes = if discrete {
        hip_dev
            .map(|h| h.total_vram_bytes)
            .filter(|&b| b > 0)
            .or_else(|| reg_dev.and_then(|r| r.vram_bytes))
            .or_else(|| vk_dev.map(|v| v.device_local_bytes))
            .unwrap_or(0)
    } else {
        reg_dev
            .and_then(|r| r.vram_bytes)
            .or_else(|| vk_dev.map(|v| v.device_local_bytes))
            .or_else(|| hip_dev.map(|h| h.total_vram_bytes))
            .unwrap_or(0)
    };

    let wgp_count = hip_dev.and_then(|h| h.multi_processor_count);
    let compute_units =
        vk_dev
            .and_then(|v| v.amd_compute_units)
            .or_else(|| match (architecture, wgp_count) {
                (
                    GpuArchitecture::Rdna1
                    | GpuArchitecture::Rdna2
                    | GpuArchitecture::Rdna3
                    | GpuArchitecture::Rdna35
                    | GpuArchitecture::Rdna4,
                    Some(wgps),
                ) => Some(wgps * 2),
                (_, cus) => cus,
            });

    let name = vk_dev
        .map(|v| v.name.clone())
        .or_else(|| hip_dev.map(|h| h.name.clone()))
        .or_else(|| reg_dev.map(|r| r.description.clone()))
        .unwrap_or_else(|| "Unknown GPU".to_string());

    GpuDeviceInfo {
        index,
        name,
        vendor_id,
        device_id,
        pci_bus_id: hip_dev
            .and_then(|h| h.pci_bus_id.clone())
            .or_else(|| vk_dev.and_then(|v| v.pci_bus_id.clone())),
        architecture,
        gfx_target,
        discrete,
        vram_bytes,
        shared_memory_per_block_bytes: hip_dev.and_then(|h| h.shared_mem_per_block_bytes),
        vulkan_shared_memory_bytes: vk_dev.map(|v| v.max_compute_shared_memory_bytes),
        compute_units,
        wgp_count,
        warp_size: hip_dev
            .and_then(|h| h.warp_size)
            .or_else(|| vk_dev.map(|v| v.subgroup_size)),
        max_buffer_size_bytes: vk_dev.and_then(|v| v.max_buffer_size_bytes),
        max_memory_allocation_bytes: vk_dev.map(|v| v.max_memory_allocation_bytes),
        rebar_enabled: vk_dev
            .map(|v| v.rebar_enabled)
            .or_else(|| hip_dev.and_then(|h| h.is_large_bar)),
        driver_version: reg_dev.and_then(|r| r.driver_version.clone()),
        adrenalin_version: reg_dev.and_then(|r| r.adrenalin_version.clone()),
        vulkan_api_version: vk_dev.map(|v| v.api_version.clone()),
        vulkan_driver_info: vk_dev.and_then(|v| v.driver_info.clone()),
        vulkan_cooperative_matrix: vk_dev.map(|v| v.cooperative_matrix).unwrap_or(false),
        vulkan_fp16: vk_dev.map(|v| v.fp16_supported).unwrap_or(false),
        vulkan_int8: vk_dev.map(|v| v.int8_supported).unwrap_or(false),
        vulkan_int8_dot_product: vk_dev
            .map(|v| v.int8_dot_product_4x8_signed)
            .unwrap_or(false),
        hip_available: hip_dev.is_some(),
        hip_runtime_version: hip_dev.and(hip_runtime_version),
        core_clock_mhz: hip_dev.and_then(|h| h.clock_mhz),
        memory_clock_mhz: hip_dev.and_then(|h| h.memory_clock_mhz),
        memory_bus_width_bits: hip_dev.and_then(|h| h.memory_bus_width_bits),
        l2_cache_bytes: hip_dev.and_then(|h| h.l2_cache_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adlx::decode_adlx_full_version;
    use crate::hip::format_hip_version;
    use crate::registry::parse_pci_ids_from_matching_id;

    #[test]
    fn hip_r0600_struct_layout_matches_amdhip64_abi() {
        assert_eq!(
            std::mem::offset_of!(HipDevicePropR0600, gcn_arch_name),
            1160
        );
        assert_eq!(std::mem::size_of::<HipDevicePropR0600>(), 1472);
    }

    #[test]
    fn pci_matching_id_extraction_works() {
        assert_eq!(
            parse_pci_ids_from_matching_id(r"PCI\VEN_1002&DEV_73AF&REV_C0"),
            (Some(0x1002), Some(0x73AF))
        );
        assert_eq!(
            parse_pci_ids_from_matching_id(r"pci\ven_1002&dev_164e&subsys_7d781462"),
            (Some(0x1002), Some(0x164E))
        );
        assert_eq!(
            parse_pci_ids_from_matching_id("ROOT\\DISPLAY"),
            (None, None)
        );
    }

    #[test]
    fn version_formatters_decode_packed_values() {
        assert_eq!(format_hip_version(60450101), "6.4.50101");
        assert_eq!(format_hip_version(0), "unknown");
        assert_eq!(decode_adlx_full_version(281492156645498), "1.4.1.122");
    }

    #[test]
    fn device_correlation_merges_vulkan_hip_and_registry() {
        let vk = vec![
            VulkanDeviceProbe {
                index: 0,
                name: "AMD Radeon(TM) Graphics".to_string(),
                vendor_id: 0x1002,
                device_id: 0x164E,
                device_type: "integrated".to_string(),
                discrete: false,
                api_version: "1.4.315".to_string(),
                driver_name: Some("AMD proprietary driver".to_string()),
                driver_info: Some("26.11.2".to_string()),
                pci_bus_id: None,
                device_local_bytes: 536_870_912,
                rebar_bytes: 268_435_456,
                rebar_enabled: false,
                max_memory_allocation_bytes: 2_147_483_648,
                max_buffer_size_bytes: Some(2_147_483_648),
                max_compute_shared_memory_bytes: 32768,
                subgroup_size: 64,
                min_subgroup_size: Some(32),
                max_subgroup_size: Some(64),
                amd_compute_units: None,
                fp16_supported: true,
                int8_supported: true,
                int8_dot_product_4x8_signed: true,
                storage_16bit: true,
                storage_8bit: true,
                cooperative_matrix: true,
                compute_queue_supported: true,
            },
            VulkanDeviceProbe {
                index: 1,
                name: "AMD Radeon RX 6900 XT".to_string(),
                vendor_id: 0x1002,
                device_id: 0x73AF,
                device_type: "discrete".to_string(),
                discrete: true,
                api_version: "1.4.315".to_string(),
                driver_name: Some("AMD proprietary driver".to_string()),
                driver_info: Some("26.11.2".to_string()),
                pci_bus_id: None,
                device_local_bytes: 16_894_656_512,
                rebar_bytes: 268_435_456,
                rebar_enabled: false,
                max_memory_allocation_bytes: 2_147_483_648,
                max_buffer_size_bytes: Some(2_147_483_648),
                max_compute_shared_memory_bytes: 32768,
                subgroup_size: 64,
                min_subgroup_size: Some(32),
                max_subgroup_size: Some(64),
                amd_compute_units: Some(80),
                fp16_supported: true,
                int8_supported: true,
                int8_dot_product_4x8_signed: true,
                storage_16bit: true,
                storage_8bit: true,
                cooperative_matrix: true,
                compute_queue_supported: true,
            },
        ];

        let hip = HipProbeReport {
            available: true,
            loaded_library: Some("amdhip64_6.dll".to_string()),
            runtime_version: Some("6.4.50101".to_string()),
            driver_version: Some("6.4.50101".to_string()),
            devices: vec![
                HipDeviceProbe {
                    hip_index: 0,
                    name: "AMD Radeon RX 6900 XT".to_string(),
                    gcn_arch_name: Some("gfx1030".to_string()),
                    pci_bus_id: Some("0000:03:00.0".to_string()),
                    total_vram_bytes: 17_163_091_968,
                    free_vram_bytes: Some(17_000_000_000),
                    shared_mem_per_block_bytes: Some(65536),
                    multi_processor_count: Some(40),
                    warp_size: Some(32),
                    clock_mhz: Some(2235),
                    memory_clock_mhz: Some(1000),
                    memory_bus_width_bits: None,
                    l2_cache_bytes: Some(4_194_304),
                    integrated: Some(false),
                    is_large_bar: Some(false),
                    compute_capability: Some((10, 3)),
                },
                HipDeviceProbe {
                    hip_index: 1,
                    name: "AMD Radeon(TM) Graphics".to_string(),
                    gcn_arch_name: Some("gfx1036".to_string()),
                    pci_bus_id: Some("0000:12:00.0".to_string()),
                    total_vram_bytes: 13_056_000_000,
                    free_vram_bytes: Some(12_000_000_000),
                    shared_mem_per_block_bytes: Some(65536),
                    multi_processor_count: Some(1),
                    warp_size: Some(32),
                    clock_mhz: Some(2200),
                    memory_clock_mhz: Some(2000),
                    memory_bus_width_bits: None,
                    l2_cache_bytes: Some(4_194_304),
                    integrated: Some(true),
                    is_large_bar: Some(false),
                    compute_capability: Some((10, 3)),
                },
            ],
            error: None,
        };

        let reg = vec![RegistryAdapterInfo {
            description: "AMD Radeon RX 6900 XT".to_string(),
            driver_version: Some("32.0.21045.29003".to_string()),
            adrenalin_version: Some("26.11.2".to_string()),
            release_version: Some("25.10.45.29".to_string()),
            matching_device_id: Some(r"PCI\VEN_1002&DEV_73AF&REV_C0".to_string()),
            vendor_id: Some(0x1002),
            device_id: Some(0x73AF),
            vram_bytes: Some(17_163_091_968),
        }];

        let correlated = correlate_devices(&vk, &hip, &reg);
        assert_eq!(correlated.len(), 2);
        // Discrete RX 6900 XT is sorted first even though it was index 1 in Vulkan.
        assert_eq!(correlated[0].name, "AMD Radeon RX 6900 XT");
        assert_eq!(correlated[0].gfx_target.as_ref().unwrap().base, "gfx1030");
        assert_eq!(correlated[0].architecture, GpuArchitecture::Rdna2);
        assert_eq!(correlated[0].compute_units, Some(80));
        assert_eq!(correlated[0].wgp_count, Some(40));
        assert_eq!(correlated[0].adrenalin_version.as_deref(), Some("26.11.2"));
        assert_eq!(correlated[0].pci_bus_id.as_deref(), Some("0000:03:00.0"));

        assert_eq!(correlated[1].name, "AMD Radeon(TM) Graphics");
        assert!(!correlated[1].discrete);
        assert_eq!(correlated[1].gfx_target.as_ref().unwrap().base, "gfx1036");
    }
}
