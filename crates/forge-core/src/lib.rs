pub mod architecture;
pub mod error;
pub mod policy;
pub mod profile;
pub mod status;

pub use architecture::{
    lookup_amd_pci_device, ArchitectureCapabilities, GfxTarget, GpuArchitecture, KnownAmdGpu,
    AMD_PCI_VENDOR_ID,
};
pub use error::{ForgeError, ForgeResult};
pub use policy::{select_preferred_device, validate_kernel_execution, KernelRequirements};
pub use profile::HardwareProfile;
pub use status::{
    BackendKind, BackendStatus, CapabilityResult, DiagnosticReport, GpuDeviceInfo, SupportStatus,
    SystemInfo,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_device(
        index: usize,
        name: &str,
        vendor_id: u32,
        device_id: u32,
        arch: GpuArchitecture,
        discrete: bool,
        vram_bytes: u64,
    ) -> GpuDeviceInfo {
        GpuDeviceInfo {
            index,
            name: name.to_string(),
            vendor_id,
            device_id,
            pci_bus_id: None,
            architecture: arch,
            gfx_target: None,
            discrete,
            vram_bytes,
            shared_memory_per_block_bytes: Some(65536),
            vulkan_shared_memory_bytes: Some(32768),
            compute_units: Some(80),
            wgp_count: Some(40),
            warp_size: Some(32),
            max_buffer_size_bytes: Some(2 * 1024 * 1024 * 1024),
            max_memory_allocation_bytes: Some(2 * 1024 * 1024 * 1024),
            rebar_enabled: Some(false),
            driver_version: None,
            adrenalin_version: None,
            vulkan_api_version: None,
            vulkan_driver_info: None,
            vulkan_cooperative_matrix: false,
            vulkan_fp16: true,
            vulkan_int8: true,
            vulkan_int8_dot_product: true,
            hip_available: true,
            hip_runtime_version: None,
            core_clock_mhz: None,
            memory_clock_mhz: None,
            memory_bus_width_bits: None,
            l2_cache_bytes: None,
        }
    }

    #[test]
    fn gfx_target_parses_valid_variants() {
        let t = GfxTarget::parse("gfx1030").unwrap();
        assert_eq!(t.base, "gfx1030");
        assert_eq!(t.major, 10);
        assert_eq!(t.minor, 3);
        assert_eq!(t.stepping, 0);
        assert_eq!(t.architecture, GpuArchitecture::Rdna2);
        assert!(t.qualifiers.is_empty());

        let t2 = GfxTarget::parse("GFX1100:xnack-").unwrap();
        assert_eq!(t2.base, "gfx1100");
        assert_eq!(t2.architecture, GpuArchitecture::Rdna3);
        assert_eq!(t2.qualifiers, vec!["xnack-"]);

        let cdna = GfxTarget::parse("gfx90a").unwrap();
        assert_eq!(cdna.major, 9);
        assert_eq!(cdna.minor, 0);
        assert_eq!(cdna.stepping, 10);
        assert_eq!(cdna.architecture, GpuArchitecture::Cdna2);
    }

    #[test]
    fn gfx_target_rejects_invalid_inputs() {
        for bad in [
            "", "   ", "sm89", "gfx", "gfx12", "gfx12345", "gfx10zz", "gfx1030:",
        ] {
            assert!(
                matches!(GfxTarget::parse(bad), Err(ForgeError::InvalidGfxTarget(_))),
                "expected InvalidGfxTarget for '{bad}'"
            );
        }
    }

    #[test]
    fn pci_lookup_identifies_known_and_ignores_unknown() {
        let rx6900 = lookup_amd_pci_device(0x73AF).expect("0x73AF should be known");
        assert_eq!(rx6900.gfx_target, "gfx1030");
        assert_eq!(rx6900.architecture, GpuArchitecture::Rdna2);
        assert!(rx6900.discrete);

        let raphael = lookup_amd_pci_device(0x164E).expect("0x164E should be known");
        assert_eq!(raphael.gfx_target, "gfx1036");
        assert!(!raphael.discrete);

        assert!(lookup_amd_pci_device(0xDEAD).is_none());
    }

    #[test]
    fn capability_gating_rejects_wmma_bf16_and_oversized_buffers_on_rdna2() {
        let rx6900 = sample_device(
            0,
            "AMD Radeon RX 6900 XT",
            AMD_PCI_VENDOR_ID,
            0x73AF,
            GpuArchitecture::Rdna2,
            true,
            16 * 1024 * 1024 * 1024,
        );

        let ok_req = KernelRequirements {
            required_vram_bytes: 256 * 1024 * 1024,
            max_single_buffer_bytes: 128 * 1024 * 1024,
            require_amd: true,
            require_fp16: true,
            require_int8_dp4a: true,
            ..Default::default()
        };
        assert!(validate_kernel_execution(&rx6900, &ok_req).is_ok());

        let wmma_req = KernelRequirements {
            require_wmma: true,
            ..Default::default()
        };
        assert!(matches!(
            validate_kernel_execution(&rx6900, &wmma_req),
            Err(ForgeError::UnsupportedCapability(_))
        ));

        let bf16_req = KernelRequirements {
            require_bf16: true,
            ..Default::default()
        };
        assert!(matches!(
            validate_kernel_execution(&rx6900, &bf16_req),
            Err(ForgeError::UnsupportedCapability(_))
        ));

        let buf_req = KernelRequirements {
            max_single_buffer_bytes: 3 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert!(matches!(
            validate_kernel_execution(&rx6900, &buf_req),
            Err(ForgeError::BufferTooLarge { .. })
        ));

        let vram_req = KernelRequirements {
            required_vram_bytes: 32 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert!(matches!(
            validate_kernel_execution(&rx6900, &vram_req),
            Err(ForgeError::InsufficientVram { .. })
        ));
    }

    #[test]
    fn profile_parses_rx6900xt_and_rejects_zero_cus() {
        let raw = include_str!("../../../profiles/gfx1030/rx6900xt.toml");
        let profile = HardwareProfile::from_toml_str(raw).expect("rx6900xt profile must parse");
        assert_eq!(profile.device.gfx_target, "gfx1030");
        assert_eq!(profile.compute.compute_units, 80);
        assert!(!profile.capabilities.wmma);

        let broken = raw.replace("compute_units = 80", "compute_units = 0");
        assert!(matches!(
            HardwareProfile::from_toml_str(&broken),
            Err(ForgeError::InvalidProfile(_))
        ));
    }

    #[test]
    fn device_selection_prefers_discrete_amd_gpu() {
        let igpu = sample_device(
            0,
            "AMD Radeon(TM) Graphics",
            AMD_PCI_VENDOR_ID,
            0x164E,
            GpuArchitecture::Rdna2,
            false,
            512 * 1024 * 1024,
        );
        let dgpu = sample_device(
            1,
            "AMD Radeon RX 6900 XT",
            AMD_PCI_VENDOR_ID,
            0x73AF,
            GpuArchitecture::Rdna2,
            true,
            16 * 1024 * 1024 * 1024,
        );
        let non_amd = sample_device(
            2,
            "Other Discrete GPU",
            0x10DE,
            0x0001,
            GpuArchitecture::Unknown,
            true,
            24 * 1024 * 1024 * 1024,
        );

        let devices = [non_amd, igpu, dgpu];
        let chosen = select_preferred_device(&devices).unwrap();
        assert_eq!(chosen.name, "AMD Radeon RX 6900 XT");
        assert_eq!(chosen.index, 1);

        assert!(matches!(
            select_preferred_device(&[]),
            Err(ForgeError::DeviceNotFound(_))
        ));
    }
}
