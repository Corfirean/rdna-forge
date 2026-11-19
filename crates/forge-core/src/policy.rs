use crate::architecture::{GpuArchitecture, AMD_PCI_VENDOR_ID};
use crate::error::{ForgeError, ForgeResult};
use crate::status::GpuDeviceInfo;

/// Hardware capability and allocation requirements checked before kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KernelRequirements {
    pub required_vram_bytes: u64,
    pub max_single_buffer_bytes: u64,
    pub require_amd: bool,
    pub require_fp16: bool,
    pub require_bf16: bool,
    pub require_fp8: bool,
    pub require_int8_dp4a: bool,
    pub require_wmma: bool,
    pub require_mfma: bool,
}

/// Validates that a GPU satisfies all capability and memory requirements before running a kernel.
pub fn validate_kernel_execution(
    device: &GpuDeviceInfo,
    req: &KernelRequirements,
) -> ForgeResult<()> {
    if req.require_amd && device.vendor_id != AMD_PCI_VENDOR_ID {
        return Err(ForgeError::UnsupportedCapability(format!(
            "device '{}' (vendor 0x{:04x}) is not an AMD GPU",
            device.name, device.vendor_id
        )));
    }

    if let Some(max_buf) = device.max_buffer_size_bytes {
        if req.max_single_buffer_bytes > max_buf {
            return Err(ForgeError::BufferTooLarge {
                requested_bytes: req.max_single_buffer_bytes,
                max_bytes: max_buf,
            });
        }
    }

    if device.vram_bytes > 0 && req.required_vram_bytes > device.vram_bytes {
        return Err(ForgeError::InsufficientVram {
            requested_bytes: req.required_vram_bytes,
            available_bytes: device.vram_bytes,
        });
    }

    let arch_caps = device.architecture.arch_capabilities();

    if req.require_fp16 && !(device.vulkan_fp16 || arch_caps.has_native_fp16) {
        return Err(ForgeError::UnsupportedCapability(format!(
            "FP16 compute is not supported on '{}'",
            device.name
        )));
    }

    if req.require_bf16 && !arch_caps.has_native_bf16 {
        return Err(ForgeError::UnsupportedCapability(format!(
            "native BF16 math is not supported on {} ({})",
            device.name, device.architecture
        )));
    }

    if req.require_fp8 && !arch_caps.has_native_fp8 {
        return Err(ForgeError::UnsupportedCapability(format!(
            "native FP8 math is not supported on {} ({})",
            device.name, device.architecture
        )));
    }

    if req.require_int8_dp4a
        && !(device.vulkan_int8_dot_product || device.vulkan_int8 || arch_caps.has_dp4a_i8)
    {
        return Err(ForgeError::UnsupportedCapability(format!(
            "INT8 DP4A compute is not supported on '{}'",
            device.name
        )));
    }

    if req.require_wmma && !arch_caps.has_wmma {
        return Err(ForgeError::UnsupportedCapability(format!(
            "WMMA matrix instructions are not supported on {} ({})",
            device.name, device.architecture
        )));
    }

    if req.require_mfma && !arch_caps.has_mfma {
        return Err(ForgeError::UnsupportedCapability(format!(
            "MFMA matrix instructions are not supported on {} ({})",
            device.name, device.architecture
        )));
    }

    Ok(())
}

/// Selects the preferred GPU device using deterministic ordering:
/// discrete AMD > integrated AMD > discrete other > integrated other,
/// breaking ties by largest VRAM then lowest discovery index.
pub fn select_preferred_device(devices: &[GpuDeviceInfo]) -> ForgeResult<&GpuDeviceInfo> {
    devices
        .iter()
        .max_by_key(|d| {
            let is_amd = u8::from(d.vendor_id == AMD_PCI_VENDOR_ID);
            let is_discrete = u8::from(d.discrete);
            let known_arch = u8::from(d.architecture != GpuArchitecture::Unknown);
            (
                is_amd,
                is_discrete,
                known_arch,
                d.vram_bytes,
                usize::MAX - d.index,
            )
        })
        .ok_or_else(|| ForgeError::DeviceNotFound("no GPU devices discovered".to_string()))
}
