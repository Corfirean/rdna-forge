use std::collections::HashSet;
use std::ffi::{CStr, CString};

use ash::vk;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Vulkan physical device metadata queried dynamically via `vulkan-1.dll`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanDeviceProbe {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub discrete: bool,
    pub api_version: String,
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    pub pci_bus_id: Option<String>,
    pub device_local_bytes: u64,
    pub rebar_bytes: u64,
    pub rebar_enabled: bool,
    pub max_memory_allocation_bytes: u64,
    pub max_buffer_size_bytes: Option<u64>,
    pub max_compute_shared_memory_bytes: u32,
    pub subgroup_size: u32,
    pub min_subgroup_size: Option<u32>,
    pub max_subgroup_size: Option<u32>,
    pub amd_compute_units: Option<u32>,
    pub fp16_supported: bool,
    pub int8_supported: bool,
    pub storage_16bit: bool,
    pub storage_8bit: bool,
    pub cooperative_matrix: bool,
    pub compute_queue_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanProbeReport {
    pub available: bool,
    pub instance_version: Option<String>,
    pub devices: Vec<VulkanDeviceProbe>,
    pub error: Option<String>,
}

pub fn format_vk_version(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

fn c_char_slice_to_string(buf: &[std::ffi::c_char]) -> String {
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
    cstr.to_string_lossy().trim().to_string()
}

/// Probes the system Vulkan loader and enumerates every physical device.
pub fn probe_vulkan() -> VulkanProbeReport {
    let entry = match unsafe { ash::Entry::load() } {
        Ok(entry) => entry,
        Err(err) => {
            return VulkanProbeReport {
                available: false,
                instance_version: None,
                devices: Vec::new(),
                error: Some(format!("failed to load vulkan-1.dll: {err}")),
            };
        }
    };

    let raw_instance_version = unsafe { entry.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or(vk::API_VERSION_1_0);
    let instance_version = format_vk_version(raw_instance_version);

    let app_name = CString::new("RDNA Forge").expect("static string has no NUL");
    let target_api = if raw_instance_version >= vk::API_VERSION_1_3 {
        vk::API_VERSION_1_3
    } else if raw_instance_version >= vk::API_VERSION_1_2 {
        vk::API_VERSION_1_2
    } else {
        vk::API_VERSION_1_1
    };

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&app_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(target_api);

    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&instance_info, None) } {
        Ok(inst) => inst,
        Err(err) => {
            return VulkanProbeReport {
                available: false,
                instance_version: Some(instance_version),
                devices: Vec::new(),
                error: Some(format!("vkCreateInstance failed: {err}")),
            };
        }
    };

    let devices_result = unsafe { instance.enumerate_physical_devices() };
    let physical_devices = match devices_result {
        Ok(list) => list,
        Err(err) => {
            unsafe { instance.destroy_instance(None) };
            return VulkanProbeReport {
                available: false,
                instance_version: Some(instance_version),
                devices: Vec::new(),
                error: Some(format!("vkEnumeratePhysicalDevices failed: {err}")),
            };
        }
    };

    let mut probed = Vec::with_capacity(physical_devices.len());
    for (idx, pd) in physical_devices.into_iter().enumerate() {
        probed.push(inspect_physical_device(&instance, pd, idx));
    }

    unsafe { instance.destroy_instance(None) };

    VulkanProbeReport {
        available: true,
        instance_version: Some(instance_version),
        devices: probed,
        error: None,
    }
}

fn inspect_physical_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    index: usize,
) -> VulkanDeviceProbe {
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let exts: HashSet<String> = unsafe { instance.enumerate_device_extension_properties(pd) }
        .unwrap_or_default()
        .into_iter()
        .map(|ext| c_char_slice_to_string(&ext.extension_name))
        .collect();

    let has_driver_props =
        props.api_version >= vk::API_VERSION_1_2 || exts.contains("VK_KHR_driver_properties");
    let has_vk13 = props.api_version >= vk::API_VERSION_1_3;
    let has_subgroup_ctrl = !has_vk13 && exts.contains("VK_EXT_subgroup_size_control");
    let has_maint4 = !has_vk13 && exts.contains("VK_KHR_maintenance4");
    let has_amd_core = exts.contains("VK_AMD_shader_core_properties");
    let has_pci_bus = exts.contains("VK_EXT_pci_bus_info");
    let has_f16_i8 =
        props.api_version >= vk::API_VERSION_1_2 || exts.contains("VK_KHR_shader_float16_int8");
    let has_16bit_storage =
        props.api_version >= vk::API_VERSION_1_1 || exts.contains("VK_KHR_16bit_storage");
    let has_8bit_storage =
        props.api_version >= vk::API_VERSION_1_2 || exts.contains("VK_KHR_8bit_storage");
    let has_coop_mat = exts.contains("VK_KHR_cooperative_matrix");

    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
    let mut maint3_props = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut vk13_props = vk::PhysicalDeviceVulkan13Properties::default();
    let mut subgroup_ctrl_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut maint4_props = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut amd_core_props = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
    let mut pci_bus_props = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();

    let mut props2 = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut subgroup_props)
        .push_next(&mut maint3_props);
    if has_driver_props {
        props2 = props2.push_next(&mut driver_props);
    }
    if has_vk13 {
        props2 = props2.push_next(&mut vk13_props);
    } else {
        if has_subgroup_ctrl {
            props2 = props2.push_next(&mut subgroup_ctrl_props);
        }
        if has_maint4 {
            props2 = props2.push_next(&mut maint4_props);
        }
    }
    if has_amd_core {
        props2 = props2.push_next(&mut amd_core_props);
    }
    if has_pci_bus {
        props2 = props2.push_next(&mut pci_bus_props);
    }

    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

    let mut f16_i8_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut storage16_feat = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut storage8_feat = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut coop_mat_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();

    let mut feat2 = vk::PhysicalDeviceFeatures2::default();
    if has_f16_i8 {
        feat2 = feat2.push_next(&mut f16_i8_feat);
    }
    if has_16bit_storage {
        feat2 = feat2.push_next(&mut storage16_feat);
    }
    if has_8bit_storage {
        feat2 = feat2.push_next(&mut storage8_feat);
    }
    if has_coop_mat {
        feat2 = feat2.push_next(&mut coop_mat_feat);
    }

    unsafe { instance.get_physical_device_features2(pd, &mut feat2) };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
    let mut device_local_bytes = 0u64;
    let mut rebar_bytes = 0u64;
    let heap_count = mem_props.memory_heap_count as usize;

    for heap_idx in 0..heap_count {
        let heap = mem_props.memory_heaps[heap_idx];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            device_local_bytes = device_local_bytes.saturating_add(heap.size);
        }
    }

    for type_idx in 0..(mem_props.memory_type_count as usize) {
        let mt = mem_props.memory_types[type_idx];
        let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        if mt.property_flags.contains(want) {
            let heap = mem_props.memory_heaps[mt.heap_index as usize];
            rebar_bytes = rebar_bytes.max(heap.size);
        }
    }

    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    let compute_queue_supported = queue_families
        .iter()
        .any(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE));

    let device_type = match props.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
    .to_string();

    let (min_subgroup_size, max_subgroup_size, max_buffer_size_bytes) = if has_vk13 {
        (
            Some(vk13_props.min_subgroup_size),
            Some(vk13_props.max_subgroup_size),
            Some(vk13_props.max_buffer_size),
        )
    } else {
        (
            has_subgroup_ctrl.then_some(subgroup_ctrl_props.min_subgroup_size),
            has_subgroup_ctrl.then_some(subgroup_ctrl_props.max_subgroup_size),
            has_maint4.then_some(maint4_props.max_buffer_size),
        )
    };

    let amd_compute_units = if has_amd_core {
        let cus = amd_core_props
            .shader_engine_count
            .saturating_mul(amd_core_props.shader_arrays_per_engine_count)
            .saturating_mul(amd_core_props.compute_units_per_shader_array);
        (cus > 0).then_some(cus)
    } else {
        None
    };

    let pci_bus_id = has_pci_bus.then(|| {
        format!(
            "{:04x}:{:02x}:{:02x}.{}",
            pci_bus_props.pci_domain,
            pci_bus_props.pci_bus,
            pci_bus_props.pci_device,
            pci_bus_props.pci_function
        )
    });

    let name = c_char_slice_to_string(&props.device_name);
    debug!(
        device = %name,
        vendor = props.vendor_id,
        device_id = props.device_id,
        "probed Vulkan physical device"
    );

    VulkanDeviceProbe {
        index,
        name,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        device_type,
        discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        api_version: format_vk_version(props.api_version),
        driver_name: has_driver_props.then(|| c_char_slice_to_string(&driver_props.driver_name)),
        driver_info: has_driver_props.then(|| c_char_slice_to_string(&driver_props.driver_info)),
        pci_bus_id,
        device_local_bytes,
        rebar_bytes,
        rebar_enabled: rebar_bytes >= 1024 * 1024 * 1024,
        max_memory_allocation_bytes: maint3_props.max_memory_allocation_size,
        max_buffer_size_bytes,
        max_compute_shared_memory_bytes: props.limits.max_compute_shared_memory_size,
        subgroup_size: subgroup_props.subgroup_size,
        min_subgroup_size,
        max_subgroup_size,
        amd_compute_units,
        fp16_supported: f16_i8_feat.shader_float16 == vk::TRUE,
        int8_supported: f16_i8_feat.shader_int8 == vk::TRUE,
        storage_16bit: storage16_feat.storage_buffer16_bit_access == vk::TRUE,
        storage_8bit: storage8_feat.storage_buffer8_bit_access == vk::TRUE,
        cooperative_matrix: coop_mat_feat.cooperative_matrix == vk::TRUE,
        compute_queue_supported,
    }
}
