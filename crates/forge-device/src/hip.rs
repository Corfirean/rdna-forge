use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tracing::debug;

/// Binary-compatible Rust layout for ROCm `hipDeviceProp_tR0600` (`hip_runtime_api.h`).
#[repr(C)]
pub struct HipDevicePropR0600 {
    pub name: [c_char; 256],
    pub uuid: [u8; 16],
    pub luid: [c_char; 8],
    pub luid_device_node_mask: c_uint,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub mem_pitch: usize,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate_khz: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub texture_alignment: usize,
    pub texture_pitch_alignment: usize,
    pub device_overlap: c_int,
    pub multi_processor_count: c_int,
    pub kernel_exec_timeout_enabled: c_int,
    pub integrated: c_int,
    pub can_map_host_memory: c_int,
    pub compute_mode: c_int,
    pub max_texture_1d: c_int,
    pub max_texture_1d_mipmap: c_int,
    pub max_texture_1d_linear: c_int,
    pub max_texture_2d: [c_int; 2],
    pub max_texture_2d_mipmap: [c_int; 2],
    pub max_texture_2d_linear: [c_int; 3],
    pub max_texture_2d_gather: [c_int; 2],
    pub max_texture_3d: [c_int; 3],
    pub max_texture_3d_alt: [c_int; 3],
    pub max_texture_cubemap: c_int,
    pub max_texture_1d_layered: [c_int; 2],
    pub max_texture_2d_layered: [c_int; 3],
    pub max_texture_cubemap_layered: [c_int; 2],
    pub max_surface_1d: c_int,
    pub max_surface_2d: [c_int; 2],
    pub max_surface_3d: [c_int; 3],
    pub max_surface_1d_layered: [c_int; 2],
    pub max_surface_2d_layered: [c_int; 3],
    pub max_surface_cubemap: c_int,
    pub max_surface_cubemap_layered: [c_int; 2],
    pub surface_alignment: usize,
    pub concurrent_kernels: c_int,
    pub ecc_enabled: c_int,
    pub pci_bus_id: c_int,
    pub pci_device_id: c_int,
    pub pci_domain_id: c_int,
    pub tcc_driver: c_int,
    pub async_engine_count: c_int,
    pub unified_addressing: c_int,
    pub memory_clock_rate_khz: c_int,
    pub memory_bus_width: c_int,
    pub l2_cache_size: c_int,
    pub persisting_l2_cache_max_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub stream_priorities_supported: c_int,
    pub global_l1_cache_supported: c_int,
    pub local_l1_cache_supported: c_int,
    pub shared_mem_per_multiprocessor: usize,
    pub regs_per_multiprocessor: c_int,
    pub managed_memory: c_int,
    pub is_multi_gpu_board: c_int,
    pub multi_gpu_board_group_id: c_int,
    pub host_native_atomic_supported: c_int,
    pub single_to_double_precision_perf_ratio: c_int,
    pub pageable_memory_access: c_int,
    pub concurrent_managed_access: c_int,
    pub compute_preemption_supported: c_int,
    pub can_use_host_pointer_for_registered_mem: c_int,
    pub cooperative_launch: c_int,
    pub cooperative_multi_device_launch: c_int,
    pub shared_mem_per_block_optin: usize,
    pub pageable_memory_access_uses_host_page_tables: c_int,
    pub direct_managed_mem_access_from_host: c_int,
    pub max_blocks_per_multi_processor: c_int,
    pub access_policy_max_window_size: c_int,
    pub reserved_shared_mem_per_block: usize,
    pub host_register_supported: c_int,
    pub sparse_hip_array_supported: c_int,
    pub host_register_read_only_supported: c_int,
    pub timeline_semaphore_interop_supported: c_int,
    pub memory_pools_supported: c_int,
    pub gpu_direct_rdma_supported: c_int,
    pub gpu_direct_rdma_flush_writes_options: c_uint,
    pub gpu_direct_rdma_writes_ordering: c_int,
    pub memory_pool_supported_handle_types: c_uint,
    pub deferred_mapping_hip_array_supported: c_int,
    pub ipc_event_supported: c_int,
    pub cluster_launch: c_int,
    pub unified_function_pointers: c_int,
    pub reserved: [c_int; 63],
    pub hip_reserved: [c_int; 32],
    pub gcn_arch_name: [c_char; 256],
    pub max_shared_memory_per_multi_processor: usize,
    pub clock_instruction_rate: c_int,
    pub arch: c_uint,
    pub hdp_mem_flush_cntl: *mut c_void,
    pub hdp_reg_flush_cntl: *mut c_void,
    pub cooperative_multi_device_unmatched_func: c_int,
    pub cooperative_multi_device_unmatched_grid_dim: c_int,
    pub cooperative_multi_device_unmatched_block_dim: c_int,
    pub cooperative_multi_device_unmatched_shared_mem: c_int,
    pub is_large_bar: c_int,
    pub asic_revision: c_int,
}

impl Default for HipDevicePropR0600 {
    fn default() -> Self {
        // SAFETY: `HipDevicePropR0600` is a plain C POD struct of scalars, arrays, and pointers.
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HipDeviceProbe {
    pub hip_index: i32,
    pub name: String,
    pub gcn_arch_name: Option<String>,
    pub pci_bus_id: Option<String>,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: Option<u64>,
    pub shared_mem_per_block_bytes: Option<u64>,
    pub multi_processor_count: Option<u32>,
    pub warp_size: Option<u32>,
    pub clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
    pub memory_bus_width_bits: Option<u32>,
    pub l2_cache_bytes: Option<u64>,
    pub integrated: Option<bool>,
    pub is_large_bar: Option<bool>,
    pub compute_capability: Option<(i32, i32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HipProbeReport {
    pub available: bool,
    pub loaded_library: Option<String>,
    pub runtime_version: Option<String>,
    pub driver_version: Option<String>,
    pub devices: Vec<HipDeviceProbe>,
    pub error: Option<String>,
}

/// Formats HIP packed version integers (`60450101` -> `6.4.50101`).
pub fn format_hip_version(ver: i32) -> String {
    if ver <= 0 {
        return "unknown".to_string();
    }
    let major = ver / 10_000_000;
    let minor = (ver / 100_000) % 100;
    let patch = ver % 100_000;
    format!("{major}.{minor}.{patch}")
}

/// Returns candidate HIP shared library paths in descending search priority.
///
/// On Windows every candidate is an absolute path: SDK locations from `HIP_PATH` / `ROCM_PATH`
/// (kept only if the file exists), then the System32 copy installed by the AMD driver. There is
/// no bare-name entry, so the Windows DLL search order is never consulted. Whether a candidate
/// is acceptable is decided by [`forge_core::load_vendor_library`] (trusted install roots only).
pub fn hip_library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    #[cfg(windows)]
    let dll_names = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
    #[cfg(not(windows))]
    let dll_names = ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"];

    for env_key in ["HIP_PATH", "ROCM_PATH"] {
        if let Ok(base) = std::env::var(env_key) {
            for sub in ["bin", "lib", ""] {
                let dir = if sub.is_empty() {
                    PathBuf::from(&base)
                } else {
                    PathBuf::from(&base).join(sub)
                };
                for name in dll_names {
                    let full = dir.join(name);
                    if full.exists() {
                        candidates.push(full);
                    }
                }
            }
        }
    }

    #[cfg(windows)]
    if let Some(sys) = forge_core::loader::system_directory() {
        for name in dll_names {
            candidates.push(sys.join(name));
        }
    }
    #[cfg(not(windows))]
    for name in dll_names {
        candidates.push(PathBuf::from(name));
    }

    candidates
}

fn c_chars_to_string(buf: &[c_char]) -> String {
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
    cstr.to_string_lossy().trim().to_string()
}

/// Loads the HIP runtime under the secure loading policy, reporting exactly why it failed
/// (not installed vs. untrusted location vs. OS load failure).
pub fn load_hip_library_checked() -> Result<forge_core::LoadedLibrary, forge_core::LibraryLoadError>
{
    let candidates = hip_library_candidates();
    #[cfg(windows)]
    let loaded = forge_core::load_vendor_library("amdhip64", &candidates)?;
    #[cfg(not(windows))]
    let loaded = {
        let mut last = None;
        let mut found = None;
        for c in &candidates {
            match forge_core::load_system_library(&c.to_string_lossy()) {
                Ok(l) => {
                    found = Some(l);
                    break;
                }
                Err(e) => last = Some(e),
            }
        }
        found.ok_or_else(|| {
            last.unwrap_or(forge_core::LibraryLoadError::NotFound {
                name: "libamdhip64".to_string(),
                searched: Vec::new(),
            })
        })?
    };
    debug!(path = %loaded.path.display(), "loaded HIP runtime library");
    Ok(loaded)
}

/// Compatibility wrapper around [`load_hip_library_checked`] that discards the failure reason.
pub fn load_hip_library() -> Option<(PathBuf, libloading::Library)> {
    load_hip_library_checked().ok().map(|l| (l.path, l.library))
}

/// Probes the installed AMD HIP driver/runtime without requiring a full ROCm SDK install.
pub fn probe_hip() -> HipProbeReport {
    let (lib_path, lib) = match load_hip_library_checked() {
        Ok(l) => (l.path, l.library),
        Err(err) => {
            return HipProbeReport {
                available: false,
                loaded_library: None,
                runtime_version: None,
                driver_version: None,
                devices: Vec::new(),
                error: Some(format!("HIP runtime unavailable: {err}")),
            };
        }
    };

    unsafe {
        type GetIntFn = extern "C" fn(*mut c_int) -> c_int;
        type SetDeviceFn = extern "C" fn(c_int) -> c_int;
        type GetNameFn = extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
        type TotalMemFn = extern "C" fn(*mut usize, c_int) -> c_int;
        type MemInfoFn = extern "C" fn(*mut usize, *mut usize) -> c_int;
        type GetPciFn = extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
        type GetPropsR0600Fn = extern "C" fn(*mut HipDevicePropR0600, c_int) -> c_int;

        let runtime_version = lib
            .get::<GetIntFn>(b"hipRuntimeGetVersion\0")
            .ok()
            .and_then(|f| {
                let mut v = 0;
                (f(&mut v) == 0).then(|| format_hip_version(v))
            });

        let driver_version = lib
            .get::<GetIntFn>(b"hipDriverGetVersion\0")
            .ok()
            .and_then(|f| {
                let mut v = 0;
                (f(&mut v) == 0).then(|| format_hip_version(v))
            });

        let get_count = match lib.get::<GetIntFn>(b"hipGetDeviceCount\0") {
            Ok(f) => f,
            Err(err) => {
                return HipProbeReport {
                    available: false,
                    loaded_library: Some(lib_path.display().to_string()),
                    runtime_version,
                    driver_version,
                    devices: Vec::new(),
                    error: Some(format!("missing hipGetDeviceCount export: {err}")),
                };
            }
        };

        let mut count: c_int = 0;
        let rc = get_count(&mut count);
        if rc != 0 || count <= 0 {
            return HipProbeReport {
                available: false,
                loaded_library: Some(lib_path.display().to_string()),
                runtime_version,
                driver_version,
                devices: Vec::new(),
                error: Some(format!("hipGetDeviceCount returned rc={rc}, count={count}")),
            };
        }

        let get_props_r0600 = lib
            .get::<GetPropsR0600Fn>(b"hipGetDevicePropertiesR0600\0")
            .ok();
        let get_name = lib.get::<GetNameFn>(b"hipDeviceGetName\0").ok();
        let get_total_mem = lib.get::<TotalMemFn>(b"hipDeviceTotalMem\0").ok();
        let set_device = lib.get::<SetDeviceFn>(b"hipSetDevice\0").ok();
        let mem_info = lib.get::<MemInfoFn>(b"hipMemGetInfo\0").ok();
        let get_pci = lib.get::<GetPciFn>(b"hipDeviceGetPCIBusId\0").ok();

        let mut devices = Vec::with_capacity(count as usize);
        for dev in 0..count {
            let mut prop = HipDevicePropR0600::default();
            let has_r0600 = get_props_r0600
                .as_ref()
                .map(|f| f(&mut prop, dev) == 0)
                .unwrap_or(false);

            let name = if has_r0600 {
                c_chars_to_string(&prop.name)
            } else if let Some(ref f) = get_name {
                let mut buf = [0 as c_char; 256];
                if f(buf.as_mut_ptr(), 256, dev) == 0 {
                    c_chars_to_string(&buf)
                } else {
                    format!("HIP Device {dev}")
                }
            } else {
                format!("HIP Device {dev}")
            };

            let gcn_arch_name = if has_r0600 {
                let s = c_chars_to_string(&prop.gcn_arch_name);
                (!s.is_empty()).then_some(s)
            } else {
                None
            };

            let pci_bus_id = get_pci.as_ref().and_then(|f| {
                let mut buf = [0 as c_char; 64];
                if f(buf.as_mut_ptr(), 64, dev) == 0 {
                    let s = c_chars_to_string(&buf).to_ascii_lowercase();
                    (!s.is_empty()).then_some(s)
                } else {
                    None
                }
            });

            let mut total_vram_bytes = if has_r0600 {
                prop.total_global_mem as u64
            } else {
                0
            };
            if total_vram_bytes == 0 {
                if let Some(ref f) = get_total_mem {
                    let mut bytes = 0usize;
                    if f(&mut bytes, dev) == 0 {
                        total_vram_bytes = bytes as u64;
                    }
                }
            }

            let free_vram_bytes = match (&set_device, &mem_info) {
                (Some(sd), Some(mi)) if sd(dev) == 0 => {
                    let mut free = 0usize;
                    let mut total = 0usize;
                    (mi(&mut free, &mut total) == 0).then_some(free as u64)
                }
                _ => None,
            };

            devices.push(HipDeviceProbe {
                hip_index: dev,
                name,
                gcn_arch_name,
                pci_bus_id,
                total_vram_bytes,
                free_vram_bytes,
                shared_mem_per_block_bytes: has_r0600.then_some(prop.shared_mem_per_block as u64),
                multi_processor_count: (has_r0600 && prop.multi_processor_count > 0)
                    .then_some(prop.multi_processor_count as u32),
                warp_size: (has_r0600 && prop.warp_size > 0).then_some(prop.warp_size as u32),
                clock_mhz: (has_r0600 && prop.clock_rate_khz > 0)
                    .then_some((prop.clock_rate_khz / 1000) as u32),
                memory_clock_mhz: (has_r0600 && prop.memory_clock_rate_khz > 0)
                    .then_some((prop.memory_clock_rate_khz / 1000) as u32),
                memory_bus_width_bits: (has_r0600 && prop.memory_bus_width > 0)
                    .then_some(prop.memory_bus_width as u32),
                l2_cache_bytes: (has_r0600 && prop.l2_cache_size > 0)
                    .then_some(prop.l2_cache_size as u64),
                integrated: has_r0600.then_some(prop.integrated != 0),
                is_large_bar: has_r0600.then_some(prop.is_large_bar != 0),
                compute_capability: has_r0600.then_some((prop.major, prop.minor)),
            });
        }

        HipProbeReport {
            available: true,
            loaded_library: Some(lib_path.display().to_string()),
            runtime_version,
            driver_version,
            devices,
            error: None,
        }
    }
}
