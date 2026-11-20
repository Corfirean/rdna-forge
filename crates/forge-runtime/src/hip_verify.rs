use std::ffi::c_void;
use std::time::Instant;

use forge_core::{ForgeError, ForgeResult};
use forge_device::hip::{load_hip_library_checked, probe_hip};
use serde::{Deserialize, Serialize};

/// Result of a live HIP memory allocation, round-trip transfer, and event timing verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HipVerificationResult {
    pub device_index: i32,
    pub device_name: String,
    pub gfx_target: Option<String>,
    pub transfer_bytes: usize,
    pub h2d_d2h_bit_exact: bool,
    pub elapsed_ms: f32,
    pub bandwidth_gb_s: f64,
}

struct HipBufferGuard<'a> {
    ptr: *mut c_void,
    free_fn: libloading::Symbol<'a, extern "C" fn(*mut c_void) -> i32>,
}

impl Drop for HipBufferGuard<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = (self.free_fn)(self.ptr);
        }
    }
}

struct HipEventGuard<'a> {
    event: *mut c_void,
    destroy_fn: libloading::Symbol<'a, extern "C" fn(*mut c_void) -> i32>,
}

impl Drop for HipEventGuard<'_> {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = (self.destroy_fn)(self.event);
        }
    }
}

/// Verifies HIP runtime memory allocation, H2D/D2H round-trip integrity, and event timing.
pub fn verify_hip_transfer(
    device_index: i32,
    transfer_bytes: usize,
) -> ForgeResult<HipVerificationResult> {
    if transfer_bytes == 0 || !transfer_bytes.is_multiple_of(4) {
        return Err(ForgeError::InvalidDimension(
            "HIP transfer_bytes must be a non-zero multiple of 4".to_string(),
        ));
    }

    let hip_report = probe_hip();
    if !hip_report.available {
        return Err(ForgeError::RuntimeUnavailable {
            runtime: "HIP".to_string(),
            reason: hip_report
                .error
                .unwrap_or_else(|| "HIP runtime not available".to_string()),
        });
    }

    let dev_info = hip_report
        .devices
        .iter()
        .find(|d| d.hip_index == device_index)
        .ok_or_else(|| {
            ForgeError::DeviceNotFound(format!("HIP device index {device_index} not found"))
        })?
        .clone();

    let lib = load_hip_library_checked()
        .map_err(|e| e.into_forge_error("HIP"))?
        .library;

    unsafe {
        type SetDeviceFn = extern "C" fn(i32) -> i32;
        type MallocFn = extern "C" fn(*mut *mut c_void, usize) -> i32;
        type FreeFn = extern "C" fn(*mut c_void) -> i32;
        type MemcpyFn = extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
        type SyncFn = extern "C" fn() -> i32;
        type EventCreateFn = extern "C" fn(*mut *mut c_void) -> i32;
        type EventDestroyFn = extern "C" fn(*mut c_void) -> i32;
        type EventRecordFn = extern "C" fn(*mut c_void, *mut c_void) -> i32;
        type EventSyncFn = extern "C" fn(*mut c_void) -> i32;
        type EventElapsedFn = extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32;

        let set_device = lib
            .get::<SetDeviceFn>(b"hipSetDevice\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let malloc = lib
            .get::<MallocFn>(b"hipMalloc\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let free = lib
            .get::<FreeFn>(b"hipFree\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let memcpy = lib
            .get::<MemcpyFn>(b"hipMemcpy\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let sync = lib
            .get::<SyncFn>(b"hipDeviceSynchronize\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let event_create = lib
            .get::<EventCreateFn>(b"hipEventCreate\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let event_destroy = lib
            .get::<EventDestroyFn>(b"hipEventDestroy\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let event_record = lib
            .get::<EventRecordFn>(b"hipEventRecord\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let event_sync = lib
            .get::<EventSyncFn>(b"hipEventSynchronize\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;
        let event_elapsed = lib
            .get::<EventElapsedFn>(b"hipEventElapsedTime\0")
            .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?;

        let rc = set_device(device_index);
        if rc != 0 {
            return Err(ForgeError::ExecutionFailed(format!(
                "hipSetDevice({device_index}) failed with code {rc}"
            )));
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let rc = malloc(&mut dev_ptr, transfer_bytes);
        if rc != 0 || dev_ptr.is_null() {
            return Err(ForgeError::ExecutionFailed(format!(
                "hipMalloc({transfer_bytes}) failed with code {rc}"
            )));
        }
        let _buf_guard = HipBufferGuard {
            ptr: dev_ptr,
            free_fn: free,
        };

        let mut start_ev: *mut c_void = std::ptr::null_mut();
        let mut stop_ev: *mut c_void = std::ptr::null_mut();
        if event_create(&mut start_ev) != 0 || event_create(&mut stop_ev) != 0 {
            return Err(ForgeError::ExecutionFailed(
                "hipEventCreate failed".to_string(),
            ));
        }
        let _start_guard = HipEventGuard {
            event: start_ev,
            destroy_fn: lib
                .get::<EventDestroyFn>(b"hipEventDestroy\0")
                .map_err(|e| ForgeError::ExecutionFailed(e.to_string()))?,
        };
        let _stop_guard = HipEventGuard {
            event: stop_ev,
            destroy_fn: event_destroy,
        };

        let word_count = transfer_bytes / 4;
        let host_src: Vec<u32> = (0..word_count)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B9) ^ 0xA5A5_5A5A)
            .collect();
        let mut host_dst = vec![0u32; word_count];

        const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
        const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

        let wall_start = Instant::now();
        if event_record(start_ev, std::ptr::null_mut()) != 0 {
            return Err(ForgeError::ExecutionFailed(
                "hipEventRecord(start) failed".to_string(),
            ));
        }

        let rc_h2d = memcpy(
            dev_ptr,
            host_src.as_ptr().cast::<c_void>(),
            transfer_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
        );
        if rc_h2d != 0 {
            return Err(ForgeError::ExecutionFailed(format!(
                "hipMemcpy H2D failed with code {rc_h2d}"
            )));
        }

        let rc_d2h = memcpy(
            host_dst.as_mut_ptr().cast::<c_void>(),
            dev_ptr,
            transfer_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
        );
        if rc_d2h != 0 {
            return Err(ForgeError::ExecutionFailed(format!(
                "hipMemcpy D2H failed with code {rc_d2h}"
            )));
        }

        if event_record(stop_ev, std::ptr::null_mut()) != 0
            || event_sync(stop_ev) != 0
            || sync() != 0
        {
            return Err(ForgeError::ExecutionFailed(
                "HIP synchronization failed".to_string(),
            ));
        }

        let mut elapsed_ms: f32 = 0.0;
        let rc_el = event_elapsed(&mut elapsed_ms, start_ev, stop_ev);
        if rc_el != 0 {
            elapsed_ms = wall_start.elapsed().as_secs_f64() as f32 * 1000.0;
        }

        let exact = host_src == host_dst;
        let secs = f64::from(elapsed_ms.max(1e-4)) / 1000.0;
        let total_moved = (transfer_bytes as f64) * 2.0;
        let bandwidth_gb_s = total_moved / secs / 1e9;

        Ok(HipVerificationResult {
            device_index,
            device_name: dev_info.name,
            gfx_target: dev_info.gcn_arch_name,
            transfer_bytes,
            h2d_d2h_bit_exact: exact,
            elapsed_ms,
            bandwidth_gb_s,
        })
    }
}
