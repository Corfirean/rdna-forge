#[cfg(windows)]
mod imp {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr;
    use std::sync::{Mutex, MutexGuard};

    use forge_core::{ForgeError, ForgeResult};
    use tracing::debug;

    use crate::types::{parse_pci_device_id, GpuTelemetrySample};

    pub(crate) const ADLX_OK: i32 = 0;
    const ADLX_GPU_TYPE_DISCRETE: i32 = 2;

    /// Process-wide lock serializing `ADLXInitialize` / `ADLXTerminate` because ADLX uses global DLL state.
    static ADLX_LIFECYCLE_LOCK: Mutex<()> = Mutex::new(());

    #[repr(C)]
    pub(crate) struct IAdlxInterface {
        pub(crate) vtbl: *const IAdlxInterfaceVtbl,
    }

    #[repr(C)]
    pub(crate) struct IAdlxInterfaceVtbl {
        pub(crate) acquire: unsafe extern "system" fn(*mut IAdlxInterface) -> isize, // [0]
        pub(crate) release: unsafe extern "system" fn(*mut IAdlxInterface) -> isize, // [1]
        pub(crate) query_interface:
            unsafe extern "system" fn(*mut IAdlxInterface, *const u16, *mut *mut c_void) -> i32, // [2]
    }

    #[repr(C)]
    struct IAdlxSystem {
        vtbl: *const IAdlxSystemVtbl,
    }

    // Verified against ADLX V1.4 SDK/Include/ISystem.h. IADLXSystem does NOT inherit IADLXInterface.
    #[repr(C)]
    struct IAdlxSystemVtbl {
        get_hybrid_graphics_type: *const c_void, // [0]
        get_gpus: unsafe extern "system" fn(*mut IAdlxSystem, *mut *mut IAdlxGpuList) -> i32, // [1]
        query_interface: *const c_void,          // [2]
        get_displays_services: *const c_void,    // [3]
        get_desktops_services: *const c_void,    // [4]
        get_gpus_changed_handling: *const c_void, // [5]
        enable_log: *const c_void,               // [6]
        get_3d_settings_services: *const c_void, // [7]
        get_gpu_tuning_services:
            unsafe extern "system" fn(*mut IAdlxSystem, *mut *mut IAdlxInterface) -> i32, // [8]
        get_performance_monitoring_services:
            unsafe extern "system" fn(*mut IAdlxSystem, *mut *mut IAdlxPerfMon) -> i32, // [9]
        total_system_ram: *const c_void,         // [10]
        get_i2c: *const c_void,                  // [11]
    }

    #[repr(C)]
    pub(crate) struct IAdlxGpuList {
        vtbl: *const IAdlxGpuListVtbl,
    }

    #[repr(C)]
    struct IAdlxGpuListVtbl {
        acquire: unsafe extern "system" fn(*mut IAdlxGpuList) -> isize, // [0]
        release: unsafe extern "system" fn(*mut IAdlxGpuList) -> isize, // [1]
        query_interface: *const c_void,                                 // [2]
        size: unsafe extern "system" fn(*mut IAdlxGpuList) -> u32,      // [3]
        empty: *const c_void,                                           // [4]
        begin: *const c_void,                                           // [5]
        end: *const c_void,                                             // [6]
        at: *const c_void,                                              // [7]
        clear: *const c_void,                                           // [8]
        remove_back: *const c_void,                                     // [9]
        add_back: *const c_void,                                        // [10]
        at_gpu_list: unsafe extern "system" fn(*mut IAdlxGpuList, u32, *mut *mut IAdlxGpu) -> i32, // [11]
        add_back_gpu_list: *const c_void, // [12]
    }

    #[repr(C)]
    pub(crate) struct IAdlxGpu {
        vtbl: *const IAdlxGpuVtbl,
    }

    #[repr(C)]
    struct IAdlxGpuVtbl {
        acquire: unsafe extern "system" fn(*mut IAdlxGpu) -> isize, // [0]
        release: unsafe extern "system" fn(*mut IAdlxGpu) -> isize, // [1]
        query_interface: *const c_void,                             // [2]
        vendor_id: *const c_void,                                   // [3]
        asic_family_type: *const c_void,                            // [4]
        gpu_type: unsafe extern "system" fn(*mut IAdlxGpu, *mut i32) -> i32, // [5]
        is_external: *const c_void,                                 // [6]
        name: unsafe extern "system" fn(*mut IAdlxGpu, *mut *const c_char) -> i32, // [7]
        driver_path: *const c_void,                                 // [8]
        pnp_string: *const c_void,                                  // [9]
        has_desktops: *const c_void,                                // [10]
        total_vram: unsafe extern "system" fn(*mut IAdlxGpu, *mut u32) -> i32, // [11]
        vram_type: *const c_void,                                   // [12]
        bios_info: *const c_void,                                   // [13]
        device_id: unsafe extern "system" fn(*mut IAdlxGpu, *mut *const c_char) -> i32, // [14]
        revision_id: *const c_void,                                 // [15]
        sub_system_id: *const c_void,                               // [16]
        sub_system_vendor_id: *const c_void,                        // [17]
        unique_id: unsafe extern "system" fn(*mut IAdlxGpu, *mut i32) -> i32, // [18]
    }

    #[repr(C)]
    struct IAdlxPerfMon {
        vtbl: *const IAdlxPerfMonVtbl,
    }

    #[repr(C)]
    struct IAdlxPerfMonVtbl {
        acquire: unsafe extern "system" fn(*mut IAdlxPerfMon) -> isize, // [0]
        release: unsafe extern "system" fn(*mut IAdlxPerfMon) -> isize, // [1]
        query_interface: *const c_void,                                 // [2]
        get_sampling_interval_range: *const c_void,                     // [3]
        set_sampling_interval: *const c_void,                           // [4]
        get_sampling_interval: *const c_void,                           // [5]
        get_max_perf_metrics_history_size_range: *const c_void,         // [6]
        set_max_perf_metrics_history_size: *const c_void,               // [7]
        get_max_perf_metrics_history_size: *const c_void,               // [8]
        clear_perf_metrics_history: *const c_void,                      // [9]
        get_current_perf_metrics_history_size: *const c_void,           // [10]
        start_perf_metrics_tracking: *const c_void,                     // [11]
        stop_perf_metrics_tracking: *const c_void,                      // [12]
        get_all_metrics_history: *const c_void,                         // [13]
        get_gpu_metrics_history: *const c_void,                         // [14]
        get_system_metrics_history: *const c_void,                      // [15]
        get_fps_history: *const c_void,                                 // [16]
        get_current_all_metrics: *const c_void,                         // [17]
        get_current_gpu_metrics: unsafe extern "system" fn(
            *mut IAdlxPerfMon,
            *mut IAdlxGpu,
            *mut *mut IAdlxGpuMetrics,
        ) -> i32, // [18]
    }

    #[repr(C)]
    struct IAdlxGpuMetrics {
        vtbl: *const IAdlxGpuMetricsVtbl,
    }

    #[repr(C)]
    struct IAdlxGpuMetricsVtbl {
        acquire: unsafe extern "system" fn(*mut IAdlxGpuMetrics) -> isize, // [0]
        release: unsafe extern "system" fn(*mut IAdlxGpuMetrics) -> isize, // [1]
        query_interface: *const c_void,                                    // [2]
        time_stamp: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i64) -> i32, // [3]
        gpu_usage: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut f64) -> i32, // [4]
        gpu_clock_speed: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i32) -> i32, // [5]
        gpu_vram_clock_speed: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i32) -> i32, // [6]
        gpu_temperature: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut f64) -> i32, // [7]
        gpu_hotspot_temperature: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut f64) -> i32, // [8]
        gpu_power: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut f64) -> i32, // [9]
        gpu_total_board_power: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut f64) -> i32, // [10]
        gpu_fan_speed: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i32) -> i32, // [11]
        gpu_vram: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i32) -> i32,      // [12]
        gpu_voltage: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i32) -> i32,   // [13]
    }

    type QueryFullVersionFn = unsafe extern "C" fn(*mut u64) -> i32;
    type AdlxInitializeFn = unsafe extern "C" fn(u64, *mut *mut IAdlxSystem) -> i32;
    type AdlxTerminateFn = unsafe extern "C" fn() -> i32;

    struct OwnedGpuList(*mut IAdlxGpuList);
    impl Drop for OwnedGpuList {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0) };
            }
        }
    }

    struct OwnedPerfMon(*mut IAdlxPerfMon);
    impl Drop for OwnedPerfMon {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0) };
            }
        }
    }

    pub(crate) struct OwnedGpu(pub(crate) *mut IAdlxGpu);
    impl Drop for OwnedGpu {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0) };
            }
        }
    }

    struct OwnedGpuMetrics(*mut IAdlxGpuMetrics);
    impl Drop for OwnedGpuMetrics {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0) };
            }
        }
    }

    /// RAII guard for any ADLX child interface inheriting `IADLXInterface`.
    pub(crate) struct OwnedInterface(*mut IAdlxInterface);

    impl OwnedInterface {
        pub(crate) fn from_raw(ptr: *mut IAdlxInterface) -> Option<Self> {
            (!ptr.is_null()).then_some(Self(ptr))
        }

        pub(crate) fn as_raw(&self) -> *mut IAdlxInterface {
            self.0
        }

        pub(crate) fn query_interface(&self, iid_utf16: &[u16]) -> Option<Self> {
            if self.0.is_null() {
                return None;
            }
            let mut out: *mut c_void = ptr::null_mut();
            let rc = unsafe {
                ((*(*self.0).vtbl).query_interface)(self.0, iid_utf16.as_ptr(), &mut out)
            };
            if rc == ADLX_OK && !out.is_null() {
                Some(Self(out.cast::<IAdlxInterface>()))
            } else {
                None
            }
        }

        fn release_now(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0) };
                self.0 = ptr::null_mut();
            }
        }
    }

    impl Drop for OwnedInterface {
        fn drop(&mut self) {
            self.release_now();
        }
    }

    /// Exclusive RAII ADLX session holding the global ADLX lifecycle lock until terminated.
    pub struct AdlxSession {
        _lock: MutexGuard<'static, ()>,
        // Drop order matters: release all child COM interfaces before `ADLXTerminate` and library unload.
        tuning_svc: Option<OwnedInterface>,
        perf_mon: OwnedPerfMon,
        gpu_list: OwnedGpuList,
        terminate_fn: AdlxTerminateFn,
        _lib: libloading::Library,
    }

    impl Drop for AdlxSession {
        fn drop(&mut self) {
            // Manually release child COM objects BEFORE ADLXTerminate() while keeping `_lib` loaded.
            if let Some(mut tuning) = self.tuning_svc.take() {
                tuning.release_now();
            }
            if !self.perf_mon.0.is_null() {
                unsafe { ((*(*self.perf_mon.0).vtbl).release)(self.perf_mon.0) };
                self.perf_mon.0 = ptr::null_mut();
            }
            if !self.gpu_list.0.is_null() {
                unsafe { ((*(*self.gpu_list.0).vtbl).release)(self.gpu_list.0) };
                self.gpu_list.0 = ptr::null_mut();
            }
            let rc = unsafe { (self.terminate_fn)() };
            debug!(rc, "ADLXTerminate completed");
        }
    }

    impl AdlxSession {
        /// Loads `amdadlx64.dll`, initializes ADLX, and acquires GPU list + performance + tuning services.
        pub fn open() -> ForgeResult<Self> {
            let lock = ADLX_LIFECYCLE_LOCK.lock().map_err(|_| {
                ForgeError::ExecutionFailed("ADLX lifecycle mutex poisoned".to_string())
            })?;

            let lib = unsafe { libloading::Library::new("amdadlx64.dll") }.map_err(|e| {
                ForgeError::RuntimeUnavailable {
                    runtime: "ADLX".to_string(),
                    reason: format!("amdadlx64.dll unavailable: {e}"),
                }
            })?;

            let (full_ver, init_fn, terminate_fn) = unsafe {
                let qfv = *lib
                    .get::<QueryFullVersionFn>(b"ADLXQueryFullVersion\0")
                    .map_err(|e| ForgeError::RuntimeUnavailable {
                        runtime: "ADLX".to_string(),
                        reason: format!("ADLXQueryFullVersion missing: {e}"),
                    })?;
                let init = *lib
                    .get::<AdlxInitializeFn>(b"ADLXInitialize\0")
                    .map_err(|e| ForgeError::RuntimeUnavailable {
                        runtime: "ADLX".to_string(),
                        reason: format!("ADLXInitialize missing: {e}"),
                    })?;
                let term = *lib
                    .get::<AdlxTerminateFn>(b"ADLXTerminate\0")
                    .map_err(|e| ForgeError::RuntimeUnavailable {
                        runtime: "ADLX".to_string(),
                        reason: format!("ADLXTerminate missing: {e}"),
                    })?;

                let mut ver: u64 = 0;
                let rc = qfv(&mut ver);
                if rc != ADLX_OK || ver == 0 {
                    return Err(ForgeError::RuntimeUnavailable {
                        runtime: "ADLX".to_string(),
                        reason: format!("ADLXQueryFullVersion returned error {rc}"),
                    });
                }
                (ver, init, term)
            };

            let mut system_ptr: *mut IAdlxSystem = ptr::null_mut();
            let init_rc = unsafe { init_fn(full_ver, &mut system_ptr) };
            if init_rc != ADLX_OK || system_ptr.is_null() {
                return Err(ForgeError::RuntimeUnavailable {
                    runtime: "ADLX".to_string(),
                    reason: format!("ADLXInitialize failed with code {init_rc}"),
                });
            }

            let mut gpu_list_ptr: *mut IAdlxGpuList = ptr::null_mut();
            let gpus_rc =
                unsafe { ((*(*system_ptr).vtbl).get_gpus)(system_ptr, &mut gpu_list_ptr) };
            if gpus_rc != ADLX_OK || gpu_list_ptr.is_null() {
                unsafe { terminate_fn() };
                return Err(ForgeError::ExecutionFailed(format!(
                    "IADLXSystem::GetGPUs failed with code {gpus_rc}"
                )));
            }
            let gpu_list = OwnedGpuList(gpu_list_ptr);

            let mut perf_mon_ptr: *mut IAdlxPerfMon = ptr::null_mut();
            let pm_rc = unsafe {
                ((*(*system_ptr).vtbl).get_performance_monitoring_services)(
                    system_ptr,
                    &mut perf_mon_ptr,
                )
            };
            if pm_rc != ADLX_OK || perf_mon_ptr.is_null() {
                drop(gpu_list);
                unsafe { terminate_fn() };
                return Err(ForgeError::ExecutionFailed(format!(
                    "IADLXSystem::GetPerformanceMonitoringServices failed with code {pm_rc}"
                )));
            }

            let mut tuning_ptr: *mut IAdlxInterface = ptr::null_mut();
            let tuning_rc = unsafe {
                ((*(*system_ptr).vtbl).get_gpu_tuning_services)(system_ptr, &mut tuning_ptr)
            };
            let tuning_svc = if tuning_rc == ADLX_OK {
                OwnedInterface::from_raw(tuning_ptr)
            } else {
                debug!(
                    tuning_rc,
                    "IADLXSystem::GetGPUTuningServices returned non-OK status"
                );
                None
            };

            Ok(Self {
                _lock: lock,
                tuning_svc,
                perf_mon: OwnedPerfMon(perf_mon_ptr),
                gpu_list,
                terminate_fn,
                _lib: lib,
            })
        }

        pub(crate) fn tuning_services_ptr(&self) -> Option<*mut IAdlxInterface> {
            self.tuning_svc.as_ref().map(OwnedInterface::as_raw)
        }

        pub(crate) fn gpu_count(&self) -> u32 {
            unsafe { ((*(*self.gpu_list.0).vtbl).size)(self.gpu_list.0) }
        }

        pub(crate) fn gpu_at(&self, idx: u32) -> Option<OwnedGpu> {
            let mut gpu_ptr: *mut IAdlxGpu = ptr::null_mut();
            let rc = unsafe {
                ((*(*self.gpu_list.0).vtbl).at_gpu_list)(self.gpu_list.0, idx, &mut gpu_ptr)
            };
            if rc == ADLX_OK && !gpu_ptr.is_null() {
                Some(OwnedGpu(gpu_ptr))
            } else {
                None
            }
        }

        pub(crate) fn read_gpu_identity(
            gpu: &OwnedGpu,
        ) -> (String, Option<u32>, bool, Option<u32>) {
            let gpu_name = unsafe {
                let mut p: *const c_char = ptr::null();
                if ((*(*gpu.0).vtbl).name)(gpu.0, &mut p) == ADLX_OK && !p.is_null() {
                    CStr::from_ptr(p).to_string_lossy().trim().to_string()
                } else {
                    "Unknown AMD GPU".to_string()
                }
            };

            let pci_device_id = unsafe {
                let mut p: *const c_char = ptr::null();
                if ((*(*gpu.0).vtbl).device_id)(gpu.0, &mut p) == ADLX_OK && !p.is_null() {
                    parse_pci_device_id(&CStr::from_ptr(p).to_string_lossy())
                } else {
                    None
                }
            };

            let discrete = unsafe {
                let mut t: i32 = 0;
                ((*(*gpu.0).vtbl).gpu_type)(gpu.0, &mut t) == ADLX_OK && t == ADLX_GPU_TYPE_DISCRETE
            };

            let total_vram_mib = unsafe {
                let mut mb: u32 = 0;
                (((*(*gpu.0).vtbl).total_vram)(gpu.0, &mut mb) == ADLX_OK && mb > 0).then_some(mb)
            };

            (gpu_name, pci_device_id, discrete, total_vram_mib)
        }

        /// Reads a telemetry sample for every GPU exposed by ADLX.
        pub fn read_all_samples(&self) -> ForgeResult<Vec<GpuTelemetrySample>> {
            let count = self.gpu_count();
            let mut out = Vec::with_capacity(count as usize);
            for idx in 0..count {
                let Some(gpu) = self.gpu_at(idx) else {
                    continue;
                };
                if let Some(sample) = self.read_gpu_sample(&gpu) {
                    out.push(sample);
                }
            }
            Ok(out)
        }

        /// Reads a single telemetry sample for the GPU matching `pci_device_id` (or discrete/index 0 fallback).
        pub fn read_matching_sample(
            &self,
            pci_device_id: Option<u32>,
        ) -> ForgeResult<Option<GpuTelemetrySample>> {
            let samples = self.read_all_samples()?;
            if let Some(target_id) = pci_device_id {
                if let Some(matched) = samples
                    .iter()
                    .find(|s| s.pci_device_id == Some(target_id))
                    .cloned()
                {
                    return Ok(Some(matched));
                }
            }
            Ok(samples
                .iter()
                .find(|s| s.discrete)
                .cloned()
                .or_else(|| samples.into_iter().next()))
        }

        fn read_gpu_sample(&self, gpu: &OwnedGpu) -> Option<GpuTelemetrySample> {
            let mut metrics_ptr: *mut IAdlxGpuMetrics = ptr::null_mut();
            let m_rc = unsafe {
                ((*(*self.perf_mon.0).vtbl).get_current_gpu_metrics)(
                    self.perf_mon.0,
                    gpu.0,
                    &mut metrics_ptr,
                )
            };
            if m_rc != ADLX_OK || metrics_ptr.is_null() {
                return None;
            }
            let metrics = OwnedGpuMetrics(metrics_ptr);
            let (gpu_name, pci_device_id, discrete, total_vram_mib) = Self::read_gpu_identity(gpu);

            unsafe {
                let vt = &*(*metrics.0).vtbl;
                Some(GpuTelemetrySample {
                    gpu_name,
                    pci_device_id,
                    discrete,
                    total_vram_mib,
                    timestamp_ms: read_i64(metrics.0, vt.time_stamp),
                    gpu_usage_pct: read_f64(metrics.0, vt.gpu_usage),
                    gpu_clock_mhz: read_i32(metrics.0, vt.gpu_clock_speed),
                    vram_clock_mhz: read_i32(metrics.0, vt.gpu_vram_clock_speed),
                    gpu_temp_c: read_f64(metrics.0, vt.gpu_temperature),
                    hotspot_temp_c: read_f64(metrics.0, vt.gpu_hotspot_temperature),
                    gpu_power_w: read_f64(metrics.0, vt.gpu_power),
                    board_power_w: read_f64(metrics.0, vt.gpu_total_board_power),
                    fan_rpm: read_i32(metrics.0, vt.gpu_fan_speed),
                    vram_used_mib: read_i32(metrics.0, vt.gpu_vram),
                    voltage_mv: read_i32(metrics.0, vt.gpu_voltage),
                })
            }
        }
    }

    unsafe fn read_i64(
        m: *mut IAdlxGpuMetrics,
        f: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i64) -> i32,
    ) -> Option<i64> {
        let mut v = 0i64;
        (f(m, &mut v) == ADLX_OK).then_some(v)
    }

    unsafe fn read_i32(
        m: *mut IAdlxGpuMetrics,
        f: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut i32) -> i32,
    ) -> Option<i32> {
        let mut v = 0i32;
        (f(m, &mut v) == ADLX_OK).then_some(v)
    }

    unsafe fn read_f64(
        m: *mut IAdlxGpuMetrics,
        f: unsafe extern "system" fn(*mut IAdlxGpuMetrics, *mut f64) -> i32,
    ) -> Option<f64> {
        let mut v = 0.0f64;
        (f(m, &mut v) == ADLX_OK && v.is_finite()).then_some(v)
    }
}

#[cfg(windows)]
pub use imp::AdlxSession;

#[cfg(windows)]
pub(crate) use imp::{IAdlxGpu, IAdlxInterface, OwnedGpu, OwnedInterface, ADLX_OK};

#[cfg(not(windows))]
mod stub {
    use forge_core::{ForgeError, ForgeResult};

    use crate::tuning::{GpuTuningReport, PowerLimitChangeResult, TuningResetResult};
    use crate::types::GpuTelemetrySample;

    pub struct AdlxSession;

    impl AdlxSession {
        pub fn open() -> ForgeResult<Self> {
            Err(ForgeError::RuntimeUnavailable {
                runtime: "ADLX".to_string(),
                reason: "ADLX telemetry is Windows-only".to_string(),
            })
        }

        pub fn read_all_samples(&self) -> ForgeResult<Vec<GpuTelemetrySample>> {
            Ok(Vec::new())
        }

        pub fn read_matching_sample(
            &self,
            _pci_device_id: Option<u32>,
        ) -> ForgeResult<Option<GpuTelemetrySample>> {
            Ok(None)
        }

        pub fn read_all_tuning(&self) -> ForgeResult<Vec<GpuTuningReport>> {
            Ok(Vec::new())
        }

        pub fn set_power_limit(
            &self,
            _pci_device_id: Option<u32>,
            _limit_pct: i32,
            _dry_run: bool,
        ) -> ForgeResult<PowerLimitChangeResult> {
            Err(ForgeError::RuntimeUnavailable {
                runtime: "ADLX".to_string(),
                reason: "ADLX tuning is Windows-only".to_string(),
            })
        }

        pub fn reset_tuning_to_factory(
            &self,
            _pci_device_id: Option<u32>,
            _dry_run: bool,
        ) -> ForgeResult<TuningResetResult> {
            Err(ForgeError::RuntimeUnavailable {
                runtime: "ADLX".to_string(),
                reason: "ADLX tuning is Windows-only".to_string(),
            })
        }
    }
}

#[cfg(not(windows))]
pub use stub::AdlxSession;
