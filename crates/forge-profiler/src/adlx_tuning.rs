#![cfg(windows)]

use std::ffi::c_void;
use std::ptr;

use forge_core::{ForgeError, ForgeResult};

use crate::adlx_ffi::{AdlxSession, IAdlxGpu, IAdlxInterface, OwnedGpu, OwnedInterface, ADLX_OK};
use crate::tuning::{
    adlx_succeeded, change_power_limit, reset_to_factory, ApiResult, FactoryResetControl,
    FanCurvePoint, FanTuningInfo, GfxTuningInfo, GpuTuningReport, IntRange, MemoryTimingMode,
    OperationStatus, PowerLimitChangeResult, PowerLimitControl, PowerTuningInfo, PresetTuningInfo,
    TuningCapabilities, TuningResetResult, VramTuningInfo,
};

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct AdlxIntRange {
    min_value: i32,
    max_value: i32,
    step: i32,
}

impl AdlxIntRange {
    fn to_domain(self) -> Option<IntRange> {
        IntRange::from_adlx(self.min_value, self.max_value, self.step)
    }
}

#[repr(C)]
struct IAdlxGpuTuningServicesVtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    _get_gpu_tuning_changed_handling: *const c_void,
    is_at_factory: unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [4]
    reset_to_factory: unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu) -> i32,       // [5]
    is_supported_auto_tuning: unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [6]
    is_supported_preset_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [7]
    is_supported_manual_gfx_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [8]
    is_supported_manual_vram_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [9]
    is_supported_manual_fan_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [10]
    is_supported_manual_power_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32, // [11]
    _get_auto_tuning: *const c_void, // [12]
    get_preset_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut *mut IAdlxInterface) -> i32, // [13]
    get_manual_gfx_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut *mut IAdlxInterface) -> i32, // [14]
    get_manual_vram_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut *mut IAdlxInterface) -> i32, // [15]
    get_manual_fan_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut *mut IAdlxInterface) -> i32, // [16]
    get_manual_power_tuning:
        unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut *mut IAdlxInterface) -> i32, // [17]
}

#[repr(C)]
struct IAdlxGpuPresetTuningVtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    is_supported_power_saver: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [3]
    is_supported_quiet: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,       // [4]
    is_supported_balanced: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,    // [5]
    is_supported_turbo: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,       // [6]
    is_supported_rage: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,        // [7]
    is_current_power_saver: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,   // [8]
    is_current_quiet: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,         // [9]
    is_current_balanced: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,      // [10]
    is_current_turbo: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,         // [11]
    is_current_rage: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,          // [12]
}

#[repr(C)]
struct IAdlxManualPowerTuning1Vtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    get_power_limit_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [3]
    get_power_limit: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,                // [4]
    set_power_limit: unsafe extern "system" fn(*mut c_void, i32) -> i32,                     // [5]
    is_supported_tdc_limit: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,          // [6]
    get_tdc_limit_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32,   // [7]
    get_tdc_limit: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,                  // [8]
    _set_tdc_limit: *const c_void,                                                           // [9]
    get_power_limit_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,        // [10]
    get_tdc_limit_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,          // [11]
}

#[repr(C)]
struct IAdlxManualGraphicsTuning2_1Vtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    get_gpu_min_frequency_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [3]
    get_gpu_min_frequency: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [4]
    _set_gpu_min_frequency: *const c_void,                                          // [5]
    get_gpu_max_frequency_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [6]
    get_gpu_max_frequency: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [7]
    _set_gpu_max_frequency: *const c_void,                                          // [8]
    get_gpu_voltage_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [9]
    get_gpu_voltage: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,       // [10]
    _set_gpu_voltage: *const c_void,                                                // [11]
    get_gpu_min_frequency_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [12]
    get_gpu_max_frequency_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [13]
    get_gpu_voltage_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [14]
}

#[repr(C)]
struct IAdlxManualVramTuning2_1Vtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    is_supported_memory_timing: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [3]
    get_supported_memory_timing_description_list:
        unsafe extern "system" fn(*mut c_void, *mut *mut IAdlxInterface) -> i32, // [4]
    get_memory_timing_description: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [5]
    _set_memory_timing_description: *const c_void,                                      // [6]
    get_max_vram_frequency_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [7]
    get_max_vram_frequency: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [8]
    _set_max_vram_frequency: *const c_void,                                          // [9]
    get_max_vram_frequency_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [10]
}

#[repr(C)]
struct IAdlxMemoryTimingDescriptionListVtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    size: unsafe extern "system" fn(*mut c_void) -> u32, // [3]
    _empty: *const c_void,
    _begin: *const c_void,
    _end: *const c_void,
    _at: *const c_void,
    _clear: *const c_void,
    _remove_back: *const c_void,
    _add_back: *const c_void,
    at_list: unsafe extern "system" fn(*mut c_void, u32, *mut *mut IAdlxInterface) -> i32, // [11]
}

#[repr(C)]
struct IAdlxMemoryTimingDescriptionVtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    get_description: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [3]
}

#[repr(C)]
struct IAdlxManualFanTuning1Vtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    get_fan_tuning_ranges:
        unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange, *mut AdlxIntRange) -> i32, // [3]
    get_fan_tuning_states: unsafe extern "system" fn(*mut c_void, *mut *mut IAdlxInterface) -> i32, // [4]
    _get_empty_fan_tuning_states: *const c_void, // [5]
    _is_valid_fan_tuning_states: *const c_void,  // [6]
    _set_fan_tuning_states: *const c_void,       // [7]
    is_supported_zero_rpm: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [8]
    get_zero_rpm_state: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [9]
    _set_zero_rpm_state: *const c_void,          // [10]
    is_supported_min_acoustic_limit: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [11]
    get_min_acoustic_limit_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [12]
    get_min_acoustic_limit: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [13]
    _set_min_acoustic_limit: *const c_void,                                          // [14]
    is_supported_min_fan_speed: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [15]
    get_min_fan_speed_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [16]
    get_min_fan_speed: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [17]
    _set_min_fan_speed: *const c_void,                                          // [18]
    is_supported_target_fan_speed: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [19]
    get_target_fan_speed_range: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32, // [20]
    get_target_fan_speed: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [21]
    _set_target_fan_speed: *const c_void,                                          // [22]
    get_default_fan_tuning_states:
        unsafe extern "system" fn(*mut c_void, *mut *mut IAdlxInterface) -> i32, // [23]
    get_min_acoustic_limit_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [24]
    get_min_fan_speed_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [25]
    get_target_fan_speed_default: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [26]
    get_default_zero_rpm_state: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32, // [27]
}

#[repr(C)]
struct IAdlxManualFanTuningStateListVtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    size: unsafe extern "system" fn(*mut c_void) -> u32, // [3]
    _empty: *const c_void,
    _begin: *const c_void,
    _end: *const c_void,
    _at: *const c_void,
    _clear: *const c_void,
    _remove_back: *const c_void,
    _add_back: *const c_void,
    at_list: unsafe extern "system" fn(*mut c_void, u32, *mut *mut IAdlxInterface) -> i32, // [11]
}

#[repr(C)]
struct IAdlxManualFanTuningStateVtbl {
    _acquire: *const c_void,
    _release: *const c_void,
    _query_interface: *const c_void,
    get_fan_speed: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [3]
    _set_fan_speed: *const c_void,
    get_temperature: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32, // [5]
}

fn utf16_iid(ascii: &str) -> Vec<u16> {
    ascii.encode_utf16().chain(std::iter::once(0)).collect()
}

unsafe fn vtbl<V>(obj: &OwnedInterface) -> &V {
    let raw = obj.as_raw().cast::<*const V>();
    &**raw
}

fn query_with_fallback(
    base: &OwnedInterface,
    preferred_iid: &str,
    fallback_iid: &str,
) -> Option<(OwnedInterface, bool)> {
    if let Some(ext) = base.query_interface(&utf16_iid(preferred_iid)) {
        Some((ext, true))
    } else {
        base.query_interface(&utf16_iid(fallback_iid))
            .map(|obj| (obj, false))
    }
}

unsafe fn call_gpu_bool(
    svc: &IAdlxGpuTuningServicesVtbl,
    svc_ptr: *mut c_void,
    func: unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut u8) -> i32,
    gpu: *mut IAdlxGpu,
) -> Option<bool> {
    let _ = svc;
    let mut v: u8 = 0;
    (func(svc_ptr, gpu, &mut v) == ADLX_OK).then_some(v != 0)
}

unsafe fn get_iface(
    func: unsafe extern "system" fn(*mut c_void, *mut IAdlxGpu, *mut *mut IAdlxInterface) -> i32,
    svc_ptr: *mut c_void,
    gpu: *mut IAdlxGpu,
) -> Option<OwnedInterface> {
    let mut out: *mut IAdlxInterface = ptr::null_mut();
    if func(svc_ptr, gpu, &mut out) == ADLX_OK {
        OwnedInterface::from_raw(out)
    } else {
        None
    }
}

unsafe fn read_bool(
    obj: &OwnedInterface,
    f: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,
) -> Option<bool> {
    let mut v: u8 = 0;
    (f(obj.as_raw().cast(), &mut v) == ADLX_OK).then_some(v != 0)
}

/// Real ADLX implementation of [`PowerLimitControl`] over `IADLXManualPowerTuning(1)`.
struct AdlxPowerControl<'a> {
    obj: &'a OwnedInterface,
    vt: &'a IAdlxManualPowerTuning1Vtbl,
}

impl PowerLimitControl for AdlxPowerControl<'_> {
    fn read(&mut self) -> Option<i32> {
        // SAFETY: `vt` was derived from `obj`, which keeps the COM object alive for `'a`.
        unsafe { read_i32(self.obj, self.vt.get_power_limit) }
    }

    fn write(&mut self, value_pct: i32) -> Result<(), i32> {
        // SAFETY: slot [5] of IADLXManualPowerTuning is `SetPowerLimit(adlx_int)`; verified against
        // the upstream header (see the ABI layout tests).
        let rc = unsafe { (self.vt.set_power_limit)(self.obj.as_raw().cast(), value_pct) };
        if adlx_succeeded(rc) {
            Ok(())
        } else {
            Err(rc)
        }
    }
}

/// Real ADLX implementation of [`FactoryResetControl`] over `IADLXGPUTuningServices`.
struct AdlxResetControl<'a> {
    svc_ptr: *mut c_void,
    vt: &'a IAdlxGpuTuningServicesVtbl,
    gpu: *mut IAdlxGpu,
}

impl FactoryResetControl for AdlxResetControl<'_> {
    fn is_at_factory(&mut self) -> Option<bool> {
        // SAFETY: `svc_ptr`/`gpu` are live COM pointers owned by the caller's session.
        unsafe { call_gpu_bool(self.vt, self.svc_ptr, self.vt.is_at_factory, self.gpu) }
    }

    fn reset(&mut self) -> i32 {
        // SAFETY: slot [5] of IADLXGPUTuningServices is `ResetToFactory(IADLXGPU*)`. The raw
        // ADLX_RESULT is returned unchanged; the state machine decides what it means.
        unsafe { (self.vt.reset_to_factory)(self.svc_ptr, self.gpu) }
    }
}

unsafe fn read_i32(
    obj: &OwnedInterface,
    f: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,
) -> Option<i32> {
    let mut v: i32 = 0;
    (f(obj.as_raw().cast(), &mut v) == ADLX_OK).then_some(v)
}

unsafe fn read_range(
    obj: &OwnedInterface,
    f: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32,
) -> Option<IntRange> {
    let mut r = AdlxIntRange::default();
    if f(obj.as_raw().cast(), &mut r) == ADLX_OK {
        r.to_domain()
    } else {
        None
    }
}

unsafe fn read_supported_int_param(
    obj: &OwnedInterface,
    is_extended: bool,
    is_supported_fn: unsafe extern "system" fn(*mut c_void, *mut u8) -> i32,
    range_fn: unsafe extern "system" fn(*mut c_void, *mut AdlxIntRange) -> i32,
    val_fn: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,
    default_fn: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,
) -> (bool, Option<i32>, Option<IntRange>, Option<i32>) {
    let supported = read_bool(obj, is_supported_fn).unwrap_or(false);
    if !supported {
        return (false, None, None, None);
    }
    let range = read_range(obj, range_fn);
    let val = read_i32(obj, val_fn);
    let default = if is_extended {
        read_i32(obj, default_fn)
    } else {
        None
    };
    (true, val, range, default)
}

fn read_capabilities(
    vt: &IAdlxGpuTuningServicesVtbl,
    svc_ptr: *mut c_void,
    gpu: *mut IAdlxGpu,
) -> (Option<bool>, TuningCapabilities) {
    unsafe {
        let at_factory = call_gpu_bool(vt, svc_ptr, vt.is_at_factory, gpu);
        let caps = TuningCapabilities {
            auto_tuning: call_gpu_bool(vt, svc_ptr, vt.is_supported_auto_tuning, gpu)
                .unwrap_or(false),
            preset_tuning: call_gpu_bool(vt, svc_ptr, vt.is_supported_preset_tuning, gpu)
                .unwrap_or(false),
            manual_gfx: call_gpu_bool(vt, svc_ptr, vt.is_supported_manual_gfx_tuning, gpu)
                .unwrap_or(false),
            manual_vram: call_gpu_bool(vt, svc_ptr, vt.is_supported_manual_vram_tuning, gpu)
                .unwrap_or(false),
            manual_fan: call_gpu_bool(vt, svc_ptr, vt.is_supported_manual_fan_tuning, gpu)
                .unwrap_or(false),
            manual_power: call_gpu_bool(vt, svc_ptr, vt.is_supported_manual_power_tuning, gpu)
                .unwrap_or(false),
        };
        (at_factory, caps)
    }
}

fn read_preset_info(
    vt: &IAdlxGpuTuningServicesVtbl,
    svc_ptr: *mut c_void,
    gpu: *mut IAdlxGpu,
) -> Option<PresetTuningInfo> {
    let raw = unsafe { get_iface(vt.get_preset_tuning, svc_ptr, gpu) }?;
    let preset_obj = raw.query_interface(&utf16_iid("IADLXGPUPresetTuning"))?;
    let pvt = unsafe { vtbl::<IAdlxGpuPresetTuningVtbl>(&preset_obj) };

    let candidates = [
        (
            "PowerSaver",
            pvt.is_supported_power_saver,
            pvt.is_current_power_saver,
        ),
        ("Quiet", pvt.is_supported_quiet, pvt.is_current_quiet),
        (
            "Balanced",
            pvt.is_supported_balanced,
            pvt.is_current_balanced,
        ),
        ("Turbo", pvt.is_supported_turbo, pvt.is_current_turbo),
        ("Rage", pvt.is_supported_rage, pvt.is_current_rage),
    ];

    let mut supported_presets = Vec::new();
    let mut current_preset = None;
    for (name, sup_fn, cur_fn) in candidates {
        if unsafe { read_bool(&preset_obj, sup_fn) }.unwrap_or(false) {
            supported_presets.push(name.to_string());
            if current_preset.is_none()
                && unsafe { read_bool(&preset_obj, cur_fn) }.unwrap_or(false)
            {
                current_preset = Some(name.to_string());
            }
        }
    }

    Some(PresetTuningInfo {
        supported_presets,
        current_preset,
    })
}

fn read_power_info_from_iface(obj: &OwnedInterface, is_v1: bool) -> PowerTuningInfo {
    let pvt = unsafe { vtbl::<IAdlxManualPowerTuning1Vtbl>(obj) };
    let power_limit_range = unsafe { read_range(obj, pvt.get_power_limit_range) };
    let power_limit_pct = unsafe { read_i32(obj, pvt.get_power_limit) };
    let power_limit_default_pct = if is_v1 {
        unsafe { read_i32(obj, pvt.get_power_limit_default) }
    } else {
        None
    };

    let (tdc_limit_supported, tdc_limit_a, tdc_limit_range, tdc_limit_default_a) = unsafe {
        read_supported_int_param(
            obj,
            is_v1,
            pvt.is_supported_tdc_limit,
            pvt.get_tdc_limit_range,
            pvt.get_tdc_limit,
            pvt.get_tdc_limit_default,
        )
    };

    PowerTuningInfo {
        power_limit_pct,
        power_limit_range,
        power_limit_default_pct,
        tdc_limit_supported,
        tdc_limit_a,
        tdc_limit_range,
        tdc_limit_default_a,
    }
}

fn read_gfx_info(
    vt: &IAdlxGpuTuningServicesVtbl,
    svc_ptr: *mut c_void,
    gpu: *mut IAdlxGpu,
) -> Option<GfxTuningInfo> {
    let raw = unsafe { get_iface(vt.get_manual_gfx_tuning, svc_ptr, gpu) }?;
    let (obj, is_v2_1) = query_with_fallback(
        &raw,
        "IADLXManualGraphicsTuning2_1",
        "IADLXManualGraphicsTuning2",
    )?;
    let gvt = unsafe { vtbl::<IAdlxManualGraphicsTuning2_1Vtbl>(&obj) };

    unsafe {
        Some(GfxTuningInfo {
            min_freq_mhz: read_i32(&obj, gvt.get_gpu_min_frequency),
            min_freq_range: read_range(&obj, gvt.get_gpu_min_frequency_range),
            min_freq_default_mhz: is_v2_1
                .then(|| read_i32(&obj, gvt.get_gpu_min_frequency_default))
                .flatten(),
            max_freq_mhz: read_i32(&obj, gvt.get_gpu_max_frequency),
            max_freq_range: read_range(&obj, gvt.get_gpu_max_frequency_range),
            max_freq_default_mhz: is_v2_1
                .then(|| read_i32(&obj, gvt.get_gpu_max_frequency_default))
                .flatten(),
            voltage_mv: read_i32(&obj, gvt.get_gpu_voltage),
            voltage_range: read_range(&obj, gvt.get_gpu_voltage_range),
            voltage_default_mv: is_v2_1
                .then(|| read_i32(&obj, gvt.get_gpu_voltage_default))
                .flatten(),
        })
    }
}

fn read_vram_info(
    vt: &IAdlxGpuTuningServicesVtbl,
    svc_ptr: *mut c_void,
    gpu: *mut IAdlxGpu,
) -> Option<VramTuningInfo> {
    let raw = unsafe { get_iface(vt.get_manual_vram_tuning, svc_ptr, gpu) }?;
    let (obj, is_v2_1) =
        query_with_fallback(&raw, "IADLXManualVRAMTuning2_1", "IADLXManualVRAMTuning2")?;
    let vvt = unsafe { vtbl::<IAdlxManualVramTuning2_1Vtbl>(&obj) };

    let memory_timing_supported =
        unsafe { read_bool(&obj, vvt.is_supported_memory_timing) }.unwrap_or(false);
    let memory_timing = if memory_timing_supported {
        unsafe { read_i32(&obj, vvt.get_memory_timing_description) }
            .map(MemoryTimingMode::from_adlx)
    } else {
        None
    };

    let mut supported_memory_timings = Vec::new();
    if memory_timing_supported {
        let mut list_ptr: *mut IAdlxInterface = ptr::null_mut();
        if unsafe {
            (vvt.get_supported_memory_timing_description_list)(obj.as_raw().cast(), &mut list_ptr)
        } == ADLX_OK
        {
            if let Some(list) = OwnedInterface::from_raw(list_ptr) {
                let lvt = unsafe { vtbl::<IAdlxMemoryTimingDescriptionListVtbl>(&list) };
                let count = unsafe { (lvt.size)(list.as_raw().cast()) };
                for idx in 0..count {
                    let mut item_ptr: *mut IAdlxInterface = ptr::null_mut();
                    if unsafe { (lvt.at_list)(list.as_raw().cast(), idx, &mut item_ptr) } == ADLX_OK
                    {
                        if let Some(item) = OwnedInterface::from_raw(item_ptr) {
                            let ivt = unsafe { vtbl::<IAdlxMemoryTimingDescriptionVtbl>(&item) };
                            if let Some(code) = unsafe { read_i32(&item, ivt.get_description) } {
                                supported_memory_timings.push(MemoryTimingMode::from_adlx(code));
                            }
                        }
                    }
                }
            }
        }
    }

    unsafe {
        Some(VramTuningInfo {
            max_freq_mhz: read_i32(&obj, vvt.get_max_vram_frequency),
            max_freq_range: read_range(&obj, vvt.get_max_vram_frequency_range),
            max_freq_default_mhz: is_v2_1
                .then(|| read_i32(&obj, vvt.get_max_vram_frequency_default))
                .flatten(),
            memory_timing_supported,
            memory_timing,
            supported_memory_timings,
        })
    }
}

fn read_fan_info(
    vt: &IAdlxGpuTuningServicesVtbl,
    svc_ptr: *mut c_void,
    gpu: *mut IAdlxGpu,
) -> Option<FanTuningInfo> {
    let raw = unsafe { get_iface(vt.get_manual_fan_tuning, svc_ptr, gpu) }?;
    let (obj, is_v1) = query_with_fallback(&raw, "IADLXManualFanTuning1", "IADLXManualFanTuning")?;
    let fvt = unsafe { vtbl::<IAdlxManualFanTuning1Vtbl>(&obj) };

    let mut raw_speed = AdlxIntRange::default();
    let mut raw_temp = AdlxIntRange::default();
    let (speed_range_pct, temp_range_c) = if unsafe {
        (fvt.get_fan_tuning_ranges)(obj.as_raw().cast(), &mut raw_speed, &mut raw_temp)
    } == ADLX_OK
    {
        (raw_speed.to_domain(), raw_temp.to_domain())
    } else {
        (None, None)
    };

    let zero_rpm_supported = unsafe { read_bool(&obj, fvt.is_supported_zero_rpm) }.unwrap_or(false);
    let zero_rpm_enabled = if zero_rpm_supported {
        unsafe { read_bool(&obj, fvt.get_zero_rpm_state) }
    } else {
        None
    };
    let zero_rpm_default = if zero_rpm_supported && is_v1 {
        unsafe { read_bool(&obj, fvt.get_default_zero_rpm_state) }
    } else {
        None
    };

    let (
        min_acoustic_limit_supported,
        min_acoustic_limit_mhz,
        min_acoustic_limit_range,
        min_acoustic_limit_default_mhz,
    ) = unsafe {
        read_supported_int_param(
            &obj,
            is_v1,
            fvt.is_supported_min_acoustic_limit,
            fvt.get_min_acoustic_limit_range,
            fvt.get_min_acoustic_limit,
            fvt.get_min_acoustic_limit_default,
        )
    };
    let _ = min_acoustic_limit_supported;

    let (_, min_fan_speed_pct, min_fan_speed_range, min_fan_speed_default_pct) = unsafe {
        read_supported_int_param(
            &obj,
            is_v1,
            fvt.is_supported_min_fan_speed,
            fvt.get_min_fan_speed_range,
            fvt.get_min_fan_speed,
            fvt.get_min_fan_speed_default,
        )
    };

    let (_, target_fan_speed_pct, target_fan_speed_range, target_fan_speed_default_pct) = unsafe {
        read_supported_int_param(
            &obj,
            is_v1,
            fvt.is_supported_target_fan_speed,
            fvt.get_target_fan_speed_range,
            fvt.get_target_fan_speed,
            fvt.get_target_fan_speed_default,
        )
    };

    let fan_curve = read_fan_state_list(&obj, fvt.get_fan_tuning_states);
    let default_fan_curve = if is_v1 {
        read_fan_state_list(&obj, fvt.get_default_fan_tuning_states)
    } else {
        Vec::new()
    };

    let flat_pwm_override_pct = if fan_curve.len() >= 2 {
        let first_spd = fan_curve[0].speed_pct;
        fan_curve
            .iter()
            .all(|p| p.speed_pct == first_spd)
            .then_some(first_spd)
    } else {
        None
    };

    let external_controller = detect_external_fan_controller();
    let fan_note = match (flat_pwm_override_pct, external_controller.as_deref()) {
        (Some(pct), Some(ctrl)) => {
            let min_hw = speed_range_pct.map(|r| r.min).unwrap_or(20);
            Some(format!(
                "External fan manager ({ctrl}) is actively overriding ADLX by setting all {} curve points to a flat {pct}% instant PWM target (when ADLX exposes no single-value PWM setter, it clamps values below {min_hw}% to the {min_hw}% hardware floor when Zero-RPM is off).",
                fan_curve.len()
            ))
        }
        (Some(pct), None) => Some(format!(
            "All {} ADLX curve points are set to a constant {pct}% (fixed/flat PWM override mode).",
            fan_curve.len()
        )),
        (None, Some(ctrl)) => Some(format!(
            "External fan manager ({ctrl}) detected running in background."
        )),
        (None, None) => None,
    };

    Some(FanTuningInfo {
        speed_range_pct,
        temp_range_c,
        zero_rpm_supported,
        zero_rpm_enabled,
        zero_rpm_default,
        min_acoustic_limit_mhz,
        min_acoustic_limit_range,
        min_acoustic_limit_default_mhz,
        min_fan_speed_pct,
        min_fan_speed_range,
        min_fan_speed_default_pct,
        target_fan_speed_pct,
        target_fan_speed_range,
        target_fan_speed_default_pct,
        fan_curve,
        default_fan_curve,
        flat_pwm_override_pct,
        external_controller,
        fan_note,
    })
}

fn read_fan_state_list(
    obj: &OwnedInterface,
    getter: unsafe extern "system" fn(*mut c_void, *mut *mut IAdlxInterface) -> i32,
) -> Vec<FanCurvePoint> {
    let mut curve = Vec::new();
    let mut state_list_ptr: *mut IAdlxInterface = ptr::null_mut();
    if unsafe { getter(obj.as_raw().cast(), &mut state_list_ptr) } == ADLX_OK {
        if let Some(list) = OwnedInterface::from_raw(state_list_ptr) {
            let lvt = unsafe { vtbl::<IAdlxManualFanTuningStateListVtbl>(&list) };
            let count = unsafe { (lvt.size)(list.as_raw().cast()) };
            for idx in 0..count {
                let mut state_ptr: *mut IAdlxInterface = ptr::null_mut();
                if unsafe { (lvt.at_list)(list.as_raw().cast(), idx, &mut state_ptr) } == ADLX_OK {
                    if let Some(state) = OwnedInterface::from_raw(state_ptr) {
                        let svt = unsafe { vtbl::<IAdlxManualFanTuningStateVtbl>(&state) };
                        if let (Some(temp_c), Some(speed_pct)) = unsafe {
                            (
                                read_i32(&state, svt.get_temperature),
                                read_i32(&state, svt.get_fan_speed),
                            )
                        } {
                            curve.push(FanCurvePoint { temp_c, speed_pct });
                        }
                    }
                }
            }
        }
    }
    curve
}

fn detect_external_fan_controller() -> Option<String> {
    const TH32CS_SNAPPROCESS: u32 = 0x0000_0002;
    const INVALID_HANDLE_VALUE: isize = -1;

    #[repr(C)]
    struct ProcessEntry32W {
        dw_size: u32,
        cnt_usage: u32,
        th32_process_id: u32,
        th32_default_heap_id: usize,
        th32_module_id: u32,
        cnt_threads: u32,
        th32_parent_process_id: u32,
        pc_pri_class_base: i32,
        dw_flags: u32,
        sz_exe_file: [u16; 260],
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateToolhelp32Snapshot(dw_flags: u32, th32_process_id: u32) -> isize;
        fn Process32FirstW(h_snapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
        fn Process32NextW(h_snapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
        fn CloseHandle(h_object: isize) -> i32;
    }

    const KNOWN_CONTROLLERS: &[(&str, &str)] = &[
        ("fancontrol.exe", "FanControl.exe"),
        ("msiafterburner.exe", "MSIAfterburner.exe"),
        ("argusmonitor.exe", "ArgusMonitor.exe"),
        ("hydra.exe", "Hydra.exe"),
    ];

    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE || snap == 0 {
            return None;
        }
        let mut entry: ProcessEntry32W = std::mem::zeroed();
        entry.dw_size = std::mem::size_of::<ProcessEntry32W>() as u32;

        let mut found = None;
        if Process32FirstW(snap, &mut entry) != 0 {
            loop {
                let len = entry
                    .sz_exe_file
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(entry.sz_exe_file.len());
                let name = String::from_utf16_lossy(&entry.sz_exe_file[..len]);
                let lower = name.to_ascii_lowercase();
                for &(needle, display) in KNOWN_CONTROLLERS {
                    if lower == needle {
                        found = Some(display.to_string());
                        break;
                    }
                }
                if found.is_some() || Process32NextW(snap, &mut entry) == 0 {
                    break;
                }
            }
        }
        let _ = CloseHandle(snap);
        found
    }
}

impl AdlxSession {
    fn require_tuning_services(&self) -> ForgeResult<(*mut c_void, &IAdlxGpuTuningServicesVtbl)> {
        let ptr = self
            .tuning_services_ptr()
            .ok_or_else(|| ForgeError::RuntimeUnavailable {
                runtime: "ADLX Tuning".to_string(),
                reason: "IADLXGPUTuningServices is unavailable on this driver".to_string(),
            })?;
        let raw_vtbl = unsafe { *(ptr.cast::<*const IAdlxGpuTuningServicesVtbl>()) };
        Ok((ptr.cast::<c_void>(), unsafe { &*raw_vtbl }))
    }

    fn resolve_target_gpu<F>(
        &self,
        vt: &IAdlxGpuTuningServicesVtbl,
        svc_ptr: *mut c_void,
        pci_device_id: Option<u32>,
        action_label: &str,
        supports_action: F,
    ) -> ForgeResult<(OwnedGpu, String, Option<u32>, Option<bool>)>
    where
        F: Fn(&TuningCapabilities) -> bool,
    {
        let count = self.gpu_count();
        let mut discrete_matches = Vec::new();

        for idx in 0..count {
            let Some(gpu) = self.gpu_at(idx) else {
                continue;
            };
            let (name, dev_id, discrete, _) = Self::read_gpu_identity(&gpu);
            let (at_factory, caps) = read_capabilities(vt, svc_ptr, gpu.0);

            if let Some(target_id) = pci_device_id {
                if dev_id == Some(target_id) {
                    if !supports_action(&caps) {
                        return Err(ForgeError::UnsupportedCapability(format!(
                            "GPU '{name}' (0x{target_id:04X}) does not support {action_label}"
                        )));
                    }
                    return Ok((gpu, name, dev_id, at_factory));
                }
            } else if discrete && supports_action(&caps) {
                discrete_matches.push((gpu, name, dev_id, at_factory));
            }
        }

        if let Some(target_id) = pci_device_id {
            return Err(ForgeError::DeviceNotFound(format!(
                "no AMD GPU matching PCI device ID 0x{target_id:04X} was found by ADLX"
            )));
        }

        match discrete_matches.len() {
            0 => Err(ForgeError::UnsupportedCapability(format!(
                "no discrete AMD GPU supporting {action_label} was found by ADLX"
            ))),
            1 => Ok(discrete_matches.into_iter().next().unwrap()),
            _ => Err(ForgeError::InvalidParameter(
                "multiple tunable discrete AMD GPUs detected; specify --device-id".to_string(),
            )),
        }
    }

    /// Reads complete ADLX tuning capabilities and current settings across all GPUs (strictly read-only).
    pub fn read_all_tuning(&self) -> ForgeResult<Vec<GpuTuningReport>> {
        let (svc_ptr, vt) = self.require_tuning_services()?;
        let count = self.gpu_count();
        let mut reports = Vec::with_capacity(count as usize);

        for idx in 0..count {
            let Some(gpu) = self.gpu_at(idx) else {
                continue;
            };
            let (gpu_name, pci_device_id, discrete, _) = Self::read_gpu_identity(&gpu);
            let (at_factory, capabilities) = read_capabilities(vt, svc_ptr, gpu.0);

            let preset = if capabilities.preset_tuning {
                read_preset_info(vt, svc_ptr, gpu.0)
            } else {
                None
            };
            let power = if capabilities.manual_power {
                unsafe { get_iface(vt.get_manual_power_tuning, svc_ptr, gpu.0) }.and_then(|raw| {
                    query_with_fallback(&raw, "IADLXManualPowerTuning1", "IADLXManualPowerTuning")
                        .map(|(obj, is_v1)| read_power_info_from_iface(&obj, is_v1))
                })
            } else {
                None
            };
            let gfx = if capabilities.manual_gfx {
                read_gfx_info(vt, svc_ptr, gpu.0)
            } else {
                None
            };
            let vram = if capabilities.manual_vram {
                read_vram_info(vt, svc_ptr, gpu.0)
            } else {
                None
            };
            let fan = if capabilities.manual_fan {
                read_fan_info(vt, svc_ptr, gpu.0)
            } else {
                None
            };

            reports.push(GpuTuningReport {
                gpu_name,
                pci_device_id,
                discrete,
                at_factory,
                capabilities,
                preset,
                power,
                gfx,
                vram,
                fan,
            });
        }

        Ok(reports)
    }

    /// Validates `limit_pct` against hardware ADLX power limit bounds and applies it only when `!dry_run`.
    pub fn set_power_limit(
        &self,
        pci_device_id: Option<u32>,
        limit_pct: i32,
        dry_run: bool,
    ) -> ForgeResult<PowerLimitChangeResult> {
        let (svc_ptr, vt) = self.require_tuning_services()?;
        let (gpu, gpu_name, dev_id, _) = self.resolve_target_gpu(
            vt,
            svc_ptr,
            pci_device_id,
            "ADLX manual power limit tuning",
            |c| c.manual_power,
        )?;

        let raw_power = unsafe { get_iface(vt.get_manual_power_tuning, svc_ptr, gpu.0) }
            .ok_or_else(|| {
                ForgeError::ExecutionFailed(format!(
                    "ADLX GetManualPowerTuning failed for {gpu_name}"
                ))
            })?;
        let (power_obj, is_v1) = query_with_fallback(
            &raw_power,
            "IADLXManualPowerTuning1",
            "IADLXManualPowerTuning",
        )
        .ok_or_else(|| {
            ForgeError::UnsupportedCapability(format!(
                "IADLXManualPowerTuning interface unavailable on {gpu_name}"
            ))
        })?;

        let info = read_power_info_from_iface(&power_obj, is_v1);
        let allowed_range = info.power_limit_range.ok_or_else(|| {
            ForgeError::UnsupportedCapability(format!(
                "ADLX did not expose a valid power limit range for {gpu_name}"
            ))
        })?;

        // Read -> validate -> (dry run stops) -> write -> read back -> compare -> restore on
        // contradiction. That logic lives in `change_power_limit` and is tested against a fake
        // backend; this adapter only forwards to ADLX.
        let pvt = unsafe { vtbl::<IAdlxManualPowerTuning1Vtbl>(&power_obj) };
        let mut control = AdlxPowerControl {
            obj: &power_obj,
            vt: pvt,
        };
        let outcome = change_power_limit(&mut control, &allowed_range, limit_pct, dry_run)?;
        Ok(PowerLimitChangeResult {
            gpu_name,
            pci_device_id: dev_id,
            previous_limit_pct: outcome.previous_pct,
            requested_limit_pct: limit_pct,
            allowed_range,
            default_limit_pct: info.power_limit_default_pct,
            applied: outcome.status != OperationStatus::NotApplied,
            verified_limit_pct: outcome.verified_pct,
            confirmed: outcome.status == OperationStatus::Confirmed,
            status: outcome.status,
            rollback: outcome.rollback,
            rollback_verified_limit_pct: outcome.rollback_verified_pct,
            error: outcome.error,
        })
    }

    /// Validates factory tuning reset eligibility and resets GPU tuning only when `!dry_run`.
    pub fn reset_tuning_to_factory(
        &self,
        pci_device_id: Option<u32>,
        dry_run: bool,
    ) -> ForgeResult<TuningResetResult> {
        let (svc_ptr, vt) = self.require_tuning_services()?;
        let (gpu, gpu_name, dev_id, was_at_factory) = self.resolve_target_gpu(
            vt,
            svc_ptr,
            pci_device_id,
            "ADLX tuning reset",
            TuningCapabilities::any_supported,
        )?;

        let mut control = AdlxResetControl {
            svc_ptr,
            vt,
            gpu: gpu.0,
        };
        let outcome = reset_to_factory(&mut control, dry_run)?;
        let _ = was_at_factory; // the control re-reads the state itself
        Ok(TuningResetResult {
            gpu_name,
            pci_device_id: dev_id,
            was_at_factory: outcome.was_at_factory,
            applied: outcome.api != ApiResult::NotCalled,
            now_at_factory: outcome.now_at_factory,
            confirmed: outcome.status == OperationStatus::Confirmed,
            status: outcome.status,
            rollback: outcome.rollback,
            api: outcome.api,
            api_result_code: outcome.api_result_code,
            error: outcome.error,
        })
    }
}

#[cfg(test)]
mod abi_tests {
    //! ABI layout tests. Slot numbers come from AMD's ADLX SDK headers (IGPUTuning.h,
    //! IGPUManualPowerTuning.h, IGPUManualGFXTuning.h, IGPUManualVRAMTuning.h,
    //! IGPUManualFanTuning.h, IGPUPresetTuning.h, ICollections.h; branch `main`). Each `slot!` is
    //! one verified C-vtable index; they fail if a field is reordered, removed or retyped.
    use super::*;

    /// Asserts that a vtable field sits at the C vtable slot given by the upstream ADLX header.
    macro_rules! slot {
        ($t:ty, $f:ident, $i:expr) => {
            assert_eq!(
                std::mem::offset_of!($t, $f),
                $i * std::mem::size_of::<usize>(),
                concat!(stringify!($t), "::", stringify!($f))
            );
        };
    }

    #[test]
    fn gpu_tuning_services_vtable_matches_the_header() {
        slot!(IAdlxGpuTuningServicesVtbl, is_at_factory, 4);
        slot!(IAdlxGpuTuningServicesVtbl, reset_to_factory, 5);
        slot!(IAdlxGpuTuningServicesVtbl, is_supported_auto_tuning, 6);
        slot!(IAdlxGpuTuningServicesVtbl, is_supported_preset_tuning, 7);
        slot!(
            IAdlxGpuTuningServicesVtbl,
            is_supported_manual_gfx_tuning,
            8
        );
        slot!(
            IAdlxGpuTuningServicesVtbl,
            is_supported_manual_vram_tuning,
            9
        );
        slot!(
            IAdlxGpuTuningServicesVtbl,
            is_supported_manual_fan_tuning,
            10
        );
        slot!(
            IAdlxGpuTuningServicesVtbl,
            is_supported_manual_power_tuning,
            11
        );
        slot!(IAdlxGpuTuningServicesVtbl, get_preset_tuning, 13);
        slot!(IAdlxGpuTuningServicesVtbl, get_manual_gfx_tuning, 14);
        slot!(IAdlxGpuTuningServicesVtbl, get_manual_vram_tuning, 15);
        slot!(IAdlxGpuTuningServicesVtbl, get_manual_fan_tuning, 16);
        slot!(IAdlxGpuTuningServicesVtbl, get_manual_power_tuning, 17);
        assert_eq!(std::mem::size_of::<IAdlxGpuTuningServicesVtbl>(), 18 * 8);
    }

    #[test]
    fn power_tuning_vtable_matches_the_header() {
        // IADLXManualPowerTuning1: 0-2 IADLXInterface, 3-9 inherited, 10-11 own.
        slot!(IAdlxManualPowerTuning1Vtbl, get_power_limit_range, 3);
        slot!(IAdlxManualPowerTuning1Vtbl, get_power_limit, 4);
        slot!(IAdlxManualPowerTuning1Vtbl, set_power_limit, 5);
        slot!(IAdlxManualPowerTuning1Vtbl, is_supported_tdc_limit, 6);
        slot!(IAdlxManualPowerTuning1Vtbl, get_tdc_limit_range, 7);
        slot!(IAdlxManualPowerTuning1Vtbl, get_tdc_limit, 8);
        slot!(IAdlxManualPowerTuning1Vtbl, get_power_limit_default, 10);
        slot!(IAdlxManualPowerTuning1Vtbl, get_tdc_limit_default, 11);
        assert_eq!(std::mem::size_of::<IAdlxManualPowerTuning1Vtbl>(), 12 * 8);
    }

    #[test]
    fn preset_graphics_and_vram_vtables_match_the_header() {
        // IADLXGPUPresetTuning (Set* methods 13-17 are intentionally not declared/called).
        slot!(IAdlxGpuPresetTuningVtbl, is_supported_power_saver, 3);
        slot!(IAdlxGpuPresetTuningVtbl, is_supported_rage, 7);
        slot!(IAdlxGpuPresetTuningVtbl, is_current_power_saver, 8);
        slot!(IAdlxGpuPresetTuningVtbl, is_current_rage, 12);

        // IADLXManualGraphicsTuning2_1: 3-11 from Tuning2, defaults at 12-14.
        slot!(
            IAdlxManualGraphicsTuning2_1Vtbl,
            get_gpu_min_frequency_range,
            3
        );
        slot!(IAdlxManualGraphicsTuning2_1Vtbl, get_gpu_min_frequency, 4);
        slot!(
            IAdlxManualGraphicsTuning2_1Vtbl,
            get_gpu_max_frequency_range,
            6
        );
        slot!(IAdlxManualGraphicsTuning2_1Vtbl, get_gpu_max_frequency, 7);
        slot!(IAdlxManualGraphicsTuning2_1Vtbl, get_gpu_voltage_range, 9);
        slot!(IAdlxManualGraphicsTuning2_1Vtbl, get_gpu_voltage, 10);
        slot!(
            IAdlxManualGraphicsTuning2_1Vtbl,
            get_gpu_min_frequency_default,
            12
        );
        slot!(
            IAdlxManualGraphicsTuning2_1Vtbl,
            get_gpu_max_frequency_default,
            13
        );
        slot!(
            IAdlxManualGraphicsTuning2_1Vtbl,
            get_gpu_voltage_default,
            14
        );
        assert_eq!(
            std::mem::size_of::<IAdlxManualGraphicsTuning2_1Vtbl>(),
            15 * 8
        );

        // IADLXManualVRAMTuning2_1: 3-9 from VRAMTuning2, default at 10.
        slot!(IAdlxManualVramTuning2_1Vtbl, is_supported_memory_timing, 3);
        slot!(
            IAdlxManualVramTuning2_1Vtbl,
            get_supported_memory_timing_description_list,
            4
        );
        slot!(
            IAdlxManualVramTuning2_1Vtbl,
            get_memory_timing_description,
            5
        );
        slot!(
            IAdlxManualVramTuning2_1Vtbl,
            get_max_vram_frequency_range,
            7
        );
        slot!(IAdlxManualVramTuning2_1Vtbl, get_max_vram_frequency, 8);
        slot!(
            IAdlxManualVramTuning2_1Vtbl,
            get_max_vram_frequency_default,
            10
        );
        assert_eq!(std::mem::size_of::<IAdlxManualVramTuning2_1Vtbl>(), 11 * 8);

        // IADLXMemoryTimingDescriptionList (IADLXList + At_MemoryTimingDescriptionList=11) and
        // IADLXMemoryTimingDescription::GetDescription=3.
        slot!(IAdlxMemoryTimingDescriptionListVtbl, size, 3);
        slot!(IAdlxMemoryTimingDescriptionListVtbl, at_list, 11);
        slot!(IAdlxMemoryTimingDescriptionVtbl, get_description, 3);
    }

    #[test]
    fn fan_vtables_match_the_header() {
        // IADLXManualFanTuning1: 3-22 inherited from IADLXManualFanTuning, 23-27 own.
        slot!(IAdlxManualFanTuning1Vtbl, get_fan_tuning_ranges, 3);
        slot!(IAdlxManualFanTuning1Vtbl, get_fan_tuning_states, 4);
        slot!(IAdlxManualFanTuning1Vtbl, is_supported_zero_rpm, 8);
        slot!(IAdlxManualFanTuning1Vtbl, get_zero_rpm_state, 9);
        slot!(
            IAdlxManualFanTuning1Vtbl,
            is_supported_min_acoustic_limit,
            11
        );
        slot!(IAdlxManualFanTuning1Vtbl, get_min_acoustic_limit_range, 12);
        slot!(IAdlxManualFanTuning1Vtbl, get_min_acoustic_limit, 13);
        slot!(IAdlxManualFanTuning1Vtbl, is_supported_min_fan_speed, 15);
        slot!(IAdlxManualFanTuning1Vtbl, get_min_fan_speed_range, 16);
        slot!(IAdlxManualFanTuning1Vtbl, get_min_fan_speed, 17);
        slot!(IAdlxManualFanTuning1Vtbl, is_supported_target_fan_speed, 19);
        slot!(IAdlxManualFanTuning1Vtbl, get_target_fan_speed_range, 20);
        slot!(IAdlxManualFanTuning1Vtbl, get_target_fan_speed, 21);
        slot!(IAdlxManualFanTuning1Vtbl, get_default_fan_tuning_states, 23);
        slot!(
            IAdlxManualFanTuning1Vtbl,
            get_min_acoustic_limit_default,
            24
        );
        slot!(IAdlxManualFanTuning1Vtbl, get_min_fan_speed_default, 25);
        slot!(IAdlxManualFanTuning1Vtbl, get_target_fan_speed_default, 26);
        slot!(IAdlxManualFanTuning1Vtbl, get_default_zero_rpm_state, 27);
        assert_eq!(std::mem::size_of::<IAdlxManualFanTuning1Vtbl>(), 28 * 8);

        // IADLXManualFanTuningStateList: IADLXList 3-10, At_ManualFanTuningStateList=11.
        slot!(IAdlxManualFanTuningStateListVtbl, size, 3);
        slot!(IAdlxManualFanTuningStateListVtbl, at_list, 11);
        // IADLXManualFanTuningState: GetFanSpeed=3, SetFanSpeed=4, GetTemperature=5.
        slot!(IAdlxManualFanTuningStateVtbl, get_fan_speed, 3);
        slot!(IAdlxManualFanTuningStateVtbl, get_temperature, 5);
    }

    #[test]
    fn int_range_matches_adlx_intrange() {
        // ADLX_IntRange { adlx_int minValue, maxValue, step } with adlx_int = int32.
        assert_eq!(std::mem::size_of::<AdlxIntRange>(), 12);
        assert_eq!(std::mem::align_of::<AdlxIntRange>(), 4);
        assert_eq!(std::mem::offset_of!(AdlxIntRange, min_value), 0);
        assert_eq!(std::mem::offset_of!(AdlxIntRange, max_value), 4);
        assert_eq!(std::mem::offset_of!(AdlxIntRange, step), 8);
    }
}
