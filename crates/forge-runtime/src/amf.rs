use std::fmt;

use serde::{Deserialize, Serialize};

/// Hardware/runtime acceleration level reported by `AMFCaps::GetAccelerationType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmfAcceleration {
    Hardware,
    Gpu,
    Software,
    NotSupported,
    Unknown(i32),
}

impl AmfAcceleration {
    /// Maps raw `AMF_ACCELERATION_TYPE` (`-1..=2`) into `AmfAcceleration`.
    pub fn from_amf(code: i32) -> Self {
        match code {
            -1 => Self::NotSupported,
            0 => Self::Hardware,
            1 => Self::Gpu,
            2 => Self::Software,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for AmfAcceleration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hardware => f.pad("HARDWARE"),
            Self::Gpu => f.pad("GPU"),
            Self::Software => f.pad("SOFTWARE"),
            Self::NotSupported => f.pad("NOT_SUPPORTED"),
            Self::Unknown(code) => f.pad(&format!("UNKNOWN({code})")),
        }
    }
}

/// Video codec pipeline direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmfCodecDirection {
    Encode,
    Decode,
}

impl fmt::Display for AmfCodecDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode => f.pad("Encode"),
            Self::Decode => f.pad("Decode"),
        }
    }
}

/// AMF hardware capability report for a single video encoder or decoder component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmfCodecCapability {
    pub codec: String,
    pub component_id: String,
    pub direction: AmfCodecDirection,
    pub acceleration: AmfAcceleration,
    pub min_width: Option<i32>,
    pub max_width: Option<i32>,
    pub min_height: Option<i32>,
    pub max_height: Option<i32>,
    pub vert_align: Option<i32>,
    pub interlaced_supported: Option<bool>,
    pub native_formats: Vec<String>,
    pub supported_formats: Vec<String>,
    pub detail: Option<String>,
}

/// Full AMD AMF runtime (`amfrt64.dll`) video capability probe report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmfCapabilitiesReport {
    pub runtime_version: String,
    pub dx11_initialized: bool,
    pub codecs: Vec<AmfCodecCapability>,
}

/// Decodes packed 64-bit AMF version (`16.16.16.16`) into `"major.minor.release.build"`.
pub fn format_amf_version(packed: u64) -> String {
    let major = (packed >> 48) & 0xffff;
    let minor = (packed >> 32) & 0xffff;
    let release = (packed >> 16) & 0xffff;
    let build = packed & 0xffff;
    format!("{major}.{minor}.{release}.{build}")
}

/// Maps an `AMF_SURFACE_FORMAT` enum code (`0..=26`) to its canonical format name.
pub fn amf_surface_format_name(code: i32) -> String {
    match code {
        0 => "UNKNOWN",
        1 => "NV12",
        2 => "YV12",
        3 => "BGRA",
        4 => "ARGB",
        5 => "RGBA",
        6 => "GRAY8",
        7 => "YUV420P",
        8 => "U8V8",
        9 => "YUY2",
        10 => "P010",
        11 => "RGBA_F16",
        12 => "UYVY",
        13 => "R10G10B10A2",
        14 => "Y210",
        15 => "AYUV",
        16 => "Y410",
        17 => "Y416",
        18 => "GRAY32",
        19 => "P012",
        20 => "P016",
        21 => "Y216",
        22 => "R16G16",
        23 => "R24G8",
        24 => "R32",
        25 => "R16",
        26 => "R32G8X24",
        other => return format!("FORMAT_{other}"),
    }
    .to_string()
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::Mutex;

    use forge_core::{ForgeError, ForgeResult};

    use super::{
        amf_surface_format_name, format_amf_version, AmfAcceleration, AmfCapabilitiesReport,
        AmfCodecCapability, AmfCodecDirection,
    };

    const AMF_OK: i32 = 0;
    const AMF_DX11_0: i32 = 110;

    static AMF_PROBE_LOCK: Mutex<()> = Mutex::new(());

    #[repr(C)]
    struct AmfFactory {
        vtbl: *const AmfFactoryVtbl,
    }

    // Verified against GPUOpen AMF v1.5.3 `core/Factory.h` (does NOT inherit AMFInterface).
    #[repr(C)]
    struct AmfFactoryVtbl {
        create_context: unsafe extern "system" fn(*mut c_void, *mut *mut AmfContext) -> i32, // [0]
        create_component: unsafe extern "system" fn(
            *mut c_void,
            *mut AmfContext,
            *const u16,
            *mut *mut AmfComponent,
        ) -> i32, // [1]
    }

    #[repr(C)]
    struct AmfContext {
        vtbl: *const AmfContextVtbl,
    }

    // Verified against AMF v1.5.3 `core/Context.h`.
    #[repr(C)]
    struct AmfContextVtbl {
        _acquire: *const c_void,                                  // [0]
        release: unsafe extern "system" fn(*mut c_void) -> isize, // [1]
        _query_interface: *const c_void,                          // [2]
        _set_property: *const c_void,                             // [3]
        _get_property: *const c_void,                             // [4]
        _has_property: *const c_void,                             // [5]
        _get_property_count: *const c_void,                       // [6]
        _get_property_at: *const c_void,                          // [7]
        _clear: *const c_void,                                    // [8]
        _add_to: *const c_void,                                   // [9]
        _copy_to: *const c_void,                                  // [10]
        _add_observer: *const c_void,                             // [11]
        _remove_observer: *const c_void,                          // [12]
        terminate: unsafe extern "system" fn(*mut c_void) -> i32, // [13]
        _init_dx9: *const c_void,                                 // [14]
        _get_dx9_device: *const c_void,                           // [15]
        _lock_dx9: *const c_void,                                 // [16]
        _unlock_dx9: *const c_void,                               // [17]
        init_dx11: unsafe extern "system" fn(*mut c_void, *mut c_void, i32) -> i32, // [18]
    }

    #[repr(C)]
    struct AmfComponent {
        vtbl: *const AmfComponentVtbl,
    }

    // Verified against AMF v1.5.3 `components/Component.h` (_WIN32 layout).
    #[repr(C)]
    struct AmfComponentVtbl {
        _acquire: *const c_void,                                  // [0]
        release: unsafe extern "system" fn(*mut c_void) -> isize, // [1]
        _query_interface: *const c_void,                          // [2]
        _set_property: *const c_void,                             // [3]
        _get_property: *const c_void,                             // [4]
        _has_property: *const c_void,                             // [5]
        _get_property_count: *const c_void,                       // [6]
        _get_property_at: *const c_void,                          // [7]
        _clear: *const c_void,                                    // [8]
        _add_to: *const c_void,                                   // [9]
        _copy_to: *const c_void,                                  // [10]
        _add_observer: *const c_void,                             // [11]
        _remove_observer: *const c_void,                          // [12]
        _get_properties_info_count: *const c_void,                // [13]
        _get_property_info: *const c_void,                        // [14]
        _get_property_info_at: *const c_void,                     // [15]
        _validate_property: *const c_void,                        // [16]
        _init: *const c_void,                                     // [17]
        _re_init: *const c_void,                                  // [18]
        terminate: unsafe extern "system" fn(*mut c_void) -> i32, // [19]
        _drain: *const c_void,                                    // [20]
        _flush: *const c_void,                                    // [21]
        _submit_input: *const c_void,                             // [22]
        _query_output: *const c_void,                             // [23]
        _get_context: *const c_void,                              // [24]
        _set_output_data_allocator_cb: *const c_void,             // [25]
        get_caps: unsafe extern "system" fn(*mut c_void, *mut *mut AmfCaps) -> i32, // [26]
    }

    #[repr(C)]
    struct AmfCaps {
        vtbl: *const AmfCapsVtbl,
    }

    // Verified against AMF v1.5.3 `components/ComponentCaps.h`.
    #[repr(C)]
    struct AmfCapsVtbl {
        _acquire: *const c_void,                                              // [0]
        release: unsafe extern "system" fn(*mut c_void) -> isize,             // [1]
        _query_interface: *const c_void,                                      // [2]
        _set_property: *const c_void,                                         // [3]
        _get_property: *const c_void,                                         // [4]
        _has_property: *const c_void,                                         // [5]
        _get_property_count: *const c_void,                                   // [6]
        _get_property_at: *const c_void,                                      // [7]
        _clear: *const c_void,                                                // [8]
        _add_to: *const c_void,                                               // [9]
        _copy_to: *const c_void,                                              // [10]
        _add_observer: *const c_void,                                         // [11]
        _remove_observer: *const c_void,                                      // [12]
        get_acceleration_type: unsafe extern "system" fn(*mut c_void) -> i32, // [13]
        get_input_caps: unsafe extern "system" fn(*mut c_void, *mut *mut AmfIoCaps) -> i32, // [14]
        get_output_caps: unsafe extern "system" fn(*mut c_void, *mut *mut AmfIoCaps) -> i32, // [15]
    }

    #[repr(C)]
    struct AmfIoCaps {
        vtbl: *const AmfIoCapsVtbl,
    }

    // Verified against AMF v1.5.3 `components/ComponentCaps.h`.
    #[repr(C)]
    struct AmfIoCapsVtbl {
        _acquire: *const c_void,                                  // [0]
        release: unsafe extern "system" fn(*mut c_void) -> isize, // [1]
        _query_interface: *const c_void,                          // [2]
        get_width_range: unsafe extern "system" fn(*mut c_void, *mut i32, *mut i32), // [3]
        get_height_range: unsafe extern "system" fn(*mut c_void, *mut i32, *mut i32), // [4]
        get_vert_align: unsafe extern "system" fn(*mut c_void) -> i32, // [5]
        get_num_of_formats: unsafe extern "system" fn(*mut c_void) -> i32, // [6]
        get_format_at: unsafe extern "system" fn(*mut c_void, i32, *mut i32, *mut u8) -> i32, // [7]
        _get_num_of_memory_types: *const c_void,                  // [8]
        _get_memory_type_at: *const c_void,                       // [9]
        is_interlaced_supported: unsafe extern "system" fn(*mut c_void) -> u8, // [10]
    }

    type AmfQueryVersionFn = unsafe extern "C" fn(*mut u64) -> i32;
    type AmfInitFn = unsafe extern "C" fn(u64, *mut *mut AmfFactory) -> i32;

    struct OwnedContext(*mut AmfContext);
    impl Drop for OwnedContext {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let vt = &*(*self.0).vtbl;
                    (vt.terminate)(self.0.cast());
                    (vt.release)(self.0.cast());
                }
                self.0 = ptr::null_mut();
            }
        }
    }

    struct OwnedComponent(*mut AmfComponent);
    impl Drop for OwnedComponent {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let vt = &*(*self.0).vtbl;
                    (vt.terminate)(self.0.cast());
                    (vt.release)(self.0.cast());
                }
                self.0 = ptr::null_mut();
            }
        }
    }

    struct OwnedCaps(*mut AmfCaps);
    impl Drop for OwnedCaps {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0.cast()) };
                self.0 = ptr::null_mut();
            }
        }
    }

    struct OwnedIoCaps(*mut AmfIoCaps);
    impl Drop for OwnedIoCaps {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0.cast()) };
                self.0 = ptr::null_mut();
            }
        }
    }

    struct CodecDescriptor {
        codec: &'static str,
        component_id: &'static str,
        direction: AmfCodecDirection,
    }

    const CODECS: &[CodecDescriptor] = &[
        CodecDescriptor {
            codec: "H.264/AVC",
            component_id: "AMFVideoEncoderVCE_AVC",
            direction: AmfCodecDirection::Encode,
        },
        CodecDescriptor {
            codec: "HEVC/H.265",
            component_id: "AMFVideoEncoderHW_HEVC",
            direction: AmfCodecDirection::Encode,
        },
        CodecDescriptor {
            codec: "AV1",
            component_id: "AMFVideoEncoderHW_AV1",
            direction: AmfCodecDirection::Encode,
        },
        CodecDescriptor {
            codec: "H.264/AVC",
            component_id: "AMFVideoDecoderUVD_H264_AVC",
            direction: AmfCodecDirection::Decode,
        },
        CodecDescriptor {
            codec: "HEVC/H.265",
            component_id: "AMFVideoDecoderHW_H265_HEVC",
            direction: AmfCodecDirection::Decode,
        },
        CodecDescriptor {
            codec: "VP9",
            component_id: "AMFVideoDecoderHW_VP9",
            direction: AmfCodecDirection::Decode,
        },
        CodecDescriptor {
            codec: "AV1",
            component_id: "AMFVideoDecoderHW_AV1",
            direction: AmfCodecDirection::Decode,
        },
        CodecDescriptor {
            codec: "MJPEG",
            component_id: "AMFVideoDecoderUVD_MJPEG",
            direction: AmfCodecDirection::Decode,
        },
    ];

    fn utf16_null(ascii: &str) -> Vec<u16> {
        ascii.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn unsupported_codec_entry(desc: &CodecDescriptor, detail: String) -> AmfCodecCapability {
        AmfCodecCapability {
            codec: desc.codec.to_string(),
            component_id: desc.component_id.to_string(),
            direction: desc.direction,
            acceleration: AmfAcceleration::NotSupported,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            vert_align: None,
            interlaced_supported: None,
            native_formats: Vec::new(),
            supported_formats: Vec::new(),
            detail: Some(detail),
        }
    }

    /// Dynamically loads `amfrt64.dll` and inspects AMD VCN hardware video capabilities.
    pub fn probe_amf_capabilities() -> ForgeResult<AmfCapabilitiesReport> {
        let _guard = AMF_PROBE_LOCK
            .lock()
            .map_err(|_| ForgeError::ExecutionFailed("AMF probe mutex poisoned".to_string()))?;

        let lib = unsafe { libloading::Library::new("amfrt64.dll") }.map_err(|e| {
            ForgeError::RuntimeUnavailable {
                runtime: "AMF".to_string(),
                reason: format!("amfrt64.dll unavailable: {e}"),
            }
        })?;

        let (raw_ver, factory_ptr) = unsafe {
            let query_ver = *lib
                .get::<AmfQueryVersionFn>(b"AMFQueryVersion\0")
                .map_err(|e| ForgeError::RuntimeUnavailable {
                    runtime: "AMF".to_string(),
                    reason: format!("AMFQueryVersion missing: {e}"),
                })?;
            let amf_init = *lib.get::<AmfInitFn>(b"AMFInit\0").map_err(|e| {
                ForgeError::RuntimeUnavailable {
                    runtime: "AMF".to_string(),
                    reason: format!("AMFInit missing: {e}"),
                }
            })?;

            let mut ver: u64 = 0;
            let rc = query_ver(&mut ver);
            if rc != AMF_OK || ver == 0 {
                return Err(ForgeError::RuntimeUnavailable {
                    runtime: "AMF".to_string(),
                    reason: format!("AMFQueryVersion returned error {rc}"),
                });
            }

            let mut factory: *mut AmfFactory = ptr::null_mut();
            let init_rc = amf_init(ver, &mut factory);
            if init_rc != AMF_OK || factory.is_null() {
                return Err(ForgeError::RuntimeUnavailable {
                    runtime: "AMF".to_string(),
                    reason: format!("AMFInit failed with code {init_rc}"),
                });
            }
            (ver, factory)
        };

        let mut ctx_ptr: *mut AmfContext = ptr::null_mut();
        let ctx_rc =
            unsafe { ((*(*factory_ptr).vtbl).create_context)(factory_ptr.cast(), &mut ctx_ptr) };
        if ctx_rc != AMF_OK || ctx_ptr.is_null() {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFFactory::CreateContext failed with code {ctx_rc}"
            )));
        }
        let ctx = OwnedContext(ctx_ptr);

        let dx11_rc =
            unsafe { ((*(*ctx.0).vtbl).init_dx11)(ctx.0.cast(), ptr::null_mut(), AMF_DX11_0) };
        let dx11_initialized = dx11_rc == AMF_OK;
        if !dx11_initialized {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFContext::InitDX11(NULL, AMF_DX11_0) failed with code {dx11_rc}"
            )));
        }

        let mut codecs = Vec::with_capacity(CODECS.len());
        for desc in CODECS {
            let cid = utf16_null(desc.component_id);
            let mut comp_ptr: *mut AmfComponent = ptr::null_mut();
            let create_rc = unsafe {
                ((*(*factory_ptr).vtbl).create_component)(
                    factory_ptr.cast(),
                    ctx.0,
                    cid.as_ptr(),
                    &mut comp_ptr,
                )
            };
            if create_rc != AMF_OK || comp_ptr.is_null() {
                codecs.push(unsupported_codec_entry(
                    desc,
                    format!("CreateComponent returned error {create_rc}"),
                ));
                continue;
            }
            let comp = OwnedComponent(comp_ptr);

            let mut caps_ptr: *mut AmfCaps = ptr::null_mut();
            let caps_rc = unsafe { ((*(*comp.0).vtbl).get_caps)(comp.0.cast(), &mut caps_ptr) };
            if caps_rc != AMF_OK || caps_ptr.is_null() {
                codecs.push(unsupported_codec_entry(
                    desc,
                    format!("GetCaps returned error {caps_rc}"),
                ));
                continue;
            }
            let caps = OwnedCaps(caps_ptr);

            let accel_raw = unsafe { ((*(*caps.0).vtbl).get_acceleration_type)(caps.0.cast()) };
            let acceleration = AmfAcceleration::from_amf(accel_raw);

            let mut io_ptr: *mut AmfIoCaps = ptr::null_mut();
            let io_rc = unsafe {
                match desc.direction {
                    AmfCodecDirection::Encode => {
                        ((*(*caps.0).vtbl).get_input_caps)(caps.0.cast(), &mut io_ptr)
                    }
                    AmfCodecDirection::Decode => {
                        ((*(*caps.0).vtbl).get_output_caps)(caps.0.cast(), &mut io_ptr)
                    }
                }
            };

            let mut min_w = None;
            let mut max_w = None;
            let mut min_h = None;
            let mut max_h = None;
            let mut vert_align = None;
            let mut interlaced_supported = None;
            let mut native_formats = Vec::new();
            let mut supported_formats = Vec::new();

            if io_rc == AMF_OK && !io_ptr.is_null() {
                let io = OwnedIoCaps(io_ptr);
                let io_vt = unsafe { &*(*io.0).vtbl };
                let (mut w0, mut w1, mut h0, mut h1) = (0i32, 0i32, 0i32, 0i32);
                unsafe {
                    (io_vt.get_width_range)(io.0.cast(), &mut w0, &mut w1);
                    (io_vt.get_height_range)(io.0.cast(), &mut h0, &mut h1);
                }
                if w0 > 0 && w1 >= w0 {
                    min_w = Some(w0);
                    max_w = Some(w1);
                }
                if h0 > 0 && h1 >= h0 {
                    min_h = Some(h0);
                    max_h = Some(h1);
                }
                vert_align = Some(unsafe { (io_vt.get_vert_align)(io.0.cast()) });
                interlaced_supported =
                    Some(unsafe { (io_vt.is_interlaced_supported)(io.0.cast()) } != 0);

                let num_fmt = unsafe { (io_vt.get_num_of_formats)(io.0.cast()) };
                for idx in 0..num_fmt {
                    let (mut fmt_code, mut native) = (0i32, 0u8);
                    let f_rc = unsafe {
                        (io_vt.get_format_at)(io.0.cast(), idx, &mut fmt_code, &mut native)
                    };
                    if f_rc == AMF_OK {
                        let name = amf_surface_format_name(fmt_code);
                        if native != 0 {
                            native_formats.push(name.clone());
                        }
                        supported_formats.push(name);
                    }
                }
            }

            codecs.push(AmfCodecCapability {
                codec: desc.codec.to_string(),
                component_id: desc.component_id.to_string(),
                direction: desc.direction,
                acceleration,
                min_width: min_w,
                max_width: max_w,
                min_height: min_h,
                max_height: max_h,
                vert_align,
                interlaced_supported,
                native_formats,
                supported_formats,
                detail: None,
            });
        }

        // Explicitly drop OwnedContext while `lib` is still alive, before unloading `amfrt64.dll`.
        drop(ctx);
        drop(lib);

        Ok(AmfCapabilitiesReport {
            runtime_version: format_amf_version(raw_ver),
            dx11_initialized,
            codecs,
        })
    }
}

#[cfg(not(windows))]
mod stub {
    use forge_core::{ForgeError, ForgeResult};

    use super::AmfCapabilitiesReport;

    pub fn probe_amf_capabilities() -> ForgeResult<AmfCapabilitiesReport> {
        Err(ForgeError::RuntimeUnavailable {
            runtime: "AMF".to_string(),
            reason: "AMF video probing is Windows-only".to_string(),
        })
    }
}

#[cfg(windows)]
pub use imp::probe_amf_capabilities;

#[cfg(not(windows))]
pub use stub::probe_amf_capabilities;
