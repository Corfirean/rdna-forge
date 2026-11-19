use std::fmt;

use forge_core::{ForgeError, ForgeResult};
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

/// Hardware video encoder target codec and pixel format for VCN benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmfEncodeCodec {
    H264Avc,
    Hevc,
    Hevc10Bit,
}

impl AmfEncodeCodec {
    /// Human-readable codec label.
    pub const fn codec_name(self) -> &'static str {
        match self {
            Self::H264Avc => "H.264/AVC",
            Self::Hevc => "HEVC/H.265",
            Self::Hevc10Bit => "HEVC/H.265 10-bit",
        }
    }

    /// AMF encoder component identifier string.
    pub const fn component_id(self) -> &'static str {
        match self {
            Self::H264Avc => "AMFVideoEncoderVCE_AVC",
            Self::Hevc | Self::Hevc10Bit => "AMFVideoEncoderHW_HEVC",
        }
    }

    /// Raw `AMF_SURFACE_FORMAT` code (`1 = NV12`, `10 = P010`).
    pub const fn surface_format_code(self) -> i32 {
        match self {
            Self::H264Avc | Self::Hevc => 1,
            Self::Hevc10Bit => 10,
        }
    }

    /// Canonical surface format name (`"NV12"` or `"P010"`).
    pub const fn surface_format_name(self) -> &'static str {
        match self {
            Self::H264Avc | Self::Hevc => "NV12",
            Self::Hevc10Bit => "P010",
        }
    }
}

impl fmt::Display for AmfEncodeCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.codec_name())
    }
}

/// Hardware encoder quality/speed preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmfQualityPreset {
    Speed,
    Balanced,
    Quality,
}

impl AmfQualityPreset {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Speed => "speed",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
        }
    }

    /// Maps to `AMF_VIDEO_ENCODER_QUALITY_PRESET_ENUM` (`Balanced=0, Speed=1, Quality=2`).
    pub const fn avc_value(self) -> i64 {
        match self {
            Self::Balanced => 0,
            Self::Speed => 1,
            Self::Quality => 2,
        }
    }

    /// Maps to `AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_ENUM` (`Quality=0, Balanced=5, Speed=10`).
    pub const fn hevc_value(self) -> i64 {
        match self {
            Self::Quality => 0,
            Self::Balanced => 5,
            Self::Speed => 10,
        }
    }
}

impl fmt::Display for AmfQualityPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// Configuration for a single VCN hardware encode benchmark pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmfEncodeBenchConfig {
    pub codec: AmfEncodeCodec,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub target_bitrate_bps: u64,
    pub preset: AmfQualityPreset,
    pub warmup_frames: u32,
    pub measured_frames: u32,
}

impl AmfEncodeBenchConfig {
    /// Validates resolution, alignment, frame rate, bitrate, and frame counts.
    pub fn validate(&self) -> ForgeResult<()> {
        if !(128..=7680).contains(&self.width) || !self.width.is_multiple_of(2) {
            return Err(ForgeError::InvalidDimension(format!(
                "encode width must be an even integer in 128..=7680 (got {})",
                self.width
            )));
        }
        if !(128..=4320).contains(&self.height) || !self.height.is_multiple_of(2) {
            return Err(ForgeError::InvalidDimension(format!(
                "encode height must be an even integer in 128..=4320 (got {})",
                self.height
            )));
        }
        if !(1..=240).contains(&self.fps) {
            return Err(ForgeError::InvalidDimension(format!(
                "encode fps must be in 1..=240 (got {})",
                self.fps
            )));
        }
        if !(100_000..=500_000_000).contains(&self.target_bitrate_bps) {
            return Err(ForgeError::InvalidDimension(format!(
                "target_bitrate_bps must be in 100_000..=500_000_000 (got {})",
                self.target_bitrate_bps
            )));
        }
        if self.warmup_frames > 1_000 {
            return Err(ForgeError::InvalidDimension(format!(
                "warmup_frames must be <= 1000 (got {})",
                self.warmup_frames
            )));
        }
        if !(1..=10_000).contains(&self.measured_frames) {
            return Err(ForgeError::InvalidDimension(format!(
                "measured_frames must be in 1..=10000 (got {})",
                self.measured_frames
            )));
        }
        Ok(())
    }
}

/// Measured throughput and bitstream verification report for a single AMF hardware encode pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmfEncodeBenchResult {
    pub codec: String,
    pub component_id: String,
    pub surface_format: String,
    pub width: u32,
    pub height: u32,
    pub target_fps: u32,
    pub target_bitrate_mbps: f64,
    pub preset: String,
    pub warmup_frames: u32,
    pub submitted_frames: u32,
    pub output_packets: u32,
    pub total_encoded_bytes: u64,
    pub elapsed_ms: f64,
    pub encode_fps: f64,
    pub megapixels_per_sec: f64,
    pub avg_frame_ms: f64,
    pub effective_bitrate_mbps: f64,
    pub valid_annex_b: bool,
}

/// C-ABI representation of `AMFVariantStruct` (`core/Variant.h`, 24 bytes, 8-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmfVariantStruct {
    pub ty: i32,
    pub reserved: u32,
    pub data0: i64,
    pub data1: i64,
}

impl AmfVariantStruct {
    pub const TYPE_INT64: i32 = 2;
    pub const TYPE_SIZE: i32 = 5;
    pub const TYPE_RATE: i32 = 7;

    pub const fn from_i64(val: i64) -> Self {
        Self {
            ty: Self::TYPE_INT64,
            reserved: 0,
            data0: val,
            data1: 0,
        }
    }

    pub const fn from_size(width: i32, height: i32) -> Self {
        let lo = (width as u32) as i64;
        let hi = ((height as u32) as i64) << 32;
        Self {
            ty: Self::TYPE_SIZE,
            reserved: 0,
            data0: lo | hi,
            data1: 0,
        }
    }

    pub const fn from_rate(num: u32, den: u32) -> Self {
        let lo = num as i64;
        let hi = (den as i64) << 32;
        Self {
            ty: Self::TYPE_RATE,
            reserved: 0,
            data0: lo | hi,
            data1: 0,
        }
    }
}

/// Checks whether an encoded bitstream packet begins with or contains an Annex-B NAL start code
/// (`00 00 00 01` or `00 00 01`) within its leading header window.
pub fn has_annex_b_start_code(bytes: &[u8]) -> bool {
    let prefix = &bytes[..bytes.len().min(64)];
    prefix.windows(4).any(|w| w == [0, 0, 0, 1]) || prefix.windows(3).any(|w| w == [0, 0, 1])
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
    use std::time::{Duration, Instant};

    use forge_core::{ForgeError, ForgeResult};

    use super::{
        amf_surface_format_name, format_amf_version, has_annex_b_start_code, AmfAcceleration,
        AmfCapabilitiesReport, AmfCodecCapability, AmfCodecDirection, AmfEncodeBenchConfig,
        AmfEncodeBenchResult, AmfEncodeCodec, AmfVariantStruct,
    };

    const AMF_OK: i32 = 0;
    const AMF_EOF: i32 = 23;
    const AMF_REPEAT: i32 = 24;
    const AMF_INPUT_FULL: i32 = 25;
    const AMF_NEED_MORE_INPUT: i32 = 44;

    const AMF_DX11_0: i32 = 110;
    const AMF_MEMORY_HOST: i32 = 1;
    const AMF_MEMORY_DX11: i32 = 3;
    const AMF_SURFACE_P010: i32 = 10;

    const SURFACE_RING_LEN: usize = 16;

    static AMF_PROBE_LOCK: Mutex<()> = Mutex::new(());

    #[repr(C)]
    struct AmfGuid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }

    const AMF_BUFFER_IID: AmfGuid = AmfGuid {
        data1: 0xb04b_7248,
        data2: 0xb6f0,
        data3: 0x4321,
        data4: [0xb6, 0x91, 0xba, 0xa4, 0x74, 0x0f, 0x9f, 0xcb],
    };

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
        _get_dx11_device: *const c_void,                          // [19]
        _lock_dx11: *const c_void,                                // [20]
        _unlock_dx11: *const c_void,                              // [21]
        _init_opencl: *const c_void,                              // [22]
        _get_opencl_context: *const c_void,                       // [23]
        _get_opencl_command_queue: *const c_void,                 // [24]
        _get_opencl_device_id: *const c_void,                     // [25]
        _get_opencl_compute_factory: *const c_void,               // [26]
        _init_opencl_ex: *const c_void,                           // [27]
        _lock_opencl: *const c_void,                              // [28]
        _unlock_opencl: *const c_void,                            // [29]
        _init_opengl: *const c_void,                              // [30]
        _get_opengl_context: *const c_void,                       // [31]
        _get_opengl_drawable: *const c_void,                      // [32]
        _lock_opengl: *const c_void,                              // [33]
        _unlock_opengl: *const c_void,                            // [34]
        _init_xv: *const c_void,                                  // [35]
        _get_xv_device: *const c_void,                            // [36]
        _lock_xv: *const c_void,                                  // [37]
        _unlock_xv: *const c_void,                                // [38]
        _init_gralloc: *const c_void,                             // [39]
        _get_gralloc_device: *const c_void,                       // [40]
        _lock_gralloc: *const c_void,                             // [41]
        _unlock_gralloc: *const c_void,                           // [42]
        _alloc_buffer: *const c_void,                             // [43]
        alloc_surface:
            unsafe extern "system" fn(*mut c_void, i32, i32, i32, i32, *mut *mut AmfSurface) -> i32, // [44]
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
        set_property: unsafe extern "system" fn(*mut c_void, *const u16, AmfVariantStruct) -> i32, // [3]
        _get_property: *const c_void,              // [4]
        _has_property: *const c_void,              // [5]
        _get_property_count: *const c_void,        // [6]
        _get_property_at: *const c_void,           // [7]
        _clear: *const c_void,                     // [8]
        _add_to: *const c_void,                    // [9]
        _copy_to: *const c_void,                   // [10]
        _add_observer: *const c_void,              // [11]
        _remove_observer: *const c_void,           // [12]
        _get_properties_info_count: *const c_void, // [13]
        _get_property_info: *const c_void,         // [14]
        _get_property_info_at: *const c_void,      // [15]
        _validate_property: *const c_void,         // [16]
        init: unsafe extern "system" fn(*mut c_void, i32, i32, i32) -> i32, // [17]
        _re_init: *const c_void,                   // [18]
        terminate: unsafe extern "system" fn(*mut c_void) -> i32, // [19]
        drain: unsafe extern "system" fn(*mut c_void) -> i32, // [20]
        _flush: *const c_void,                     // [21]
        submit_input: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32, // [22]
        query_output: unsafe extern "system" fn(*mut c_void, *mut *mut AmfData) -> i32, // [23]
        _get_context: *const c_void,               // [24]
        _set_output_data_allocator_cb: *const c_void, // [25]
        get_caps: unsafe extern "system" fn(*mut c_void, *mut *mut AmfCaps) -> i32, // [26]
    }

    #[repr(C)]
    struct AmfData {
        vtbl: *const AmfDataVtbl,
    }

    // Verified against AMF v1.5.3 `core/Data.h`.
    #[repr(C)]
    struct AmfDataVtbl {
        _acquire: *const c_void,                                  // [0]
        release: unsafe extern "system" fn(*mut c_void) -> isize, // [1]
        query_interface:
            unsafe extern "system" fn(*mut c_void, *const AmfGuid, *mut *mut c_void) -> i32, // [2]
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
        _get_memory_type: *const c_void,                          // [13]
        _duplicate: *const c_void,                                // [14]
        convert: unsafe extern "system" fn(*mut c_void, i32) -> i32, // [15]
        _interop: *const c_void,                                  // [16]
        _get_data_type: *const c_void,                            // [17]
        _is_reusable: *const c_void,                              // [18]
        set_pts: unsafe extern "system" fn(*mut c_void, i64),     // [19]
        _get_pts: *const c_void,                                  // [20]
        set_duration: unsafe extern "system" fn(*mut c_void, i64), // [21]
        _get_duration: *const c_void,                             // [22]
    }

    #[repr(C)]
    struct AmfSurface {
        vtbl: *const AmfSurfaceVtbl,
    }

    // Verified against AMF v1.5.3 `core/Surface.h` (extends AMFData).
    #[repr(C)]
    struct AmfSurfaceVtbl {
        data: AmfDataVtbl,                                                 // [0..22]
        _get_format: *const c_void,                                        // [23]
        get_planes_count: unsafe extern "system" fn(*mut c_void) -> usize, // [24]
        get_plane_at: unsafe extern "system" fn(*mut c_void, usize) -> *mut AmfPlane, // [25]
    }

    #[repr(C)]
    struct AmfPlane {
        vtbl: *const AmfPlaneVtbl,
    }

    // Verified against AMF v1.5.3 `core/Plane.h` (extends AMFInterface).
    #[repr(C)]
    struct AmfPlaneVtbl {
        _acquire: *const c_void,                                       // [0]
        _release: *const c_void,                                       // [1]
        _query_interface: *const c_void,                               // [2]
        _get_type: *const c_void,                                      // [3]
        get_native: unsafe extern "system" fn(*mut c_void) -> *mut u8, // [4]
        _get_pixel_size_in_bytes: *const c_void,                       // [5]
        _get_offset_x: *const c_void,                                  // [6]
        _get_offset_y: *const c_void,                                  // [7]
        get_width: unsafe extern "system" fn(*mut c_void) -> i32,      // [8]
        get_height: unsafe extern "system" fn(*mut c_void) -> i32,     // [9]
        get_h_pitch: unsafe extern "system" fn(*mut c_void) -> i32,    // [10]
    }

    #[repr(C)]
    struct AmfBuffer {
        vtbl: *const AmfBufferVtbl,
    }

    // Verified against AMF v1.5.3 `core/Buffer.h` (extends AMFData).
    #[repr(C)]
    struct AmfBufferVtbl {
        data: AmfDataVtbl,                                               // [0..22]
        _set_size: *const c_void,                                        // [23]
        get_size: unsafe extern "system" fn(*mut c_void) -> usize,       // [24]
        get_native: unsafe extern "system" fn(*mut c_void) -> *const u8, // [25]
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

    struct OwnedSurface(*mut AmfSurface);
    impl Drop for OwnedSurface {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).data.release)(self.0.cast()) };
                self.0 = ptr::null_mut();
            }
        }
    }

    struct OwnedData(*mut AmfData);
    impl Drop for OwnedData {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).release)(self.0.cast()) };
                self.0 = ptr::null_mut();
            }
        }
    }

    struct OwnedBuffer(*mut AmfBuffer);
    impl Drop for OwnedBuffer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ((*(*self.0).vtbl).data.release)(self.0.cast()) };
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

    fn set_comp_property(
        comp: *mut AmfComponent,
        name: &str,
        value: AmfVariantStruct,
    ) -> ForgeResult<()> {
        let key = utf16_null(name);
        let rc = unsafe { ((*(*comp).vtbl).set_property)(comp.cast(), key.as_ptr(), value) };
        if rc != AMF_OK {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFComponent::SetProperty({name}) failed with code {rc}"
            )));
        }
        Ok(())
    }

    fn fill_host_surface_pattern(
        surface: *mut AmfSurface,
        fmt_code: i32,
        frame_idx: usize,
    ) -> ForgeResult<()> {
        let surf_vt = unsafe { &*(*surface).vtbl };
        let planes = unsafe { (surf_vt.get_planes_count)(surface.cast()) };
        if planes < 2 {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFSurface::GetPlanesCount returned {planes} (expected >= 2 for Y/UV)"
            )));
        }

        // Note: GetPlaneAt returns a borrowed AMFPlane* pointer owned by AMFSurface (do NOT Release).
        let y_plane = unsafe { (surf_vt.get_plane_at)(surface.cast(), 0) };
        let uv_plane = unsafe { (surf_vt.get_plane_at)(surface.cast(), 1) };
        if y_plane.is_null() || uv_plane.is_null() {
            return Err(ForgeError::ExecutionFailed(
                "AMFSurface::GetPlaneAt returned NULL plane pointer".to_string(),
            ));
        }

        unsafe {
            let y_vt = &*(*y_plane).vtbl;
            let y_ptr = (y_vt.get_native)(y_plane.cast());
            let y_w = (y_vt.get_width)(y_plane.cast()) as usize;
            let y_h = (y_vt.get_height)(y_plane.cast()) as usize;
            let y_pitch = (y_vt.get_h_pitch)(y_plane.cast()) as usize;

            let uv_vt = &*(*uv_plane).vtbl;
            let uv_ptr = (uv_vt.get_native)(uv_plane.cast());
            let uv_w = (uv_vt.get_width)(uv_plane.cast()) as usize;
            let uv_h = (uv_vt.get_height)(uv_plane.cast()) as usize;
            let uv_pitch = (uv_vt.get_h_pitch)(uv_plane.cast()) as usize;

            if y_ptr.is_null() || uv_ptr.is_null() {
                return Err(ForgeError::ExecutionFailed(
                    "AMFPlane::GetNative returned NULL host pointer".to_string(),
                ));
            }

            if fmt_code == AMF_SURFACE_P010 {
                let min_y_bytes = y_w * 2;
                if y_pitch < min_y_bytes {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "P010 Y plane pitch ({y_pitch}) smaller than row bytes ({min_y_bytes})"
                    )));
                }
                for row in 0..y_h {
                    let row_slice =
                        std::slice::from_raw_parts_mut(y_ptr.add(row * y_pitch), min_y_bytes);
                    for col in 0..y_w {
                        let y10 = (64 + ((col + row * 2 + frame_idx * 17) % 876)) as u16;
                        let bytes = (y10 << 6).to_le_bytes();
                        row_slice[col * 2] = bytes[0];
                        row_slice[col * 2 + 1] = bytes[1];
                    }
                }

                let min_uv_bytes = uv_w * 4;
                if uv_pitch < min_uv_bytes {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "P010 UV plane pitch ({uv_pitch}) smaller than row bytes ({min_uv_bytes})"
                    )));
                }
                for row in 0..uv_h {
                    let row_slice =
                        std::slice::from_raw_parts_mut(uv_ptr.add(row * uv_pitch), min_uv_bytes);
                    for col in 0..uv_w {
                        let u10 = (256 + ((col * 5 + frame_idx * 11) % 512)) as u16;
                        let v10 = (256 + ((row * 5 + frame_idx * 13) % 512)) as u16;
                        let u_bytes = (u10 << 6).to_le_bytes();
                        let v_bytes = (v10 << 6).to_le_bytes();
                        let base = col * 4;
                        row_slice[base] = u_bytes[0];
                        row_slice[base + 1] = u_bytes[1];
                        row_slice[base + 2] = v_bytes[0];
                        row_slice[base + 3] = v_bytes[1];
                    }
                }
            } else {
                if y_pitch < y_w {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "NV12 Y plane pitch ({y_pitch}) smaller than width ({y_w})"
                    )));
                }
                for row in 0..y_h {
                    let row_slice = std::slice::from_raw_parts_mut(y_ptr.add(row * y_pitch), y_w);
                    for (col, px) in row_slice.iter_mut().enumerate() {
                        *px = 16u8.wrapping_add(((col + row * 2 + frame_idx * 9) % 219) as u8);
                    }
                }

                let min_uv_bytes = uv_w * 2;
                if uv_pitch < min_uv_bytes {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "NV12 UV plane pitch ({uv_pitch}) smaller than row bytes ({min_uv_bytes})"
                    )));
                }
                for row in 0..uv_h {
                    let row_slice =
                        std::slice::from_raw_parts_mut(uv_ptr.add(row * uv_pitch), min_uv_bytes);
                    for col in 0..uv_w {
                        row_slice[col * 2] =
                            64u8.wrapping_add(((col * 3 + frame_idx * 5) % 128) as u8);
                        row_slice[col * 2 + 1] =
                            64u8.wrapping_add(((row * 3 + frame_idx * 7) % 128) as u8);
                    }
                }
            }
        }
        Ok(())
    }

    fn create_and_init_encoder(
        factory_ptr: *mut AmfFactory,
        ctx_ptr: *mut AmfContext,
        config: &AmfEncodeBenchConfig,
    ) -> ForgeResult<OwnedComponent> {
        let cid = utf16_null(config.codec.component_id());
        let mut comp_ptr: *mut AmfComponent = ptr::null_mut();
        let create_rc = unsafe {
            ((*(*factory_ptr).vtbl).create_component)(
                factory_ptr.cast(),
                ctx_ptr,
                cid.as_ptr(),
                &mut comp_ptr,
            )
        };
        if create_rc != AMF_OK || comp_ptr.is_null() {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFFactory::CreateComponent({}) failed with code {create_rc}",
                config.codec.component_id()
            )));
        }
        let comp = OwnedComponent(comp_ptr);
        let w = config.width as i32;
        let h = config.height as i32;
        let bitrate = config.target_bitrate_bps as i64;

        match config.codec {
            AmfEncodeCodec::H264Avc => {
                set_comp_property(comp.0, "Usage", AmfVariantStruct::from_i64(0))?;
                set_comp_property(
                    comp.0,
                    "QualityPreset",
                    AmfVariantStruct::from_i64(config.preset.avc_value()),
                )?;
                set_comp_property(comp.0, "TargetBitrate", AmfVariantStruct::from_i64(bitrate))?;
                set_comp_property(comp.0, "FrameSize", AmfVariantStruct::from_size(w, h))?;
                set_comp_property(
                    comp.0,
                    "FrameRate",
                    AmfVariantStruct::from_rate(config.fps, 1),
                )?;
            }
            AmfEncodeCodec::Hevc => {
                set_comp_property(comp.0, "HevcUsage", AmfVariantStruct::from_i64(0))?;
                set_comp_property(
                    comp.0,
                    "HevcQualityPreset",
                    AmfVariantStruct::from_i64(config.preset.hevc_value()),
                )?;
                set_comp_property(
                    comp.0,
                    "HevcTargetBitrate",
                    AmfVariantStruct::from_i64(bitrate),
                )?;
                set_comp_property(comp.0, "HevcFrameSize", AmfVariantStruct::from_size(w, h))?;
                set_comp_property(
                    comp.0,
                    "HevcFrameRate",
                    AmfVariantStruct::from_rate(config.fps, 1),
                )?;
            }
            AmfEncodeCodec::Hevc10Bit => {
                set_comp_property(comp.0, "HevcUsage", AmfVariantStruct::from_i64(0))?;
                set_comp_property(comp.0, "HevcProfile", AmfVariantStruct::from_i64(2))?;
                set_comp_property(comp.0, "HevcColorBitDepth", AmfVariantStruct::from_i64(10))?;
                set_comp_property(
                    comp.0,
                    "HevcQualityPreset",
                    AmfVariantStruct::from_i64(config.preset.hevc_value()),
                )?;
                set_comp_property(
                    comp.0,
                    "HevcTargetBitrate",
                    AmfVariantStruct::from_i64(bitrate),
                )?;
                set_comp_property(comp.0, "HevcFrameSize", AmfVariantStruct::from_size(w, h))?;
                set_comp_property(
                    comp.0,
                    "HevcFrameRate",
                    AmfVariantStruct::from_rate(config.fps, 1),
                )?;
            }
        }

        let fmt_code = config.codec.surface_format_code();
        let init_rc = unsafe { ((*(*comp.0).vtbl).init)(comp.0.cast(), fmt_code, w, h) };
        if init_rc != AMF_OK {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFComponent::Init({}, {}x{}, fmt={}) failed with code {init_rc}",
                config.codec.component_id(),
                w,
                h,
                config.codec.surface_format_name()
            )));
        }
        Ok(comp)
    }

    fn drain_ready_packets(
        comp: *mut AmfComponent,
        output_packets: &mut u32,
        total_encoded_bytes: &mut u64,
        valid_annex_b: &mut bool,
    ) -> ForgeResult<bool> {
        let comp_vt = unsafe { &*(*comp).vtbl };
        loop {
            let mut data_ptr: *mut AmfData = ptr::null_mut();
            let q_rc = unsafe { (comp_vt.query_output)(comp.cast(), &mut data_ptr) };
            if q_rc == AMF_OK && !data_ptr.is_null() {
                let data = OwnedData(data_ptr);
                let mut buf_void: *mut c_void = ptr::null_mut();
                let qi_rc = unsafe {
                    ((*(*data.0).vtbl).query_interface)(
                        data.0.cast(),
                        &AMF_BUFFER_IID,
                        &mut buf_void,
                    )
                };
                if qi_rc == AMF_OK && !buf_void.is_null() {
                    let buf = OwnedBuffer(buf_void.cast());
                    let buf_vt = unsafe { &*(*buf.0).vtbl };
                    let size = unsafe { (buf_vt.get_size)(buf.0.cast()) };
                    let native = unsafe { (buf_vt.get_native)(buf.0.cast()) };
                    *output_packets += 1;
                    *total_encoded_bytes += size as u64;
                    if !*valid_annex_b && size >= 4 && !native.is_null() {
                        let prefix = unsafe { std::slice::from_raw_parts(native, size.min(64)) };
                        if has_annex_b_start_code(prefix) {
                            *valid_annex_b = true;
                        }
                    }
                }
            } else if q_rc == AMF_EOF {
                return Ok(true);
            } else if q_rc == AMF_OK || q_rc == AMF_REPEAT || q_rc == AMF_NEED_MORE_INPUT {
                return Ok(false);
            } else {
                return Err(ForgeError::ExecutionFailed(format!(
                    "AMFComponent::QueryOutput failed with code {q_rc}"
                )));
            }
        }
    }

    fn execute_encode_pass(
        comp: *mut AmfComponent,
        ring: &[OwnedSurface],
        frame_count: u32,
        fps: u32,
    ) -> ForgeResult<(u32, u64, bool)> {
        let comp_vt = unsafe { &*(*comp).vtbl };
        let frame_duration_100ns = (10_000_000i64 / i64::from(fps.max(1))).max(1);
        let mut output_packets = 0u32;
        let mut total_encoded_bytes = 0u64;
        let mut valid_annex_b = false;

        for idx in 0..(frame_count as usize) {
            let surf = ring[idx % ring.len()].0;
            let pts = (idx as i64) * frame_duration_100ns;
            unsafe {
                let data_vt = &(*(*surf).vtbl).data;
                (data_vt.set_pts)(surf.cast(), pts);
                (data_vt.set_duration)(surf.cast(), frame_duration_100ns);
            }

            let mut retries = 0u32;
            loop {
                let sub_rc = unsafe { (comp_vt.submit_input)(comp.cast(), surf.cast()) };
                if sub_rc == AMF_OK {
                    break;
                } else if sub_rc == AMF_INPUT_FULL || sub_rc == AMF_REPEAT {
                    drain_ready_packets(
                        comp,
                        &mut output_packets,
                        &mut total_encoded_bytes,
                        &mut valid_annex_b,
                    )?;
                    retries += 1;
                    if retries > 20_000 {
                        return Err(ForgeError::ExecutionFailed(
                            "AMFComponent::SubmitInput timed out waiting for VCN input queue"
                                .to_string(),
                        ));
                    }
                    std::thread::sleep(Duration::from_micros(50));
                } else {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "AMFComponent::SubmitInput(frame={idx}) failed with code {sub_rc}"
                    )));
                }
            }

            drain_ready_packets(
                comp,
                &mut output_packets,
                &mut total_encoded_bytes,
                &mut valid_annex_b,
            )?;
        }

        let drain_rc = unsafe { (comp_vt.drain)(comp.cast()) };
        if drain_rc != AMF_OK {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFComponent::Drain failed with code {drain_rc}"
            )));
        }

        let mut drain_polls = 0u32;
        loop {
            let eof = drain_ready_packets(
                comp,
                &mut output_packets,
                &mut total_encoded_bytes,
                &mut valid_annex_b,
            )?;
            if eof {
                break;
            }
            drain_polls += 1;
            if drain_polls > 40_000 {
                return Err(ForgeError::ExecutionFailed(
                    "AMFComponent::QueryOutput timed out waiting for AMF_EOF after Drain"
                        .to_string(),
                ));
            }
            std::thread::sleep(Duration::from_micros(50));
        }

        Ok((output_packets, total_encoded_bytes, valid_annex_b))
    }

    /// Executes a hardware VCN video encode benchmark pass on pre-uploaded DX11 surfaces.
    pub fn run_amf_encode_benchmark(
        config: &AmfEncodeBenchConfig,
    ) -> ForgeResult<AmfEncodeBenchResult> {
        config.validate()?;

        let _guard = AMF_PROBE_LOCK
            .lock()
            .map_err(|_| ForgeError::ExecutionFailed("AMF probe mutex poisoned".to_string()))?;

        let lib = unsafe { libloading::Library::new("amfrt64.dll") }.map_err(|e| {
            ForgeError::RuntimeUnavailable {
                runtime: "AMF".to_string(),
                reason: format!("amfrt64.dll unavailable: {e}"),
            }
        })?;

        let factory_ptr = unsafe {
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
            factory
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
        if dx11_rc != AMF_OK {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFContext::InitDX11(NULL, AMF_DX11_0) failed with code {dx11_rc}"
            )));
        }

        // Pre-allocate and pre-upload a 16-surface ring into AMF_MEMORY_DX11 prior to timing.
        let fmt_code = config.codec.surface_format_code();
        let w = config.width as i32;
        let h = config.height as i32;
        let mut ring = Vec::with_capacity(SURFACE_RING_LEN);
        for idx in 0..SURFACE_RING_LEN {
            let mut surf_ptr: *mut AmfSurface = ptr::null_mut();
            let alloc_rc = unsafe {
                ((*(*ctx.0).vtbl).alloc_surface)(
                    ctx.0.cast(),
                    AMF_MEMORY_HOST,
                    fmt_code,
                    w,
                    h,
                    &mut surf_ptr,
                )
            };
            if alloc_rc != AMF_OK || surf_ptr.is_null() {
                return Err(ForgeError::ExecutionFailed(format!(
                    "AMFContext::AllocSurface(HOST, fmt={fmt_code}, {w}x{h}) failed with code {alloc_rc}"
                )));
            }
            let surf = OwnedSurface(surf_ptr);
            fill_host_surface_pattern(surf.0, fmt_code, idx)?;

            let conv_rc =
                unsafe { ((*(*surf.0).vtbl).data.convert)(surf.0.cast(), AMF_MEMORY_DX11) };
            if conv_rc != AMF_OK {
                return Err(ForgeError::ExecutionFailed(format!(
                    "AMFData::Convert(AMF_MEMORY_DX11) failed with code {conv_rc}"
                )));
            }
            ring.push(surf);
        }

        // Optional untimed warmup pass to warm VCN 3.0 hardware clocks & driver queues.
        if config.warmup_frames > 0 {
            let warmup_comp = create_and_init_encoder(factory_ptr, ctx.0, config)?;
            let _ = execute_encode_pass(warmup_comp.0, &ring, config.warmup_frames, config.fps)?;
            drop(warmup_comp);
        }

        // Initialize measured encoder component before starting the timer.
        let measured_comp = create_and_init_encoder(factory_ptr, ctx.0, config)?;
        let start = Instant::now();
        let (output_packets, total_encoded_bytes, valid_annex_b) =
            execute_encode_pass(measured_comp.0, &ring, config.measured_frames, config.fps)?;
        let elapsed_sec = start.elapsed().as_secs_f64();

        // Explicitly drop all AMF objects before unloading `amfrt64.dll`.
        drop(measured_comp);
        drop(ring);
        drop(ctx);
        drop(lib);

        let elapsed_ms = elapsed_sec * 1_000.0;
        let frames_f64 = f64::from(config.measured_frames);
        let encode_fps = if elapsed_sec > 0.0 {
            frames_f64 / elapsed_sec
        } else {
            0.0
        };
        let megapixels_per_sec =
            encode_fps * (f64::from(config.width) * f64::from(config.height)) / 1_000_000.0;
        let avg_frame_ms = if config.measured_frames > 0 {
            elapsed_ms / frames_f64
        } else {
            0.0
        };
        let stream_duration_sec = frames_f64 / f64::from(config.fps.max(1));
        let effective_bitrate_mbps = if stream_duration_sec > 0.0 {
            (total_encoded_bytes as f64 * 8.0) / stream_duration_sec / 1_000_000.0
        } else {
            0.0
        };

        Ok(AmfEncodeBenchResult {
            codec: config.codec.codec_name().to_string(),
            component_id: config.codec.component_id().to_string(),
            surface_format: config.codec.surface_format_name().to_string(),
            width: config.width,
            height: config.height,
            target_fps: config.fps,
            target_bitrate_mbps: config.target_bitrate_bps as f64 / 1_000_000.0,
            preset: config.preset.as_str().to_string(),
            warmup_frames: config.warmup_frames,
            submitted_frames: config.measured_frames,
            output_packets,
            total_encoded_bytes,
            elapsed_ms,
            encode_fps,
            megapixels_per_sec,
            avg_frame_ms,
            effective_bitrate_mbps,
            valid_annex_b,
        })
    }
}

#[cfg(not(windows))]
mod stub {
    use forge_core::{ForgeError, ForgeResult};

    use super::{AmfCapabilitiesReport, AmfEncodeBenchConfig, AmfEncodeBenchResult};

    pub fn probe_amf_capabilities() -> ForgeResult<AmfCapabilitiesReport> {
        Err(ForgeError::RuntimeUnavailable {
            runtime: "AMF".to_string(),
            reason: "AMF video probing is Windows-only".to_string(),
        })
    }

    pub fn run_amf_encode_benchmark(
        config: &AmfEncodeBenchConfig,
    ) -> ForgeResult<AmfEncodeBenchResult> {
        config.validate()?;
        Err(ForgeError::RuntimeUnavailable {
            runtime: "AMF".to_string(),
            reason: "AMF hardware video encoding is Windows-only".to_string(),
        })
    }
}

#[cfg(windows)]
pub use imp::{probe_amf_capabilities, run_amf_encode_benchmark};

#[cfg(not(windows))]
pub use stub::{probe_amf_capabilities, run_amf_encode_benchmark};
