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
    Av1,
    Av110Bit,
}

impl AmfEncodeCodec {
    /// Human-readable codec label.
    pub const fn codec_name(self) -> &'static str {
        match self {
            Self::H264Avc => "H.264/AVC",
            Self::Hevc => "HEVC/H.265",
            Self::Hevc10Bit => "HEVC/H.265 10-bit",
            Self::Av1 => "AV1",
            Self::Av110Bit => "AV1 10-bit",
        }
    }

    /// AMF encoder component identifier string.
    pub const fn component_id(self) -> &'static str {
        match self {
            Self::H264Avc => "AMFVideoEncoderVCE_AVC",
            Self::Hevc | Self::Hevc10Bit => "AMFVideoEncoderHW_HEVC",
            Self::Av1 | Self::Av110Bit => "AMFVideoEncoderHW_AV1",
        }
    }

    /// Raw `AMF_SURFACE_FORMAT` code (`1 = NV12`, `10 = P010`).
    pub const fn surface_format_code(self) -> i32 {
        match self {
            Self::H264Avc | Self::Hevc | Self::Av1 => 1,
            Self::Hevc10Bit | Self::Av110Bit => 10,
        }
    }

    /// Canonical surface format name (`"NV12"` or `"P010"`).
    pub const fn surface_format_name(self) -> &'static str {
        match self {
            Self::H264Avc | Self::Hevc | Self::Av1 => "NV12",
            Self::Hevc10Bit | Self::Av110Bit => "P010",
        }
    }

    /// Returns `true` when this codec emits AV1 Open Bitstream Units (OBU) instead of Annex-B NAL units.
    pub const fn is_av1(self) -> bool {
        matches!(self, Self::Av1 | Self::Av110Bit)
    }
}

/// Hardware video decoder target codec and pixel format for VCN decode benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmfDecodeCodec {
    H264Avc,
    Hevc,
    Hevc10Bit,
}

impl AmfDecodeCodec {
    /// Human-readable codec label.
    pub const fn codec_name(self) -> &'static str {
        match self {
            Self::H264Avc => "H.264/AVC",
            Self::Hevc => "HEVC/H.265",
            Self::Hevc10Bit => "HEVC/H.265 10-bit",
        }
    }

    /// AMF hardware decoder component identifier string.
    pub const fn component_id(self) -> &'static str {
        match self {
            Self::H264Avc => "AMFVideoDecoderUVD_H264_AVC",
            Self::Hevc | Self::Hevc10Bit => "AMFVideoDecoderHW_H265_HEVC",
        }
    }

    /// Matching AMF hardware encoder codec used to synthesize valid input packets in memory.
    pub const fn encode_source_codec(self) -> AmfEncodeCodec {
        match self {
            Self::H264Avc => AmfEncodeCodec::H264Avc,
            Self::Hevc => AmfEncodeCodec::Hevc,
            Self::Hevc10Bit => AmfEncodeCodec::Hevc10Bit,
        }
    }

    /// Raw `AMF_SURFACE_FORMAT` code (`1 = NV12`, `10 = P010`).
    pub const fn surface_format_code(self) -> i32 {
        match self {
            Self::H264Avc | Self::Hevc => 1,
            Self::Hevc10Bit => 10,
        }
    }

    /// Canonical output surface format name (`"NV12"` or `"P010"`).
    pub const fn surface_format_name(self) -> &'static str {
        match self {
            Self::H264Avc | Self::Hevc => "NV12",
            Self::Hevc10Bit => "P010",
        }
    }
}

impl fmt::Display for AmfDecodeCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.codec_name())
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

    /// Maps to `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_ENUM` (`Quality=30, Balanced=70, Speed=100`).
    pub const fn av1_value(self) -> i64 {
        match self {
            Self::Quality => 30,
            Self::Balanced => 70,
            Self::Speed => 100,
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
    #[serde(default)]
    pub valid_bitstream: bool,
}

/// Configuration for a single VCN hardware decode benchmark pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmfDecodeBenchConfig {
    pub codec: AmfDecodeCodec,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub source_bitrate_bps: u64,
    pub warmup_frames: u32,
    pub measured_frames: u32,
}

impl AmfDecodeBenchConfig {
    /// Validates decode resolution, frame rate, source bitrate, and frame counts.
    pub fn validate(&self) -> ForgeResult<()> {
        if !(128..=7680).contains(&self.width) || !self.width.is_multiple_of(2) {
            return Err(ForgeError::InvalidDimension(format!(
                "decode width must be an even integer in 128..=7680 (got {})",
                self.width
            )));
        }
        if !(128..=4320).contains(&self.height) || !self.height.is_multiple_of(2) {
            return Err(ForgeError::InvalidDimension(format!(
                "decode height must be an even integer in 128..=4320 (got {})",
                self.height
            )));
        }
        if !(1..=240).contains(&self.fps) {
            return Err(ForgeError::InvalidDimension(format!(
                "decode fps must be in 1..=240 (got {})",
                self.fps
            )));
        }
        if !(100_000..=500_000_000).contains(&self.source_bitrate_bps) {
            return Err(ForgeError::InvalidDimension(format!(
                "source_bitrate_bps must be in 100_000..=500_000_000 (got {})",
                self.source_bitrate_bps
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

/// Measured throughput and surface verification report for a single AMF hardware decode pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmfDecodeBenchResult {
    pub codec: String,
    pub component_id: String,
    pub surface_format: String,
    pub width: u32,
    pub height: u32,
    pub gop_packets: u32,
    pub source_stream_bytes: u64,
    pub warmup_frames: u32,
    pub submitted_packets: u32,
    pub decoded_frames: u32,
    pub elapsed_ms: f64,
    pub decode_fps: f64,
    pub megapixels_per_sec: f64,
    pub avg_frame_ms: f64,
    pub valid_surfaces: bool,
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

/// Checks whether a leading AV1 packet byte sequence begins with a syntactically valid OBU header
/// (`obu_forbidden_bit == 0` and known `obu_type` in `1..=8`, e.g. SequenceHeader=1, TemporalDelimiter=2, Frame=6).
pub fn has_av1_obu_header(bytes: &[u8]) -> bool {
    let Some(&hdr) = bytes.first() else {
        return false;
    };
    if (hdr & 0x80) != 0 {
        return false;
    }
    let obu_type = (hdr >> 3) & 0x0F;
    (1..=8).contains(&obu_type)
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
        amf_surface_format_name, format_amf_version, has_annex_b_start_code, has_av1_obu_header,
        AmfAcceleration, AmfCapabilitiesReport, AmfCodecCapability, AmfCodecDirection,
        AmfDecodeBenchConfig, AmfDecodeBenchResult, AmfDecodeCodec, AmfEncodeBenchConfig,
        AmfEncodeBenchResult, AmfEncodeCodec, AmfVariantStruct,
    };

    const AMF_OK: i32 = 0;
    const AMF_NOT_SUPPORTED: i32 = 10;
    const AMF_EOF: i32 = 23;
    const AMF_REPEAT: i32 = 24;
    const AMF_INPUT_FULL: i32 = 25;
    const AMF_RESOLUTION_UPDATED: i32 = 27;
    const AMF_CODEC_NOT_SUPPORTED: i32 = 30;
    const AMF_DECODER_NOT_PRESENT: i32 = 33;
    const AMF_DECODER_NO_FREE_SURFACES: i32 = 35;
    const AMF_ENCODER_NOT_PRESENT: i32 = 36;
    const AMF_NEED_MORE_INPUT: i32 = 44;

    const AMF_DATA_SURFACE: i32 = 1;
    const AMF_DX11_0: i32 = 110;
    const AMF_MEMORY_HOST: i32 = 1;
    const AMF_MEMORY_DX11: i32 = 3;
    const AMF_SURFACE_P010: i32 = 10;

    const SURFACE_RING_LEN: usize = 16;
    const DECODE_SOURCE_GOP_FRAMES: u32 = 30;

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

    const AMF_SURFACE_IID: AmfGuid = AmfGuid {
        data1: 0x3075_dbe3,
        data2: 0x8718,
        data3: 0x4cfa,
        data4: [0x86, 0xfb, 0x21, 0x14, 0xc0, 0xa5, 0xa4, 0x51],
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
        alloc_buffer:
            unsafe extern "system" fn(*mut c_void, i32, usize, *mut *mut AmfBuffer) -> i32, // [43]
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
        get_data_type: unsafe extern "system" fn(*mut c_void) -> i32, // [17]
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
        get_format: unsafe extern "system" fn(*mut c_void) -> i32,         // [23]
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
        data: AmfDataVtbl,                                             // [0..22]
        _set_size: *const c_void,                                      // [23]
        get_size: unsafe extern "system" fn(*mut c_void) -> usize,     // [24]
        get_native: unsafe extern "system" fn(*mut c_void) -> *mut u8, // [25]
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

        let lib = forge_core::load_system_library("amfrt64.dll")
            .map_err(|e| e.into_forge_error("AMF"))?
            .library;

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

    fn is_unsupported_amf_code(rc: i32) -> bool {
        matches!(
            rc,
            AMF_NOT_SUPPORTED
                | AMF_CODEC_NOT_SUPPORTED
                | AMF_DECODER_NOT_PRESENT
                | AMF_ENCODER_NOT_PRESENT
        )
    }

    fn verify_component_hardware_caps(comp: *mut AmfComponent, label: &str) -> ForgeResult<()> {
        let mut caps_ptr: *mut AmfCaps = ptr::null_mut();
        let caps_rc = unsafe { ((*(*comp).vtbl).get_caps)(comp.cast(), &mut caps_ptr) };
        if caps_rc == AMF_OK && !caps_ptr.is_null() {
            let caps = OwnedCaps(caps_ptr);
            let accel_raw = unsafe { ((*(*caps.0).vtbl).get_acceleration_type)(caps.0.cast()) };
            if AmfAcceleration::from_amf(accel_raw) == AmfAcceleration::NotSupported {
                return Err(ForgeError::UnsupportedCapability(format!(
                    "{label} reports AMF acceleration NOT_SUPPORTED on this GPU"
                )));
            }
        }
        Ok(())
    }

    fn init_amf_dx11_context() -> ForgeResult<(libloading::Library, *mut AmfFactory, OwnedContext)>
    {
        let lib = forge_core::load_system_library("amfrt64.dll")
            .map_err(|e| e.into_forge_error("AMF"))?
            .library;

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

        Ok((lib, factory_ptr, ctx))
    }

    fn alloc_dx11_surface_ring(
        ctx_ptr: *mut AmfContext,
        fmt_code: i32,
        width: u32,
        height: u32,
    ) -> ForgeResult<Vec<OwnedSurface>> {
        let w = width as i32;
        let h = height as i32;
        let mut ring = Vec::with_capacity(SURFACE_RING_LEN);
        for idx in 0..SURFACE_RING_LEN {
            let mut surf_ptr: *mut AmfSurface = ptr::null_mut();
            let alloc_rc = unsafe {
                ((*(*ctx_ptr).vtbl).alloc_surface)(
                    ctx_ptr.cast(),
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
        Ok(ring)
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
            if is_unsupported_amf_code(create_rc) {
                return Err(ForgeError::UnsupportedCapability(format!(
                    "{} ({}) is unsupported on this AMD GPU (AMF code {create_rc})",
                    config.codec.codec_name(),
                    config.codec.component_id()
                )));
            }
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFFactory::CreateComponent({}) failed with code {create_rc}",
                config.codec.component_id()
            )));
        }
        let comp = OwnedComponent(comp_ptr);
        verify_component_hardware_caps(comp.0, config.codec.codec_name())?;

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
            AmfEncodeCodec::Av1 => {
                // Verified against AMF v1.5.3 VideoEncoderAV1.h & ColorSpace.h
                set_comp_property(comp.0, "Av1Usage", AmfVariantStruct::from_i64(0))?;
                set_comp_property(comp.0, "Av1Profile", AmfVariantStruct::from_i64(1))?;
                set_comp_property(comp.0, "Av1ColorBitDepth", AmfVariantStruct::from_i64(8))?;
                set_comp_property(comp.0, "Av1AlignmentMode", AmfVariantStruct::from_i64(3))?;
                set_comp_property(
                    comp.0,
                    "Av1QualityPreset",
                    AmfVariantStruct::from_i64(config.preset.av1_value()),
                )?;
                set_comp_property(
                    comp.0,
                    "Av1TargetBitrate",
                    AmfVariantStruct::from_i64(bitrate),
                )?;
                set_comp_property(comp.0, "Av1FrameSize", AmfVariantStruct::from_size(w, h))?;
                set_comp_property(
                    comp.0,
                    "Av1FrameRate",
                    AmfVariantStruct::from_rate(config.fps, 1),
                )?;
            }
            AmfEncodeCodec::Av110Bit => {
                set_comp_property(comp.0, "Av1Usage", AmfVariantStruct::from_i64(0))?;
                set_comp_property(comp.0, "Av1Profile", AmfVariantStruct::from_i64(1))?;
                set_comp_property(comp.0, "Av1ColorBitDepth", AmfVariantStruct::from_i64(10))?;
                set_comp_property(comp.0, "Av1AlignmentMode", AmfVariantStruct::from_i64(3))?;
                set_comp_property(
                    comp.0,
                    "Av1QualityPreset",
                    AmfVariantStruct::from_i64(config.preset.av1_value()),
                )?;
                set_comp_property(
                    comp.0,
                    "Av1TargetBitrate",
                    AmfVariantStruct::from_i64(bitrate),
                )?;
                set_comp_property(comp.0, "Av1FrameSize", AmfVariantStruct::from_size(w, h))?;
                set_comp_property(
                    comp.0,
                    "Av1FrameRate",
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
        is_av1: bool,
        output_packets: &mut u32,
        total_encoded_bytes: &mut u64,
        valid_annex_b: &mut bool,
        valid_bitstream: &mut bool,
        mut captured_packets: Option<&mut Vec<Vec<u8>>>,
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
                    if size > 0 && !native.is_null() {
                        let packet_slice =
                            unsafe { std::slice::from_raw_parts(native.cast_const(), size) };
                        let prefix = &packet_slice[..packet_slice.len().min(64)];
                        if is_av1 {
                            if !*valid_bitstream && has_av1_obu_header(prefix) {
                                *valid_bitstream = true;
                            }
                        } else if size >= 4 && !*valid_annex_b && has_annex_b_start_code(prefix) {
                            *valid_annex_b = true;
                            *valid_bitstream = true;
                        }
                        if let Some(ref mut list) = captured_packets {
                            list.push(packet_slice.to_vec());
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
        is_av1: bool,
        mut captured_packets: Option<&mut Vec<Vec<u8>>>,
    ) -> ForgeResult<(u32, u64, bool, bool)> {
        let comp_vt = unsafe { &*(*comp).vtbl };
        let frame_duration_100ns = (10_000_000i64 / i64::from(fps.max(1))).max(1);
        let mut output_packets = 0u32;
        let mut total_encoded_bytes = 0u64;
        let mut valid_annex_b = false;
        let mut valid_bitstream = false;

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
                        is_av1,
                        &mut output_packets,
                        &mut total_encoded_bytes,
                        &mut valid_annex_b,
                        &mut valid_bitstream,
                        captured_packets.as_deref_mut(),
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
                is_av1,
                &mut output_packets,
                &mut total_encoded_bytes,
                &mut valid_annex_b,
                &mut valid_bitstream,
                captured_packets.as_deref_mut(),
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
                is_av1,
                &mut output_packets,
                &mut total_encoded_bytes,
                &mut valid_annex_b,
                &mut valid_bitstream,
                captured_packets.as_deref_mut(),
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

        Ok((
            output_packets,
            total_encoded_bytes,
            valid_annex_b,
            valid_bitstream,
        ))
    }

    /// Executes a hardware VCN video encode benchmark pass on pre-uploaded DX11 surfaces.
    pub fn run_amf_encode_benchmark(
        config: &AmfEncodeBenchConfig,
    ) -> ForgeResult<AmfEncodeBenchResult> {
        config.validate()?;

        let _guard = AMF_PROBE_LOCK
            .lock()
            .map_err(|_| ForgeError::ExecutionFailed("AMF probe mutex poisoned".to_string()))?;

        let (lib, factory_ptr, ctx) = init_amf_dx11_context()?;
        let ring = alloc_dx11_surface_ring(
            ctx.0,
            config.codec.surface_format_code(),
            config.width,
            config.height,
        )?;
        let is_av1 = config.codec.is_av1();

        if config.warmup_frames > 0 {
            let warmup_comp = create_and_init_encoder(factory_ptr, ctx.0, config)?;
            let _ = execute_encode_pass(
                warmup_comp.0,
                &ring,
                config.warmup_frames,
                config.fps,
                is_av1,
                None,
            )?;
            drop(warmup_comp);
        }

        let measured_comp = create_and_init_encoder(factory_ptr, ctx.0, config)?;
        let start = Instant::now();
        let (output_packets, total_encoded_bytes, valid_annex_b, valid_bitstream) =
            execute_encode_pass(
                measured_comp.0,
                &ring,
                config.measured_frames,
                config.fps,
                is_av1,
                None,
            )?;
        let elapsed_sec = start.elapsed().as_secs_f64();

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
            valid_bitstream,
        })
    }

    fn create_and_init_decoder(
        factory_ptr: *mut AmfFactory,
        ctx_ptr: *mut AmfContext,
        codec: AmfDecodeCodec,
        width: u32,
        height: u32,
    ) -> ForgeResult<OwnedComponent> {
        let cid = utf16_null(codec.component_id());
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
            if is_unsupported_amf_code(create_rc) {
                return Err(ForgeError::UnsupportedCapability(format!(
                    "{} decode ({}) is unsupported on this AMD GPU (AMF code {create_rc})",
                    codec.codec_name(),
                    codec.component_id()
                )));
            }
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFFactory::CreateComponent({}) failed with code {create_rc}",
                codec.component_id()
            )));
        }
        let comp = OwnedComponent(comp_ptr);
        verify_component_hardware_caps(comp.0, codec.codec_name())?;

        // Verified in AMF VideoDecoderUVD.h: TimestampMode = AMF_TS_PRESENTATION (0)
        set_comp_property(comp.0, "TimestampMode", AmfVariantStruct::from_i64(0))?;

        let fmt_code = codec.surface_format_code();
        let init_rc = unsafe {
            ((*(*comp.0).vtbl).init)(comp.0.cast(), fmt_code, width as i32, height as i32)
        };
        if init_rc != AMF_OK {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMFComponent::Init({}, {}x{}, fmt={}) failed with code {init_rc}",
                codec.component_id(),
                width,
                height,
                codec.surface_format_name()
            )));
        }
        Ok(comp)
    }

    fn drain_decoded_surfaces(
        comp: *mut AmfComponent,
        decoded_frames: &mut u32,
        valid_surfaces: &mut bool,
    ) -> ForgeResult<bool> {
        let comp_vt = unsafe { &*(*comp).vtbl };
        loop {
            let mut data_ptr: *mut AmfData = ptr::null_mut();
            let q_rc = unsafe { (comp_vt.query_output)(comp.cast(), &mut data_ptr) };
            if q_rc == AMF_OK && !data_ptr.is_null() {
                let data = OwnedData(data_ptr);
                let dtype = unsafe { ((*(*data.0).vtbl).get_data_type)(data.0.cast()) };
                *decoded_frames += 1;
                if !*valid_surfaces && dtype == AMF_DATA_SURFACE {
                    let mut surf_void: *mut c_void = ptr::null_mut();
                    let qi_rc = unsafe {
                        ((*(*data.0).vtbl).query_interface)(
                            data.0.cast(),
                            &AMF_SURFACE_IID,
                            &mut surf_void,
                        )
                    };
                    if qi_rc == AMF_OK && !surf_void.is_null() {
                        let surf = OwnedSurface(surf_void.cast());
                        let planes = unsafe { ((*(*surf.0).vtbl).get_planes_count)(surf.0.cast()) };
                        let _fmt = unsafe { ((*(*surf.0).vtbl).get_format)(surf.0.cast()) };
                        if planes >= 2 {
                            *valid_surfaces = true;
                        }
                    }
                }
            } else if q_rc == AMF_EOF {
                return Ok(true);
            } else if matches!(
                q_rc,
                AMF_OK | AMF_REPEAT | AMF_NEED_MORE_INPUT | AMF_RESOLUTION_UPDATED
            ) {
                return Ok(false);
            } else {
                return Err(ForgeError::ExecutionFailed(format!(
                    "AMF decoder QueryOutput failed with code {q_rc}"
                )));
            }
        }
    }

    fn execute_decode_pass(
        ctx_ptr: *mut AmfContext,
        comp: *mut AmfComponent,
        gop_packets: &[Vec<u8>],
        submitted_packets: u32,
        fps: u32,
    ) -> ForgeResult<(u32, bool)> {
        let comp_vt = unsafe { &*(*comp).vtbl };
        let ctx_vt = unsafe { &*(*ctx_ptr).vtbl };
        let frame_duration_100ns = (10_000_000i64 / i64::from(fps.max(1))).max(1);
        let mut decoded_frames = 0u32;
        let mut valid_surfaces = false;

        for idx in 0..(submitted_packets as usize) {
            let pkt = &gop_packets[idx % gop_packets.len()];
            let mut buf_ptr: *mut AmfBuffer = ptr::null_mut();
            let alloc_rc = unsafe {
                (ctx_vt.alloc_buffer)(ctx_ptr.cast(), AMF_MEMORY_HOST, pkt.len(), &mut buf_ptr)
            };
            if alloc_rc != AMF_OK || buf_ptr.is_null() {
                return Err(ForgeError::ExecutionFailed(format!(
                    "AMFContext::AllocBuffer(HOST, {} bytes) failed with code {alloc_rc}",
                    pkt.len()
                )));
            }
            let buf = OwnedBuffer(buf_ptr);
            unsafe {
                let buf_vt = &*(*buf.0).vtbl;
                let native = (buf_vt.get_native)(buf.0.cast());
                if native.is_null() {
                    return Err(ForgeError::ExecutionFailed(
                        "AMFBuffer::GetNative returned NULL pointer".to_string(),
                    ));
                }
                ptr::copy_nonoverlapping(pkt.as_ptr(), native, pkt.len());
                let pts = (idx as i64) * frame_duration_100ns;
                (buf_vt.data.set_pts)(buf.0.cast(), pts);
                (buf_vt.data.set_duration)(buf.0.cast(), frame_duration_100ns);
            }

            let mut retries = 0u32;
            loop {
                let sub_rc = unsafe { (comp_vt.submit_input)(comp.cast(), buf.0.cast()) };
                if sub_rc == AMF_OK || sub_rc == AMF_NEED_MORE_INPUT {
                    break;
                } else if matches!(
                    sub_rc,
                    AMF_INPUT_FULL | AMF_REPEAT | AMF_DECODER_NO_FREE_SURFACES
                ) {
                    drain_decoded_surfaces(comp, &mut decoded_frames, &mut valid_surfaces)?;
                    retries += 1;
                    if retries > 20_000 {
                        return Err(ForgeError::ExecutionFailed(
                            "AMF decoder SubmitInput timed out waiting for free DPB/input slot"
                                .to_string(),
                        ));
                    }
                    std::thread::sleep(Duration::from_micros(50));
                } else {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "AMF decoder SubmitInput(packet={idx}) failed with code {sub_rc}"
                    )));
                }
            }

            drain_decoded_surfaces(comp, &mut decoded_frames, &mut valid_surfaces)?;
        }

        let drain_rc = unsafe { (comp_vt.drain)(comp.cast()) };
        if drain_rc != AMF_OK {
            return Err(ForgeError::ExecutionFailed(format!(
                "AMF decoder Drain failed with code {drain_rc}"
            )));
        }

        let mut drain_polls = 0u32;
        loop {
            let eof = drain_decoded_surfaces(comp, &mut decoded_frames, &mut valid_surfaces)?;
            if eof {
                break;
            }
            drain_polls += 1;
            if drain_polls > 40_000 {
                return Err(ForgeError::ExecutionFailed(
                    "AMF decoder QueryOutput timed out waiting for AMF_EOF after Drain".to_string(),
                ));
            }
            std::thread::sleep(Duration::from_micros(50));
        }

        Ok((decoded_frames, valid_surfaces))
    }

    /// Synthesizes a closed-GOP Annex-B bitstream in memory via the GPU's VCN encoder and
    /// benchmarks hardware VCN DX11 decoding without requiring external sample video files.
    pub fn run_amf_decode_benchmark(
        config: &AmfDecodeBenchConfig,
    ) -> ForgeResult<AmfDecodeBenchResult> {
        config.validate()?;

        let _guard = AMF_PROBE_LOCK
            .lock()
            .map_err(|_| ForgeError::ExecutionFailed("AMF probe mutex poisoned".to_string()))?;

        let (lib, factory_ptr, ctx) = init_amf_dx11_context()?;

        // Check hardware decoder availability before encoding synthetic input packets.
        let probe_dec = create_and_init_decoder(
            factory_ptr,
            ctx.0,
            config.codec,
            config.width,
            config.height,
        )?;
        drop(probe_dec);

        // Synthesize a closed GOP (IDR frame at packet 0 + predicted frames) in memory.
        let enc_cfg = AmfEncodeBenchConfig {
            codec: config.codec.encode_source_codec(),
            width: config.width,
            height: config.height,
            fps: config.fps,
            target_bitrate_bps: config.source_bitrate_bps,
            preset: super::AmfQualityPreset::Speed,
            warmup_frames: 0,
            measured_frames: DECODE_SOURCE_GOP_FRAMES,
        };
        let ring = alloc_dx11_surface_ring(
            ctx.0,
            enc_cfg.codec.surface_format_code(),
            enc_cfg.width,
            enc_cfg.height,
        )?;
        let synth_enc = create_and_init_encoder(factory_ptr, ctx.0, &enc_cfg)?;
        let mut gop_packets: Vec<Vec<u8>> = Vec::with_capacity(DECODE_SOURCE_GOP_FRAMES as usize);
        let (_, total_bytes, valid_annex_b, _) = execute_encode_pass(
            synth_enc.0,
            &ring,
            DECODE_SOURCE_GOP_FRAMES,
            enc_cfg.fps,
            false,
            Some(&mut gop_packets),
        )?;
        drop(synth_enc);
        drop(ring);

        if gop_packets.is_empty() || !valid_annex_b {
            return Err(ForgeError::ExecutionFailed(format!(
                "synthesized {} source stream did not produce valid Annex-B GOP packets",
                config.codec.codec_name()
            )));
        }

        if config.warmup_frames > 0 {
            let warmup_dec = create_and_init_decoder(
                factory_ptr,
                ctx.0,
                config.codec,
                config.width,
                config.height,
            )?;
            let _ = execute_decode_pass(
                ctx.0,
                warmup_dec.0,
                &gop_packets,
                config.warmup_frames,
                config.fps,
            )?;
            drop(warmup_dec);
        }

        let measured_dec = create_and_init_decoder(
            factory_ptr,
            ctx.0,
            config.codec,
            config.width,
            config.height,
        )?;
        let start = Instant::now();
        let (decoded_frames, valid_surfaces) = execute_decode_pass(
            ctx.0,
            measured_dec.0,
            &gop_packets,
            config.measured_frames,
            config.fps,
        )?;
        let elapsed_sec = start.elapsed().as_secs_f64();

        drop(measured_dec);
        drop(ctx);
        drop(lib);

        let elapsed_ms = elapsed_sec * 1_000.0;
        let frames_f64 = f64::from(decoded_frames);
        let decode_fps = if elapsed_sec > 0.0 {
            frames_f64 / elapsed_sec
        } else {
            0.0
        };
        let megapixels_per_sec =
            decode_fps * (f64::from(config.width) * f64::from(config.height)) / 1_000_000.0;
        let avg_frame_ms = if decoded_frames > 0 {
            elapsed_ms / frames_f64
        } else {
            0.0
        };

        Ok(AmfDecodeBenchResult {
            codec: config.codec.codec_name().to_string(),
            component_id: config.codec.component_id().to_string(),
            surface_format: config.codec.surface_format_name().to_string(),
            width: config.width,
            height: config.height,
            gop_packets: gop_packets.len() as u32,
            source_stream_bytes: total_bytes,
            warmup_frames: config.warmup_frames,
            submitted_packets: config.measured_frames,
            decoded_frames,
            elapsed_ms,
            decode_fps,
            megapixels_per_sec,
            avg_frame_ms,
            valid_surfaces,
        })
    }
}

#[cfg(not(windows))]
mod stub {
    use forge_core::{ForgeError, ForgeResult};

    use super::{
        AmfCapabilitiesReport, AmfDecodeBenchConfig, AmfDecodeBenchResult, AmfEncodeBenchConfig,
        AmfEncodeBenchResult,
    };

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

    pub fn run_amf_decode_benchmark(
        config: &AmfDecodeBenchConfig,
    ) -> ForgeResult<AmfDecodeBenchResult> {
        config.validate()?;
        Err(ForgeError::RuntimeUnavailable {
            runtime: "AMF".to_string(),
            reason: "AMF hardware video decoding is Windows-only".to_string(),
        })
    }
}

#[cfg(windows)]
pub use imp::{probe_amf_capabilities, run_amf_decode_benchmark, run_amf_encode_benchmark};

#[cfg(not(windows))]
pub use stub::{probe_amf_capabilities, run_amf_decode_benchmark, run_amf_encode_benchmark};
