use std::time::Duration;

use forge_core::{ForgeError, ForgeResult};
use forge_profiler::{TelemetryGuard, TelemetrySummary};
use forge_runtime::{
    probe_amf_capabilities, run_amf_decode_benchmark, run_amf_encode_benchmark, AmfAcceleration,
    AmfCapabilitiesReport, AmfCodecDirection, AmfDecodeBenchConfig, AmfDecodeBenchResult,
    AmfDecodeCodec, AmfEncodeBenchConfig, AmfEncodeBenchResult, AmfEncodeCodec, AmfQualityPreset,
};
use serde::{Deserialize, Serialize};

use crate::memory::peak_process_rss_bytes;

/// Suite-level configuration for AMD AMF VCN hardware video encode benchmarks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoBenchSuiteConfig {
    pub codecs: Vec<AmfEncodeCodec>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub target_bitrate_bps: u64,
    pub preset: AmfQualityPreset,
    pub warmup_frames: u32,
    pub measured_frames: u32,
    #[serde(default)]
    pub skip_unsupported: bool,
}

impl Default for VideoBenchSuiteConfig {
    fn default() -> Self {
        Self {
            codecs: vec![
                AmfEncodeCodec::H264Avc,
                AmfEncodeCodec::Hevc,
                AmfEncodeCodec::Hevc10Bit,
                AmfEncodeCodec::Av1,
            ],
            width: 1920,
            height: 1080,
            fps: 60,
            target_bitrate_bps: 12_000_000,
            preset: AmfQualityPreset::Speed,
            warmup_frames: 15,
            measured_frames: 120,
            skip_unsupported: true,
        }
    }
}

/// Complete report produced by the AMD AMF VCN hardware video encode benchmark suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoBenchReport {
    pub runtime_version: String,
    pub results: Vec<AmfEncodeBenchResult>,
    #[serde(default)]
    pub skipped_codecs: Vec<String>,
    pub peak_rss_bytes: Option<u64>,
    #[serde(default)]
    pub telemetry: Option<TelemetrySummary>,
}

/// Suite-level configuration for AMD AMF VCN hardware video decode benchmarks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoDecodeBenchSuiteConfig {
    pub codecs: Vec<AmfDecodeCodec>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub source_bitrate_bps: u64,
    pub warmup_frames: u32,
    pub measured_frames: u32,
}

impl Default for VideoDecodeBenchSuiteConfig {
    fn default() -> Self {
        Self {
            codecs: vec![
                AmfDecodeCodec::H264Avc,
                AmfDecodeCodec::Hevc,
                AmfDecodeCodec::Hevc10Bit,
            ],
            width: 1920,
            height: 1080,
            fps: 60,
            source_bitrate_bps: 12_000_000,
            warmup_frames: 15,
            measured_frames: 120,
        }
    }
}

/// Complete report produced by the AMD AMF VCN hardware video decode benchmark suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoDecodeBenchReport {
    pub runtime_version: String,
    pub results: Vec<AmfDecodeBenchResult>,
    pub peak_rss_bytes: Option<u64>,
    #[serde(default)]
    pub telemetry: Option<TelemetrySummary>,
}

fn codec_hardware_supported(
    caps: &AmfCapabilitiesReport,
    component_id: &str,
    dir: AmfCodecDirection,
) -> bool {
    caps.codecs
        .iter()
        .find(|c| c.component_id == component_id && c.direction == dir)
        .is_none_or(|c| c.acceleration != AmfAcceleration::NotSupported)
}

/// Executes the configured AMD AMF VCN hardware video encode benchmark suite with live ADLX telemetry.
pub fn run_video_benchmark_suite(config: &VideoBenchSuiteConfig) -> ForgeResult<VideoBenchReport> {
    if config.codecs.is_empty() {
        return Err(ForgeError::InvalidDimension(
            "video benchmark suite requires at least one codec".to_string(),
        ));
    }

    for &codec in &config.codecs {
        AmfEncodeBenchConfig {
            codec,
            width: config.width,
            height: config.height,
            fps: config.fps,
            target_bitrate_bps: config.target_bitrate_bps,
            preset: config.preset,
            warmup_frames: config.warmup_frames,
            measured_frames: config.measured_frames,
        }
        .validate()?;
    }

    let caps = probe_amf_capabilities()?;
    let guard = TelemetryGuard::start(None, Duration::from_millis(200));

    let mut results = Vec::with_capacity(config.codecs.len());
    let mut skipped_codecs = Vec::new();
    for &codec in &config.codecs {
        if !codec_hardware_supported(&caps, codec.component_id(), AmfCodecDirection::Encode) {
            if config.skip_unsupported {
                skipped_codecs.push(format!(
                    "{} ({} unsupported on this VCN engine)",
                    codec.codec_name(),
                    codec.component_id()
                ));
                continue;
            }
            return Err(ForgeError::UnsupportedCapability(format!(
                "{} hardware encode ({}) is not supported on this GPU's VCN engine",
                codec.codec_name(),
                codec.component_id()
            )));
        }

        let pass_cfg = AmfEncodeBenchConfig {
            codec,
            width: config.width,
            height: config.height,
            fps: config.fps,
            target_bitrate_bps: config.target_bitrate_bps,
            preset: config.preset,
            warmup_frames: config.warmup_frames,
            measured_frames: config.measured_frames,
        };
        match run_amf_encode_benchmark(&pass_cfg) {
            Ok(res) => results.push(res),
            Err(ForgeError::UnsupportedCapability(msg)) if config.skip_unsupported => {
                skipped_codecs.push(msg);
            }
            Err(e) => return Err(e),
        }
    }

    let telemetry = guard.and_then(|g| g.stop_and_summarize());
    let peak_rss_bytes = peak_process_rss_bytes();

    if results.is_empty() && !skipped_codecs.is_empty() {
        return Err(ForgeError::UnsupportedCapability(skipped_codecs.join("; ")));
    }

    Ok(VideoBenchReport {
        runtime_version: caps.runtime_version,
        results,
        skipped_codecs,
        peak_rss_bytes,
        telemetry,
    })
}

/// Executes the configured AMD AMF VCN hardware video decode benchmark suite using in-memory
/// closed-GOP streams and live ADLX telemetry.
pub fn run_video_decode_benchmark_suite(
    config: &VideoDecodeBenchSuiteConfig,
) -> ForgeResult<VideoDecodeBenchReport> {
    if config.codecs.is_empty() {
        return Err(ForgeError::InvalidDimension(
            "video decode benchmark suite requires at least one codec".to_string(),
        ));
    }

    for &codec in &config.codecs {
        AmfDecodeBenchConfig {
            codec,
            width: config.width,
            height: config.height,
            fps: config.fps,
            source_bitrate_bps: config.source_bitrate_bps,
            warmup_frames: config.warmup_frames,
            measured_frames: config.measured_frames,
        }
        .validate()?;
    }

    let caps = probe_amf_capabilities()?;
    let guard = TelemetryGuard::start(None, Duration::from_millis(200));

    let mut results = Vec::with_capacity(config.codecs.len());
    for &codec in &config.codecs {
        let pass_cfg = AmfDecodeBenchConfig {
            codec,
            width: config.width,
            height: config.height,
            fps: config.fps,
            source_bitrate_bps: config.source_bitrate_bps,
            warmup_frames: config.warmup_frames,
            measured_frames: config.measured_frames,
        };
        results.push(run_amf_decode_benchmark(&pass_cfg)?);
    }

    let telemetry = guard.and_then(|g| g.stop_and_summarize());
    let peak_rss_bytes = peak_process_rss_bytes();

    Ok(VideoDecodeBenchReport {
        runtime_version: caps.runtime_version,
        results,
        peak_rss_bytes,
        telemetry,
    })
}
