use std::time::Duration;

use forge_core::{ForgeError, ForgeResult};
use forge_profiler::{TelemetryGuard, TelemetrySummary};
use forge_runtime::{
    probe_amf_capabilities, run_amf_encode_benchmark, AmfEncodeBenchConfig, AmfEncodeBenchResult,
    AmfEncodeCodec, AmfQualityPreset,
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
}

impl Default for VideoBenchSuiteConfig {
    fn default() -> Self {
        Self {
            codecs: vec![
                AmfEncodeCodec::H264Avc,
                AmfEncodeCodec::Hevc,
                AmfEncodeCodec::Hevc10Bit,
            ],
            width: 1920,
            height: 1080,
            fps: 60,
            target_bitrate_bps: 12_000_000,
            preset: AmfQualityPreset::Speed,
            warmup_frames: 15,
            measured_frames: 120,
        }
    }
}

/// Complete report produced by the AMD AMF VCN hardware video encode benchmark suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoBenchReport {
    pub runtime_version: String,
    pub results: Vec<AmfEncodeBenchResult>,
    pub peak_rss_bytes: Option<u64>,
    #[serde(default)]
    pub telemetry: Option<TelemetrySummary>,
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
    for &codec in &config.codecs {
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
        results.push(run_amf_encode_benchmark(&pass_cfg)?);
    }

    let telemetry = guard.and_then(|g| g.stop_and_summarize());
    let peak_rss_bytes = peak_process_rss_bytes();

    Ok(VideoBenchReport {
        runtime_version: caps.runtime_version,
        results,
        peak_rss_bytes,
        telemetry,
    })
}
