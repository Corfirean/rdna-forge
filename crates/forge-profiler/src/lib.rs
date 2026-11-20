pub mod adlx_ffi;
#[cfg(windows)]
mod adlx_tuning;
pub mod sampler;
pub mod tuning;
pub mod types;
pub mod vram_processes;

pub use adlx_ffi::AdlxSession;
pub use sampler::{
    read_all_gpu_tuning, reset_gpu_tuning_to_factory, set_gpu_power_limit, snapshot_all_gpus,
    snapshot_for_device, TelemetryGuard,
};
pub use tuning::{
    FanCurvePoint, FanTuningInfo, GfxTuningInfo, GpuTuningReport, IntRange, MemoryTimingMode,
    PowerLimitChangeResult, PowerTuningInfo, PresetTuningInfo, TuningCapabilities,
    TuningResetResult, VramTuningInfo,
};
pub use types::{
    check_competing_workload_warning, check_vram_pressure_warning, format_sample_csv_row,
    format_sample_table_row, parse_pci_device_id, summarize_samples, GpuTelemetrySample,
    MetricSummary, MonitorRecord, TelemetrySummary, COMPETING_GPU_USAGE_PCT, TELEMETRY_CSV_HEADER,
    TELEMETRY_TABLE_HEADER, WDDM_VRAM_PRESSURE_PCT,
};
pub use vram_processes::{
    aggregate_process_vram_samples, parse_gpu_process_instance_name, query_top_vram_processes,
    ProcessVramUsage,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with(
        usage: f64,
        clock: i32,
        vram_clock: i32,
        temp: f64,
        hotspot: f64,
        power: f64,
        vram_used: i32,
    ) -> GpuTelemetrySample {
        GpuTelemetrySample {
            gpu_name: "AMD Radeon RX 6900 XT".to_string(),
            pci_device_id: Some(0x73AF),
            discrete: true,
            total_vram_mib: Some(16368),
            timestamp_ms: Some(1000),
            gpu_usage_pct: Some(usage),
            gpu_clock_mhz: Some(clock),
            vram_clock_mhz: Some(vram_clock),
            gpu_temp_c: Some(temp),
            hotspot_temp_c: Some(hotspot),
            gpu_power_w: Some(power),
            board_power_w: None,
            fan_rpm: Some(1500),
            vram_used_mib: Some(vram_used),
            voltage_mv: Some(1050),
        }
    }

    #[test]
    fn pci_device_id_parser_handles_hex_prefixes() {
        assert_eq!(parse_pci_device_id("73AF"), Some(0x73AF));
        assert_eq!(parse_pci_device_id("0x73af"), Some(0x73AF));
        assert_eq!(parse_pci_device_id(" 0X164E "), Some(0x164E));
        assert_eq!(parse_pci_device_id(""), None);
        assert_eq!(parse_pci_device_id("zzzz"), None);
    }

    #[test]
    fn summarize_samples_computes_min_median_max_and_handles_empty() {
        assert!(summarize_samples(&[], 100, None).is_none());

        let samples = vec![
            sample_with(10.0, 1800, 1986, 50.0, 60.0, 120.0, 2000),
            sample_with(80.0, 2200, 1990, 58.0, 72.0, 255.0, 4000),
            sample_with(60.0, 2000, 1988, 54.0, 66.0, 210.0, 3000),
        ];

        let summary = summarize_samples(&samples, 100, Some(&samples[0])).unwrap();
        assert_eq!(summary.sample_count, 3);
        assert_eq!(summary.pci_device_id, Some(0x73AF));
        let clk = summary.gpu_clock_mhz.as_ref().unwrap();
        assert!((clk.min - 1800.0).abs() < 1e-6);
        assert!((clk.median - 2000.0).abs() < 1e-6);
        assert!((clk.max - 2200.0).abs() < 1e-6);
        assert!(summary.board_power_w.is_none());
        assert!(summary.warnings.is_empty());
    }

    #[test]
    fn summarize_samples_flags_wddm_vram_pressure_and_competing_load() {
        let pre = sample_with(45.0, 2100, 1986, 62.0, 75.0, 190.0, 13000);
        let samples = vec![
            pre.clone(),
            sample_with(95.0, 2250, 1990, 65.0, 80.0, 260.0, 14330),
        ];

        let summary = summarize_samples(&samples, 100, Some(&pre)).unwrap();
        assert_eq!(summary.warnings.len(), 2);
        assert!(summary.warnings[0].contains("Competing GPU workload"));
        assert!(summary.warnings[1].contains("WDDM VRAM occupancy"));
    }

    #[test]
    fn int_range_validation_enforces_bounds_step_and_validity() {
        assert!(IntRange::from_adlx(0, 0, 0).is_none());
        assert!(IntRange::from_adlx(10, 5, 1).is_none());

        let power_range = IntRange::from_adlx(-10, 12, 1).expect("valid power range");
        assert!(power_range.validate(-10, "power_limit").is_ok());
        assert!(power_range.validate(0, "power_limit").is_ok());
        assert!(power_range.validate(12, "power_limit").is_ok());
        assert!(matches!(
            power_range.validate(-11, "power_limit"),
            Err(forge_core::ForgeError::InvalidParameter(_))
        ));
        assert!(matches!(
            power_range.validate(13, "power_limit"),
            Err(forge_core::ForgeError::InvalidParameter(_))
        ));

        let vram_range = IntRange::from_adlx(2000, 2624, 2).expect("valid step-2 range");
        assert!(vram_range.validate(2000, "vram_freq").is_ok());
        assert!(vram_range.validate(2002, "vram_freq").is_ok());
        assert!(vram_range.validate(2624, "vram_freq").is_ok());
        assert!(matches!(
            vram_range.validate(2001, "vram_freq"),
            Err(forge_core::ForgeError::InvalidParameter(_))
        ));
        assert!(matches!(
            vram_range.validate(2626, "vram_freq"),
            Err(forge_core::ForgeError::InvalidParameter(_))
        ));

        let broken = IntRange {
            min: 5,
            max: 1,
            step: 0,
        };
        assert!(matches!(
            broken.validate(3, "broken"),
            Err(forge_core::ForgeError::InvalidParameter(_))
        ));
    }

    #[test]
    fn memory_timing_mode_maps_all_adlx_variants() {
        assert_eq!(MemoryTimingMode::from_adlx(0), MemoryTimingMode::Default);
        assert_eq!(MemoryTimingMode::from_adlx(1), MemoryTimingMode::FastTiming);
        assert_eq!(
            MemoryTimingMode::from_adlx(2),
            MemoryTimingMode::FastTimingLevel2
        );
        assert_eq!(MemoryTimingMode::from_adlx(3), MemoryTimingMode::Automatic);
        assert_eq!(MemoryTimingMode::from_adlx(4), MemoryTimingMode::Level1);
        assert_eq!(MemoryTimingMode::from_adlx(5), MemoryTimingMode::Level2);
        assert_eq!(
            MemoryTimingMode::from_adlx(99),
            MemoryTimingMode::Unknown(99)
        );
        assert_eq!(MemoryTimingMode::FastTiming.to_string(), "Fast Timing");
    }

    #[test]
    fn telemetry_csv_and_table_formatting_handles_quoting_and_none_fields() {
        let mut s = sample_with(42.5, 2150, 1996, 61.0, 74.5, 240.2, 4096);
        s.gpu_name = "AMD \"Radeon\", RX 6900 XT".to_string();
        s.timestamp_ms = None;

        let csv_row = format_sample_csv_row(1, 500, &s);
        assert_eq!(TELEMETRY_CSV_HEADER.split(',').count(), 16);
        assert!(csv_row.starts_with("1,500,,\"AMD \"\"Radeon\"\", RX 6900 XT\",0x73AF,"));
        // Verify empty field for `board_power_w: None` between `240.2` and `1500`
        assert!(csv_row.contains("240.2,,1500,4096,16368,1050"));

        let table_row = format_sample_table_row(1, 0.5, &s);
        assert_eq!(table_row.len(), TELEMETRY_TABLE_HEADER.len());
        assert!(table_row.contains("42.5%"));
        assert!(table_row.contains("4096/16368"));
    }

    #[test]
    fn gpu_tuning_report_json_roundtrip() {
        let report = GpuTuningReport {
            gpu_name: "AMD Radeon RX 6900 XT".to_string(),
            pci_device_id: Some(0x73AF),
            discrete: true,
            at_factory: Some(false),
            capabilities: TuningCapabilities {
                auto_tuning: true,
                preset_tuning: true,
                manual_gfx: true,
                manual_vram: true,
                manual_fan: true,
                manual_power: true,
            },
            preset: Some(PresetTuningInfo {
                supported_presets: vec!["Quiet".into(), "Balanced".into(), "Rage".into()],
                current_preset: Some("Balanced".into()),
            }),
            power: Some(PowerTuningInfo {
                power_limit_pct: Some(0),
                power_limit_range: IntRange::from_adlx(-10, 12, 1),
                power_limit_default_pct: Some(0),
                tdc_limit_supported: false,
                tdc_limit_a: None,
                tdc_limit_range: None,
                tdc_limit_default_a: None,
            }),
            gfx: Some(GfxTuningInfo {
                min_freq_mhz: Some(500),
                min_freq_range: IntRange::from_adlx(500, 4000, 1),
                min_freq_default_mhz: Some(500),
                max_freq_mhz: Some(2564),
                max_freq_range: IntRange::from_adlx(500, 4000, 1),
                max_freq_default_mhz: Some(2564),
                voltage_mv: Some(1200),
                voltage_range: IntRange::from_adlx(900, 1200, 1),
                voltage_default_mv: Some(1200),
            }),
            vram: Some(VramTuningInfo {
                max_freq_mhz: Some(2000),
                max_freq_range: IntRange::from_adlx(2000, 2624, 2),
                max_freq_default_mhz: Some(2000),
                memory_timing_supported: true,
                memory_timing: Some(MemoryTimingMode::Default),
                supported_memory_timings: vec![
                    MemoryTimingMode::Default,
                    MemoryTimingMode::FastTiming,
                ],
            }),
            fan: Some(FanTuningInfo {
                speed_range_pct: IntRange::from_adlx(20, 100, 1),
                temp_range_c: IntRange::from_adlx(25, 100, 1),
                zero_rpm_supported: true,
                zero_rpm_enabled: Some(false),
                zero_rpm_default: Some(true),
                min_acoustic_limit_mhz: None,
                min_acoustic_limit_range: None,
                min_acoustic_limit_default_mhz: None,
                min_fan_speed_pct: None,
                min_fan_speed_range: None,
                min_fan_speed_default_pct: None,
                target_fan_speed_pct: None,
                target_fan_speed_range: None,
                target_fan_speed_default_pct: None,
                fan_curve: vec![FanCurvePoint {
                    temp_c: 50,
                    speed_pct: 20,
                }],
                default_fan_curve: vec![FanCurvePoint {
                    temp_c: 50,
                    speed_pct: 35,
                }],
                flat_pwm_override_pct: None,
                external_controller: None,
                fan_note: None,
            }),
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize report");
        let parsed: GpuTuningReport = serde_json::from_str(&json).expect("deserialize report");
        assert_eq!(parsed, report);
    }
}
