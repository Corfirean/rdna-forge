pub mod adlx_ffi;
pub mod sampler;
pub mod types;

pub use sampler::{snapshot_all_gpus, snapshot_for_device, TelemetryGuard};
pub use types::{
    check_competing_workload_warning, check_vram_pressure_warning, parse_pci_device_id,
    summarize_samples, GpuTelemetrySample, MetricSummary, TelemetrySummary,
    COMPETING_GPU_USAGE_PCT, WDDM_VRAM_PRESSURE_PCT,
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
}
