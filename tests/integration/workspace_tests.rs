use std::path::PathBuf;
use std::process::Command;

use forge_benchmark::{compute_latency_stats, parse_llama_bench_json};
use forge_core::{
    validate_kernel_execution, ForgeError, GfxTarget, GpuArchitecture, GpuDeviceInfo,
    HardwareProfile, KernelRequirements, AMD_PCI_VENDOR_ID,
};
use forge_profiler::{summarize_samples, GpuTelemetrySample};
use forge_runtime::{
    compile_wgsl_to_spirv, compile_wgsl_to_spirv_dp4a, compile_wgsl_to_spirv_subgroup,
    ref_gemm_f32, spirv_contains_opcode, FLASH_ATTENTION_F32_WAVE32_WGSL, GEMM_F32_TILED_VEC4_WGSL,
    GEMM_F32_WGSL, GEMM_I8_DP4A_WGSL, GEMV_Q4_0_WAVE32_WGSL, GEMV_Q4_K_WAVE32_WGSL,
    GEMV_W8A32_WAVE32_WGSL, RMSNORM_F32_WGSL, RMSNORM_ROPE_F32_SUBGROUP_WGSL, ROPE_F32_WGSL,
    SPIRV_OP_GROUP_NON_UNIFORM_FADD, SPIRV_OP_SDOT,
};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is apps/forge-cli
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root exists")
        .to_path_buf()
}

#[test]
fn rx6900xt_profile_loads_and_validates() {
    let profile_path = repo_root().join("profiles/gfx1030/rx6900xt.toml");
    let profile = HardwareProfile::from_file(&profile_path).expect("profile must load");
    let target = GfxTarget::parse(&profile.device.gfx_target).unwrap();
    assert_eq!(target.base, "gfx1030");
    assert_eq!(target.architecture, GpuArchitecture::Rdna2);
    assert_eq!(profile.compute.compute_units, 80);
    assert_eq!(profile.compute.wgp_count, 40);
    assert!(!profile.capabilities.wmma);
    assert!(!profile.capabilities.mfma);
}

#[test]
fn wgsl_compute_shaders_compile_offline_to_spirv() {
    for src in [
        GEMM_F32_WGSL,
        GEMM_F32_TILED_VEC4_WGSL,
        RMSNORM_F32_WGSL,
        ROPE_F32_WGSL,
    ] {
        let words = compile_wgsl_to_spirv(src).expect("shader must compile to SPIR-V");
        assert_eq!(words[0], 0x0723_0203);
    }

    let dp4a = compile_wgsl_to_spirv_dp4a(GEMM_I8_DP4A_WGSL)
        .expect("DP4A shader must compile to SPIR-V 1.6");
    assert_eq!(dp4a[0], 0x0723_0203);
    assert!(
        spirv_contains_opcode(&dp4a, SPIRV_OP_SDOT),
        "DP4A shader must emit native OpSDot (4450)"
    );

    for src in [
        GEMV_W8A32_WAVE32_WGSL,
        GEMV_Q4_0_WAVE32_WGSL,
        GEMV_Q4_K_WAVE32_WGSL,
        RMSNORM_ROPE_F32_SUBGROUP_WGSL,
        FLASH_ATTENTION_F32_WAVE32_WGSL,
    ] {
        let words =
            compile_wgsl_to_spirv_subgroup(src).expect("subgroup shader must compile to SPIR-V");
        assert_eq!(words[0], 0x0723_0203);
        assert!(
            spirv_contains_opcode(&words, SPIRV_OP_GROUP_NON_UNIFORM_FADD),
            "subgroup shader must emit OpGroupNonUniformFAdd (350)"
        );
    }
}

#[test]
fn profiler_summary_detects_wddm_vram_pressure_and_competing_load() {
    let pre = GpuTelemetrySample {
        gpu_name: "Synthetic RDNA2 GPU".to_string(),
        pci_device_id: Some(0x73AF),
        discrete: true,
        total_vram_mib: Some(16368),
        timestamp_ms: Some(100),
        gpu_usage_pct: Some(35.0),
        gpu_clock_mhz: Some(1200),
        vram_clock_mhz: Some(1986),
        gpu_temp_c: Some(45.0),
        hotspot_temp_c: Some(52.0),
        gpu_power_w: Some(80.0),
        board_power_w: None,
        fan_rpm: Some(1100),
        vram_used_mib: Some(14000),
        voltage_mv: Some(900),
    };
    let summary = summarize_samples(std::slice::from_ref(&pre), 250, Some(&pre)).unwrap();
    assert_eq!(summary.warnings.len(), 2);
    assert!(summary
        .warnings
        .iter()
        .any(|w| w.contains("WDDM VRAM occupancy")));
    assert!(summary
        .warnings
        .iter()
        .any(|w| w.contains("Competing GPU workload")));
}

#[test]
fn capability_gating_and_dimension_guards_enforce_limits() {
    let device = GpuDeviceInfo {
        index: 0,
        name: "Synthetic RDNA2 GPU".to_string(),
        vendor_id: AMD_PCI_VENDOR_ID,
        device_id: 0x73AF,
        pci_bus_id: None,
        architecture: GpuArchitecture::Rdna2,
        gfx_target: Some(GfxTarget::parse("gfx1030").unwrap()),
        discrete: true,
        vram_bytes: 16 * 1024 * 1024 * 1024,
        shared_memory_per_block_bytes: Some(65536),
        vulkan_shared_memory_bytes: Some(32768),
        compute_units: Some(80),
        wgp_count: Some(40),
        warp_size: Some(32),
        max_buffer_size_bytes: Some(2 * 1024 * 1024 * 1024),
        max_memory_allocation_bytes: Some(2 * 1024 * 1024 * 1024),
        rebar_enabled: Some(false),
        driver_version: None,
        adrenalin_version: None,
        vulkan_api_version: None,
        vulkan_driver_info: None,
        vulkan_cooperative_matrix: false,
        vulkan_fp16: true,
        vulkan_int8: true,
        vulkan_int8_dot_product: true,
        hip_available: true,
        hip_runtime_version: None,
        core_clock_mhz: None,
        memory_clock_mhz: None,
        memory_bus_width_bits: None,
        l2_cache_bytes: None,
    };

    assert!(matches!(
        validate_kernel_execution(
            &device,
            &KernelRequirements {
                require_wmma: true,
                ..Default::default()
            }
        ),
        Err(ForgeError::UnsupportedCapability(_))
    ));

    assert!(matches!(
        validate_kernel_execution(
            &device,
            &KernelRequirements {
                max_single_buffer_bytes: 4 * 1024 * 1024 * 1024,
                ..Default::default()
            }
        ),
        Err(ForgeError::BufferTooLarge { .. })
    ));

    assert!(matches!(
        ref_gemm_f32(&[1.0], &[1.0], 0, 1, 1),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
fn benchmark_statistics_and_llama_parser_work_end_to_end() {
    let stats = compute_latency_stats(5, &[10.0, 12.0, 11.0, 13.0, 14.0]).unwrap();
    assert_eq!(stats.measured_iters, 5);
    assert!((stats.median_ms - 12.0).abs() < 1e-9);
    assert!((stats.iqr_ms - 2.0).abs() < 1e-9);

    let json = r#"[{
        "build_commit": "deadbeef",
        "build_number": 12532,
        "gpu_info": "AMD Radeon RX 6900 XT",
        "backends": "Vulkan",
        "model_type": "test",
        "model_size": 1000,
        "n_gpu_layers": 99,
        "main_gpu": 0,
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ts": 1000.0,
        "stddev_ts": 5.0,
        "avg_ns": 512000000,
        "stddev_ns": 2500000,
        "samples_ns": [510000000, 512000000, 514000000],
        "samples_ts": [1003.9, 1000.0, 996.1]
    }]"#;

    let parsed = parse_llama_bench_json(json, "vulkan").unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].test_kind, "pp512");
    assert!((parsed[0].median_tok_s - 1000.0).abs() < 1e-6);
}

#[test]
fn forge_cli_help_and_profile_show_succeed() {
    let bin = env!("CARGO_BIN_EXE_forge");
    let help = Command::new(bin)
        .arg("--help")
        .output()
        .expect("forge binary must execute");
    assert!(help.status.success());
    let stdout = String::from_utf8_lossy(&help.stdout);
    assert!(stdout.contains("doctor"));
    assert!(stdout.contains("bench"));
    assert!(stdout.contains("monitor"));
    assert!(stdout.contains("tune"));
    assert!(stdout.contains("video"));
    assert!(stdout.contains("profile"));

    for args in [
        &["monitor", "--help"][..],
        &["tune", "--help"][..],
        &["tune", "power-limit", "--help"][..],
        &["video", "caps", "--help"][..],
        &["video", "bench", "--help"][..],
        &["llm", "--help"][..],
        &["llm", "plan", "--help"][..],
        &["llm", "autotune", "--help"][..],
        &["llm", "apply-lmstudio", "--help"][..],
        &["llm", "apply-env", "--help"][..],
    ] {
        let out = Command::new(bin)
            .args(args)
            .output()
            .expect("subcommand --help must execute");
        assert!(out.status.success(), "help failed for {args:?}");
    }

    let profile_path = repo_root().join("profiles/gfx1030/rx6900xt.toml");
    let prof_out = Command::new(bin)
        .arg("profile")
        .arg("show")
        .arg(&profile_path)
        .output()
        .expect("forge profile show must execute");
    assert!(prof_out.status.success());
    let prof_stdout = String::from_utf8_lossy(&prof_out.stdout);
    assert!(prof_stdout.contains("gfx1030"));
    assert!(prof_stdout.contains("CUs: 80 (40 WGPs)"));
}
