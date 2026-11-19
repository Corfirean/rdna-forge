use std::time::Duration;

use forge_core::{validate_kernel_execution, ForgeError, KernelRequirements, AMD_PCI_VENDOR_ID};
use forge_device::discover_gpus;
use forge_profiler::{
    read_all_gpu_tuning, reset_gpu_tuning_to_factory, set_gpu_power_limit, snapshot_all_gpus,
    TelemetryGuard,
};
use forge_runtime::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    deterministic_q4_0_weights, deterministic_q4_k_weights, pack_i8_b_transposed_k4,
    pack_i8_row_major_k4, probe_amf_capabilities, ref_flash_attention_f32, ref_gemm_f32_exact_f64,
    ref_gemm_i8_to_i32, ref_gemv_q4_0_f32, ref_gemv_q4_k_f32, ref_gemv_w8a32_f32, ref_rmsnorm_f32,
    ref_rmsnorm_rope_f32, ref_rope_f32, verify_hip_transfer, AmfAcceleration, AmfCodecDirection,
    VulkanComputeEngine,
};

#[test]
#[ignore = "requires physical AMD GPU"]
fn gpu_discovery_confirms_discrete_amd_vulkan_and_hip() {
    let gpus = discover_gpus();
    assert!(!gpus.is_empty(), "expected at least one GPU");
    let primary = &gpus[0];
    assert_eq!(primary.vendor_id, AMD_PCI_VENDOR_ID);
    assert!(primary.discrete, "primary GPU should be discrete AMD GPU");
    assert!(primary.vram_bytes > 0);
    assert!(primary.vulkan_api_version.is_some());
    assert!(
        primary.hip_available,
        "expected HIP runtime DLL to detect GPU"
    );
    assert!(primary.gfx_target.is_some());
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn hip_memory_roundtrip_is_bit_exact() {
    let res = verify_hip_transfer(0, 4 * 1024 * 1024).expect("HIP transfer must succeed");
    assert!(res.h2d_d2h_bit_exact, "H2D+D2H transfer must be bit-exact");
    assert!(res.elapsed_ms > 0.0);
    assert!(res.bandwidth_gb_s > 0.0);
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_gemm_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");

    for (m, k, n) in [
        (1u32, 1u32, 1u32),
        (37, 53, 29),
        (129, 257, 65),
        (256, 256, 256),
    ] {
        let a = deterministic_f32_data((m * k) as usize, 101 + m as u64);
        let b = deterministic_f32_data((k * n) as usize, 202 + n as u64);
        let (actual, _timing) = engine
            .run_gemm_f32(&a, &b, m, k, n)
            .unwrap_or_else(|e| panic!("GEMM {m}x{k}x{n} failed: {e}"));
        let expected = ref_gemm_f32_exact_f64(&a, &b, m as usize, k as usize, n as usize).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-2, 1e-3).unwrap();
        assert!(
            acc.passed,
            "GEMM {m}x{k}x{n} accuracy failed: max_abs={:.3e}, rmse={:.3e}",
            acc.max_abs_err, acc.rmse
        );
    }

    assert!(matches!(
        engine.run_gemm_f32(&[], &[], 0, 4, 4),
        Err(ForgeError::InvalidDimension(_))
    ));

    let too_big_vram = KernelRequirements {
        required_vram_bytes: engine.device_info().vram_bytes.saturating_add(1024),
        ..Default::default()
    };
    assert!(matches!(
        validate_kernel_execution(engine.device_info(), &too_big_vram),
        Err(ForgeError::InsufficientVram { .. })
    ));

    if let Some(max_buf) = engine.device_info().max_buffer_size_bytes {
        let too_big_buf = KernelRequirements {
            max_single_buffer_bytes: max_buf.saturating_add(4),
            ..Default::default()
        };
        assert!(matches!(
            validate_kernel_execution(engine.device_info(), &too_big_buf),
            Err(ForgeError::BufferTooLarge { .. })
        ));
    }
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_rmsnorm_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    let eps = 1e-5_f32;

    for (rows, cols) in [(1u32, 1u32), (17, 259), (1024, 4096)] {
        let x = deterministic_f32_data((rows * cols) as usize, 303 + rows as u64);
        let w = deterministic_f32_data(cols as usize, 404 + cols as u64);
        let (actual, _timing) = engine
            .run_rmsnorm_f32(&x, &w, rows, cols, eps)
            .unwrap_or_else(|e| panic!("RMSNorm {rows}x{cols} failed: {e}"));
        let expected = ref_rmsnorm_f32(&x, &w, rows as usize, cols as usize, eps).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-4, 1e-4).unwrap();
        assert!(
            acc.passed,
            "RMSNorm {rows}x{cols} accuracy failed: max_abs={:.3e}",
            acc.max_abs_err
        );
    }

    assert!(matches!(
        engine.run_rmsnorm_f32(&[], &[1.0], 0, 1, eps),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_rope_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    let theta = 10_000.0_f32;

    for (seq, heads, dim) in [(1u32, 1u32, 2u32), (13, 5, 66), (512, 32, 128)] {
        let x = deterministic_f32_data((seq * heads * dim) as usize, 505 + seq as u64);
        let (actual, _timing) = engine
            .run_rope_f32(&x, seq, heads, dim, theta)
            .unwrap_or_else(|e| panic!("RoPE {seq}x{heads}x{dim} failed: {e}"));
        let expected = ref_rope_f32(&x, seq as usize, heads as usize, dim as usize, theta).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-4, 1e-4).unwrap();
        assert!(
            acc.passed,
            "RoPE {seq}x{heads}x{dim} accuracy failed: max_abs={:.3e}",
            acc.max_abs_err
        );
    }

    assert!(matches!(
        engine.run_rope_f32(&[1.0, 2.0, 3.0], 1, 1, 3, theta),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_repeated_workload_stability() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");

    let n = 256u32;
    let a = deterministic_f32_data((n * n) as usize, 707);
    let b = deterministic_f32_data((n * n) as usize, 808);
    let expected_gemm = ref_gemm_f32_exact_f64(&a, &b, n as usize, n as usize, n as usize).unwrap();

    let (gemm_out, gemm_timings) = engine
        .bench_gemm_f32(&a, &b, n, n, n, 2, 50)
        .expect("50 repeated GEMM dispatches must succeed");
    assert_eq!(gemm_timings.len(), 50);
    let acc_g = compare_slices(&gemm_out, &expected_gemm, 1e-2, 1e-3).unwrap();
    assert!(acc_g.passed && acc_g.all_finite);

    let (rows, cols) = (512u32, 512u32);
    let x = deterministic_f32_data((rows * cols) as usize, 909);
    let w = deterministic_f32_data(cols as usize, 1010);
    let expected_rms = ref_rmsnorm_f32(&x, &w, rows as usize, cols as usize, 1e-5).unwrap();

    let (rms_out, rms_timings) = engine
        .bench_rmsnorm_f32(&x, &w, rows, cols, 1e-5, 2, 50)
        .expect("50 repeated RMSNorm dispatches must succeed");
    assert_eq!(rms_timings.len(), 50);
    let acc_r = compare_slices(&rms_out, &expected_rms, 1e-4, 1e-4).unwrap();
    assert!(acc_r.passed && acc_r.all_finite);
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn adlx_live_telemetry_reads_discrete_amd_gpu() {
    let samples = snapshot_all_gpus().expect("ADLX snapshot should succeed on AMD Windows host");
    let discrete = samples
        .iter()
        .find(|s| s.discrete)
        .expect("expected a discrete AMD GPU telemetry sample");
    assert!(discrete.total_vram_mib.unwrap_or(0) > 0);
    assert!(discrete.gpu_temp_c.is_some());
    assert!(discrete.gpu_power_w.is_some());

    let guard = TelemetryGuard::start(discrete.pci_device_id, Duration::from_millis(80))
        .expect("ADLX TelemetryGuard should initialize");
    std::thread::sleep(Duration::from_millis(250));
    let summary = guard
        .stop_and_summarize()
        .expect("TelemetryGuard should collect at least one sample");
    assert!(summary.sample_count >= 1);
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_gemm_tiled_vec4_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");

    for (m, k, n) in [
        (1u32, 1u32, 1u32),
        (37, 53, 29),
        (129, 257, 65),
        (256, 256, 256),
    ] {
        let a = deterministic_f32_data((m * k) as usize, 311 + m as u64);
        let b = deterministic_f32_data((k * n) as usize, 412 + n as u64);
        let (actual, _timing) = engine
            .run_gemm_f32_tiled_vec4(&a, &b, m, k, n)
            .unwrap_or_else(|e| panic!("Tiled vec4 GEMM {m}x{k}x{n} failed: {e}"));
        let expected = ref_gemm_f32_exact_f64(&a, &b, m as usize, k as usize, n as usize).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-2, 1e-3).unwrap();
        assert!(
            acc.passed,
            "Tiled vec4 GEMM {m}x{k}x{n} accuracy failed: max_abs={:.3e}, rmse={:.3e}",
            acc.max_abs_err, acc.rmse
        );
    }
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_gemm_i8_dp4a_exact_integer_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    assert!(
        engine.supports_int8_dp4a(),
        "expected Vulkan 1.3 shader_integer_dot_product on RX 6900 XT"
    );

    for (m, k, n) in [
        (1u32, 4u32, 1u32),
        (37, 52, 29),
        (129, 256, 65),
        (256, 256, 256),
    ] {
        let a = deterministic_i8_data((m * k) as usize, 501 + m as u64);
        let b = deterministic_i8_data((k * n) as usize, 602 + n as u64);
        let a_packed = pack_i8_row_major_k4(&a, m as usize, k as usize).unwrap();
        let b_t_packed = pack_i8_b_transposed_k4(&b, k as usize, n as usize).unwrap();
        let (actual, _timing) = engine
            .run_gemm_i8_dp4a(&a_packed, &b_t_packed, m, k / 4, n)
            .unwrap_or_else(|e| panic!("INT8 DP4A GEMM {m}x{k}x{n} failed: {e}"));
        let expected = ref_gemm_i8_to_i32(&a, &b, m as usize, k as usize, n as usize).unwrap();
        let acc = compare_i32_exact(&actual, &expected).unwrap();
        assert!(
            acc.passed && acc.max_abs_err == 0.0,
            "INT8 DP4A GEMM {m}x{k}x{n} must be bit-exact, got max_abs={}",
            acc.max_abs_err
        );
    }

    assert!(matches!(
        pack_i8_row_major_k4(&[1, 2, 3], 1, 3),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn adlx_live_tuning_read_only_inspects_discrete_amd_gpu() {
    let reports = read_all_gpu_tuning().expect("ADLX tuning read must succeed on AMD Windows host");
    let discrete = reports
        .iter()
        .find(|r| r.discrete)
        .expect("expected a discrete AMD GPU in ADLX tuning report");

    assert!(discrete.capabilities.manual_power);
    assert!(discrete.capabilities.manual_gfx);
    assert!(discrete.capabilities.manual_vram);
    assert!(discrete.capabilities.manual_fan);

    let pwr = discrete.power.as_ref().expect("power tuning info");
    let pwr_range = pwr.power_limit_range.expect("power limit range");
    assert!(pwr_range.is_valid());
    assert!(pwr_range.validate(0, "power_limit_pct").is_ok());

    let gfx = discrete.gfx.as_ref().expect("GFX tuning info");
    assert!(gfx.max_freq_range.is_some_and(|r| r.is_valid()));

    let vram = discrete.vram.as_ref().expect("VRAM tuning info");
    assert!(vram.max_freq_range.is_some_and(|r| r.is_valid()));

    let fan = discrete.fan.as_ref().expect("fan tuning info");
    assert!(fan.speed_range_pct.is_some_and(|r| r.is_valid()));

    // Dry-run ONLY — never mutates user GPU tuning settings during automated tests.
    let dry_res = set_gpu_power_limit(discrete.pci_device_id, 0, true)
        .expect("dry-run power-limit check should succeed for 0%");
    assert!(!dry_res.applied);
    assert!(dry_res.verified_limit_pct.is_none());

    assert!(matches!(
        set_gpu_power_limit(discrete.pci_device_id, 999, true),
        Err(ForgeError::InvalidParameter(_))
    ));

    let reset_dry = reset_gpu_tuning_to_factory(discrete.pci_device_id, true)
        .expect("dry-run factory reset check should succeed");
    assert!(!reset_dry.applied);
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_gemv_w8a32_wave32_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    assert!(
        engine.supports_subgroup_arithmetic() && engine.supports_wave32_subgroup_control(),
        "expected Vulkan 1.3 Wave32 subgroup arithmetic control on RX 6900 XT"
    );

    for (n, k) in [(1u32, 4u32), (37, 52), (129, 256), (512, 1024)] {
        let w_i8 = deterministic_i8_data((n * k) as usize, 701 + n as u64);
        let w_packed = pack_i8_row_major_k4(&w_i8, n as usize, k as usize).unwrap();
        let scales = deterministic_f32_data(n as usize, 702 + n as u64);
        let x = deterministic_f32_data(k as usize, 703 + k as u64);
        let (actual, _timing) = engine
            .run_gemv_w8a32_wave32(&w_packed, &scales, &x, n, k)
            .unwrap_or_else(|e| panic!("W8A32 GEMV {n}x{k} failed: {e}"));
        let expected = ref_gemv_w8a32_f32(&w_i8, &scales, &x, n as usize, k as usize).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-2, 1e-3).unwrap();
        assert!(
            acc.passed,
            "W8A32 GEMV {n}x{k} accuracy failed: max_abs={:.3e}, rmse={:.3e}",
            acc.max_abs_err, acc.rmse
        );
    }

    assert!(matches!(
        engine.run_gemv_w8a32_wave32(&[0], &[1.0], &[1.0, 2.0, 3.0], 1, 3),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_gemv_q4_0_wave32_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    assert!(
        engine.supports_subgroup_arithmetic() && engine.supports_wave32_subgroup_control(),
        "expected Vulkan 1.3 Wave32 subgroup arithmetic control on RX 6900 XT"
    );

    for (n, k) in [(1u32, 32u32), (37, 64), (129, 256), (512, 1024)] {
        let (w_q4, scales) =
            deterministic_q4_0_weights(n as usize, k as usize, 801 + n as u64).unwrap();
        let x = deterministic_f32_data(k as usize, 802 + k as u64);
        let (actual, _timing) = engine
            .run_gemv_q4_0_wave32(&w_q4, &scales, &x, n, k)
            .unwrap_or_else(|e| panic!("Q4_0 GEMV {n}x{k} failed: {e}"));
        let expected = ref_gemv_q4_0_f32(&w_q4, &scales, &x, n as usize, k as usize).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-2, 1e-3).unwrap();
        assert!(
            acc.passed,
            "Q4_0 GEMV {n}x{k} accuracy failed: max_abs={:.3e}, rmse={:.3e}",
            acc.max_abs_err, acc.rmse
        );
    }

    assert!(matches!(
        engine.run_gemv_q4_0_wave32(&[0; 4], &[1.0], &[1.0; 30], 1, 30),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_rmsnorm_rope_subgroup_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    assert!(
        engine.supports_subgroup_arithmetic(),
        "expected Vulkan subgroup arithmetic on RX 6900 XT"
    );

    let eps = 1e-5_f32;
    let theta = 10_000.0_f32;
    for (seq, heads, dim, pos_offset) in [
        (1u32, 1u32, 2u32, 0u32),
        (1, 32, 128, 517),
        (13, 5, 66, 19),
        (128, 32, 128, 0),
    ] {
        let cols = (heads * dim) as usize;
        let x = deterministic_f32_data((seq as usize) * cols, 901 + seq as u64 + pos_offset as u64);
        let w = deterministic_f32_data(cols, 902 + dim as u64);
        let (actual, _timing) = engine
            .run_rmsnorm_rope_f32_subgroup(&x, &w, seq, heads, dim, pos_offset, eps, theta)
            .unwrap_or_else(|e| {
                panic!("Fused RMSNorm+RoPE {seq}x{heads}x{dim} (pos={pos_offset}) failed: {e}")
            });
        let expected = ref_rmsnorm_rope_f32(
            &x,
            &w,
            seq as usize,
            heads as usize,
            dim as usize,
            pos_offset as usize,
            eps,
            theta,
        )
        .unwrap();
        let acc = compare_slices(&actual, &expected, 2e-4, 2e-4).unwrap();
        assert!(
            acc.passed,
            "Fused RMSNorm+RoPE {seq}x{heads}x{dim} (pos={pos_offset}) failed: max_abs={:.3e}, rmse={:.3e}",
            acc.max_abs_err, acc.rmse
        );
    }

    assert!(matches!(
        engine.run_rmsnorm_rope_f32_subgroup(
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0, 1.0],
            1,
            1,
            3,
            0,
            eps,
            theta
        ),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn amf_live_video_capabilities_probe_verifies_vcn_codecs() {
    let report = probe_amf_capabilities().expect("AMF live capability probe must succeed");
    assert!(report.dx11_initialized);
    assert!(!report.runtime_version.is_empty());

    let h264_enc = report
        .codecs
        .iter()
        .find(|c| c.codec == "H.264/AVC" && c.direction == AmfCodecDirection::Encode)
        .expect("H.264/AVC encode entry");
    assert_eq!(h264_enc.acceleration, AmfAcceleration::Hardware);
    assert!(h264_enc.max_width.unwrap_or(0) >= 4096);

    let hevc_enc = report
        .codecs
        .iter()
        .find(|c| c.codec == "HEVC/H.265" && c.direction == AmfCodecDirection::Encode)
        .expect("HEVC encode entry");
    assert_eq!(hevc_enc.acceleration, AmfAcceleration::Hardware);
    assert!(hevc_enc.max_width.unwrap_or(0) >= 4096);

    for dec_codec in ["H.264/AVC", "HEVC/H.265", "VP9", "AV1"] {
        let dec = report
            .codecs
            .iter()
            .find(|c| c.codec == dec_codec && c.direction == AmfCodecDirection::Decode)
            .unwrap_or_else(|| panic!("{dec_codec} decode entry missing"));
        assert_eq!(dec.acceleration, AmfAcceleration::Hardware);
        assert!(dec.max_width.unwrap_or(0) >= 4096);
    }
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_batched_decode_dispatch_matches_single_dispatch() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    let (n, k) = (256u32, 256u32);
    let (w_q4, scales) = deterministic_q4_0_weights(n as usize, k as usize, 1234).unwrap();
    let x = deterministic_f32_data(k as usize, 5678);

    let (actual, timings) = engine
        .bench_gemv_q4_0_wave32_batched(&w_q4, &scales, &x, n, k, 2, 5, 8)
        .expect("batched Q4_0 dispatch must succeed");
    assert_eq!(timings.len(), 5);
    assert!(timings.iter().all(|t| t.gpu_ms.unwrap_or(t.wall_ms) > 0.0));

    let expected = ref_gemv_q4_0_f32(&w_q4, &scales, &x, n as usize, k as usize).unwrap();
    let acc = compare_slices(&actual, &expected, 1e-2, 1e-3).unwrap();
    assert!(acc.passed, "max_abs_err={}", acc.max_abs_err);

    assert!(matches!(
        engine.bench_gemv_q4_0_wave32_batched(&w_q4, &scales, &x, n, k, 0, 1, 0),
        Err(ForgeError::InvalidParameter(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_gemv_q4_k_wave32_edge_cases_and_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    assert!(engine.supports_subgroup_arithmetic());
    assert!(engine.supports_wave32_subgroup_control());

    for (n, k) in [(1u32, 256u32), (13, 512), (65, 1024)] {
        let w_q4k =
            deterministic_q4_k_weights(n as usize, k as usize, 4100 + u64::from(n)).unwrap();
        let x = deterministic_f32_data(k as usize, 4200 + u64::from(k));
        let (actual, timing) = engine
            .run_gemv_q4_k_wave32(&w_q4k, &x, n, k)
            .expect("Wave32 Q4_K GEMV should pass");
        let expected = ref_gemv_q4_k_f32(&w_q4k, &x, n as usize, k as usize).unwrap();
        let acc = compare_slices(&actual, &expected, 1e-3, 1e-4).unwrap();
        assert!(
            acc.passed,
            "Wave32 Q4_K GEMV mismatch on ({n}, {k}): max_abs={}, rmse={}",
            acc.max_abs_err, acc.rmse
        );
        assert!(timing.gpu_ms.unwrap_or(timing.wall_ms) > 0.0);
    }

    // Batched dispatch equivalence check (batch=4)
    let (n, k) = (65u32, 512u32);
    let w_q4k = deterministic_q4_k_weights(n as usize, k as usize, 4301).unwrap();
    let x = deterministic_f32_data(k as usize, 4302);
    let (batched, timings) = engine
        .bench_gemv_q4_k_wave32_batched(&w_q4k, &x, n, k, 2, 3, 4)
        .expect("batched Q4_K dispatch must succeed");
    assert_eq!(timings.len(), 3);
    let expected = ref_gemv_q4_k_f32(&w_q4k, &x, n as usize, k as usize).unwrap();
    let acc = compare_slices(&batched, &expected, 1e-3, 1e-4).unwrap();
    assert!(acc.passed, "batched Q4_K max_abs_err={}", acc.max_abs_err);

    assert!(matches!(
        engine.run_gemv_q4_k_wave32(&[0u32; 36], &[1.0f32; 128], 1, 128),
        Err(ForgeError::InvalidDimension(_))
    ));
}

#[test]
#[ignore = "requires physical AMD GPU"]
fn vulkan_flash_attention_wave32_causal_and_full_validation() {
    let engine = VulkanComputeEngine::new(0).expect("Vulkan compute engine init");
    assert!(engine.supports_subgroup_arithmetic());
    assert!(engine.supports_wave32_subgroup_control());

    for (heads, seq, dim, causal) in [
        (2u32, 64u32, 64u32, true),
        (2, 64, 64, false),
        (2, 37, 64, true),
        (2, 19, 96, false),
        (2, 48, 128, true),
    ] {
        let elems = (heads * seq * dim) as usize;
        let q = deterministic_f32_data(elems, 5100 + u64::from(seq));
        let k = deterministic_f32_data(elems, 5200 + u64::from(dim));
        let v = deterministic_f32_data(elems, 5300 + u64::from(seq));
        let scale = 1.0_f32 / (dim as f32).sqrt();

        let (actual, timing) = engine
            .run_flash_attention_f32_wave32(&q, &k, &v, heads, seq, dim, causal, scale)
            .expect("Wave32 FlashAttention should pass");
        let expected = ref_flash_attention_f32(
            &q,
            &k,
            &v,
            heads as usize,
            seq as usize,
            dim as usize,
            causal,
            scale,
        )
        .unwrap();
        let acc = compare_slices(&actual, &expected, 1e-4, 1e-5).unwrap();
        assert!(
            acc.passed,
            "Wave32 FlashAttention mismatch on ({heads}, {seq}, {dim}, causal={causal}): max_abs={}, rmse={}",
            acc.max_abs_err, acc.rmse
        );
        assert!(timing.gpu_ms.unwrap_or(timing.wall_ms) > 0.0);
    }

    assert!(matches!(
        engine.run_flash_attention_f32_wave32(&[], &[], &[], 1, 8, 0, true, 1.0),
        Err(ForgeError::InvalidDimension(_))
    ));
    let oversized = vec![0.0f32; 8 * 160];
    assert!(matches!(
        engine.run_flash_attention_f32_wave32(
            &oversized, &oversized, &oversized, 1, 8, 160, true, 1.0
        ),
        Err(ForgeError::InvalidDimension(_))
    ));
}
