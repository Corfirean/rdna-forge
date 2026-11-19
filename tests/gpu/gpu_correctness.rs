use std::time::Duration;

use forge_core::{validate_kernel_execution, ForgeError, KernelRequirements, AMD_PCI_VENDOR_ID};
use forge_device::discover_gpus;
use forge_profiler::{snapshot_all_gpus, TelemetryGuard};
use forge_runtime::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    pack_i8_b_transposed_k4, pack_i8_row_major_k4, ref_gemm_f32_exact_f64, ref_gemm_i8_to_i32,
    ref_rmsnorm_f32, ref_rope_f32, verify_hip_transfer, VulkanComputeEngine,
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
