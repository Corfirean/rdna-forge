use forge_core::{CapabilityResult, ForgeError, ForgeResult, GpuDeviceInfo, SupportStatus};

use crate::hip_verify::verify_hip_transfer;
use crate::reference::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    deterministic_q4_0_weights, pack_i8_b_transposed_k4, pack_i8_row_major_k4,
    ref_gemm_f32_exact_f64, ref_gemm_i8_to_i32, ref_gemv_q4_0_f32, ref_gemv_w8a32_f32,
    ref_rmsnorm_f32, ref_rmsnorm_rope_f32, ref_rope_f32,
};
use crate::vulkan_compute::{GpuTiming, VulkanComputeEngine};

/// Live Vulkan compute capability checks reported by `forge doctor --verify` and Forge Desktop.
pub const VULKAN_VERIFY_CAPABILITIES: &[&str] = &[
    "GEMM (Vulkan FP32)",
    "GEMM (Vulkan FP32 tiled vec4)",
    "GEMM (Vulkan INT8 DP4A)",
    "GEMV (Vulkan Wave32 W8A32)",
    "GEMV (Vulkan Wave32 Q4_0)",
    "RMSNorm (Vulkan FP32)",
    "RoPE (Vulkan FP32)",
    "Fused RMSNorm+RoPE (Vulkan Subgroup)",
];

/// Returns placeholder `NotTested` capability rows when `--verify` was not requested yet.
pub fn default_untested_capabilities() -> Vec<CapabilityResult> {
    let mut caps = Vec::with_capacity(1 + VULKAN_VERIFY_CAPABILITIES.len());
    caps.push(CapabilityResult {
        name: "HIP transfer".to_string(),
        status: SupportStatus::NotTested,
        detail: Some("pass --verify to run live HIP round-trip test".to_string()),
    });
    for &name in VULKAN_VERIFY_CAPABILITIES {
        caps.push(CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::NotTested,
            detail: Some("pass --verify to run live compute check".to_string()),
        });
    }
    caps
}

/// Runs live HIP transfer and Vulkan compute correctness checks on the primary GPU.
pub fn run_live_capability_checks(gpus: &[GpuDeviceInfo]) -> Vec<CapabilityResult> {
    let mut results = Vec::with_capacity(1 + VULKAN_VERIFY_CAPABILITIES.len());

    match verify_hip_transfer(0, 4 * 1024 * 1024) {
        Ok(res) if res.h2d_d2h_bit_exact => results.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::Verified,
            detail: Some(format!(
                "{} ({}) 4 MiB H2D+D2H bit-exact in {:.3} ms ({:.2} GB/s)",
                res.device_name,
                res.gfx_target.as_deref().unwrap_or("unknown"),
                res.elapsed_ms,
                res.bandwidth_gb_s
            )),
        }),
        Ok(res) => results.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(format!(
                "round-trip data mismatch on {} ({})",
                res.device_name,
                res.gfx_target.as_deref().unwrap_or("unknown")
            )),
        }),
        Err(err) => results.push(CapabilityResult {
            name: "HIP transfer".to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(err.to_string()),
        }),
    }

    if gpus.is_empty() {
        push_unsupported_vulkan_caps(&mut results, "no Vulkan GPU discovered");
        return results;
    }

    match VulkanComputeEngine::new(0) {
        Ok(engine) => {
            results.push(verify_f32_gemm_cap("GEMM (Vulkan FP32)", |a, b, n| {
                engine.run_gemm_f32(a, b, n, n, n)
            }));
            results.push(verify_f32_gemm_cap(
                "GEMM (Vulkan FP32 tiled vec4)",
                |a, b, n| engine.run_gemm_f32_tiled_vec4(a, b, n, n, n),
            ));

            let i8_res = (|| -> ForgeResult<String> {
                let n = 64u32;
                let n_us = n as usize;
                let a = deterministic_i8_data(n_us * n_us, 71);
                let b = deterministic_i8_data(n_us * n_us, 82);
                let ap = pack_i8_row_major_k4(&a, n_us, n_us)?;
                let bp = pack_i8_b_transposed_k4(&b, n_us, n_us)?;
                let (actual, timing) = engine.run_gemm_i8_dp4a(&ap, &bp, n, n / 4, n)?;
                let expected = ref_gemm_i8_to_i32(&a, &b, n_us, n_us, n_us)?;
                let acc = compare_i32_exact(&actual, &expected)?;
                if !acc.passed {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "integer mismatch (max_abs_err={:.3e})",
                        acc.max_abs_err
                    )));
                }
                Ok(format!(
                    "64x64x64 exact (OpSDot, max_abs_err={:.0}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("GEMM (Vulkan INT8 DP4A)", i8_res));

            let rms_res = (|| -> ForgeResult<String> {
                let (rows, cols) = (128u32, 256u32);
                let x = deterministic_f32_data((rows * cols) as usize, 33);
                let w = deterministic_f32_data(cols as usize, 44);
                let (actual, timing) = engine.run_rmsnorm_f32(&x, &w, rows, cols, 1e-5)?;
                let expected = ref_rmsnorm_f32(&x, &w, rows as usize, cols as usize, 1e-5)?;
                let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;
                if !acc.passed {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "numerical mismatch (max_abs_err={:.3e})",
                        acc.max_abs_err
                    )));
                }
                Ok(format!(
                    "128x256 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("RMSNorm (Vulkan FP32)", rms_res));

            let rope_res = (|| -> ForgeResult<String> {
                let (seq, heads, dim) = (64u32, 8u32, 64u32);
                let x = deterministic_f32_data((seq * heads * dim) as usize, 55);
                let (actual, timing) = engine.run_rope_f32(&x, seq, heads, dim, 10_000.0)?;
                let expected =
                    ref_rope_f32(&x, seq as usize, heads as usize, dim as usize, 10_000.0)?;
                let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;
                if !acc.passed {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "numerical mismatch (max_abs_err={:.3e})",
                        acc.max_abs_err
                    )));
                }
                Ok(format!(
                    "64x8x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("RoPE (Vulkan FP32)", rope_res));

            let w8_res = (|| -> ForgeResult<String> {
                let (n, k) = (64u32, 64u32);
                let w_i8 = deterministic_i8_data((n * k) as usize, 61);
                let w_packed = pack_i8_row_major_k4(&w_i8, n as usize, k as usize)?;
                let scales = deterministic_f32_data(n as usize, 62);
                let x = deterministic_f32_data(k as usize, 63);
                let (actual, timing) =
                    engine.run_gemv_w8a32_wave32(&w_packed, &scales, &x, n, k)?;
                let expected = ref_gemv_w8a32_f32(&w_i8, &scales, &x, n as usize, k as usize)?;
                let acc = compare_slices(&actual, &expected, 1e-3, 1e-4)?;
                if !acc.passed {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "numerical mismatch (max_abs_err={:.3e})",
                        acc.max_abs_err
                    )));
                }
                Ok(format!(
                    "64x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("GEMV (Vulkan Wave32 W8A32)", w8_res));

            let q4_res = (|| -> ForgeResult<String> {
                let (n, k) = (64u32, 64u32);
                let (w_q4, scales) = deterministic_q4_0_weights(n as usize, k as usize, 64)?;
                let x = deterministic_f32_data(k as usize, 65);
                let (actual, timing) = engine.run_gemv_q4_0_wave32(&w_q4, &scales, &x, n, k)?;
                let expected = ref_gemv_q4_0_f32(&w_q4, &scales, &x, n as usize, k as usize)?;
                let acc = compare_slices(&actual, &expected, 1e-3, 1e-4)?;
                if !acc.passed {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "numerical mismatch (max_abs_err={:.3e})",
                        acc.max_abs_err
                    )));
                }
                Ok(format!(
                    "64x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result("GEMV (Vulkan Wave32 Q4_0)", q4_res));

            let fused_res = (|| -> ForgeResult<String> {
                let (seq, heads, dim, pos_offset) = (64u32, 8u32, 64u32, 17u32);
                let cols = (heads * dim) as usize;
                let x = deterministic_f32_data((seq as usize) * cols, 66);
                let w = deterministic_f32_data(cols, 67);
                let (actual, timing) = engine.run_rmsnorm_rope_f32_subgroup(
                    &x, &w, seq, heads, dim, pos_offset, 1e-5, 10_000.0,
                )?;
                let expected = ref_rmsnorm_rope_f32(
                    &x,
                    &w,
                    seq as usize,
                    heads as usize,
                    dim as usize,
                    pos_offset as usize,
                    1e-5,
                    10_000.0,
                )?;
                let acc = compare_slices(&actual, &expected, 1e-4, 1e-4)?;
                if !acc.passed {
                    return Err(ForgeError::ExecutionFailed(format!(
                        "numerical mismatch (max_abs_err={:.3e})",
                        acc.max_abs_err
                    )));
                }
                Ok(format!(
                    "64x8x64 passed (max_abs_err={:.2e}, {:.3} ms)",
                    acc.max_abs_err,
                    timing.gpu_ms.unwrap_or(timing.wall_ms)
                ))
            })();
            results.push(cap_result(
                "Fused RMSNorm+RoPE (Vulkan Subgroup)",
                fused_res,
            ));
        }
        Err(err) => push_unsupported_vulkan_caps(&mut results, &err.to_string()),
    }

    results
}

fn cap_result(name: &str, res: ForgeResult<String>) -> CapabilityResult {
    match res {
        Ok(detail) => CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Verified,
            detail: Some(detail),
        },
        Err(err) => CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(err.to_string()),
        },
    }
}

fn push_unsupported_vulkan_caps(results: &mut Vec<CapabilityResult>, reason: &str) {
    for &name in VULKAN_VERIFY_CAPABILITIES {
        results.push(CapabilityResult {
            name: name.to_string(),
            status: SupportStatus::Unsupported,
            detail: Some(reason.to_string()),
        });
    }
}

fn verify_f32_gemm_cap(
    name: &str,
    run_fn: impl FnOnce(&[f32], &[f32], u32) -> ForgeResult<(Vec<f32>, GpuTiming)>,
) -> CapabilityResult {
    let res = (|| -> ForgeResult<String> {
        let n = 64u32;
        let a = deterministic_f32_data((n * n) as usize, 11);
        let b = deterministic_f32_data((n * n) as usize, 22);
        let (actual, timing) = run_fn(&a, &b, n)?;
        let expected = ref_gemm_f32_exact_f64(&a, &b, n as usize, n as usize, n as usize)?;
        let acc = compare_slices(&actual, &expected, 1e-3, 1e-4)?;
        if !acc.passed {
            return Err(ForgeError::ExecutionFailed(format!(
                "numerical mismatch (max_abs_err={:.3e}, rmse={:.3e})",
                acc.max_abs_err, acc.rmse
            )));
        }
        Ok(format!(
            "64x64x64 passed (max_abs_err={:.2e}, {:.3} ms)",
            acc.max_abs_err,
            timing.gpu_ms.unwrap_or(timing.wall_ms)
        ))
    })();
    cap_result(name, res)
}
