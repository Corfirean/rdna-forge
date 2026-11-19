pub mod hip_verify;
pub mod probe;
pub mod reference;
pub mod shaders;
pub mod vulkan_compute;

pub use hip_verify::{verify_hip_transfer, HipVerificationResult};
pub use probe::probe_all_backends;
pub use reference::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    pack_i8_b_transposed_k4, pack_i8_row_major_k4, ref_gemm_f32, ref_gemm_f32_exact_f64,
    ref_gemm_i8_to_i32, ref_rmsnorm_f32, ref_rope_f32, NumericalAccuracy,
};
pub use shaders::{
    compile_wgsl_to_spirv, compile_wgsl_to_spirv_dp4a, spirv_contains_opcode,
    GEMM_F32_TILED_VEC4_WGSL, GEMM_F32_WGSL, GEMM_I8_DP4A_WGSL, RMSNORM_F32_WGSL, ROPE_F32_WGSL,
    SPIRV_OP_SDOT,
};
pub use vulkan_compute::{GpuTiming, VulkanComputeEngine};

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::ForgeError;

    #[test]
    fn all_embedded_wgsl_shaders_compile_to_spirv() {
        for (name, src) in [
            ("gemm_f32", GEMM_F32_WGSL),
            ("gemm_f32_tiled_vec4", GEMM_F32_TILED_VEC4_WGSL),
            ("rmsnorm_f32", RMSNORM_F32_WGSL),
            ("rope_f32", ROPE_F32_WGSL),
        ] {
            let spv = compile_wgsl_to_spirv(src)
                .unwrap_or_else(|e| panic!("{name} failed SPIR-V compile: {e}"));
            assert!(
                spv.len() > 10 && spv[0] == 0x0723_0203,
                "{name} did not emit valid SPIR-V magic word"
            );
        }

        let dp4a_spv = compile_wgsl_to_spirv_dp4a(GEMM_I8_DP4A_WGSL)
            .expect("gemm_i8_dp4a failed SPIR-V 1.6 compile");
        assert!(dp4a_spv.len() > 10 && dp4a_spv[0] == 0x0723_0203);
        assert!(
            spirv_contains_opcode(&dp4a_spv, SPIRV_OP_SDOT),
            "gemm_i8_dp4a SPIR-V must emit native OpSDot (4450)"
        );
    }

    #[test]
    fn i8_packing_and_exact_reference_work() {
        // A = [[1, 2, 3, 4], [-1, -2, -3, -4]] (2x4)
        let a = [1i8, 2, 3, 4, -1, -2, -3, -4];
        // B = [[1, 0], [1, 2], [1, 3], [1, 4]] (4x2)
        // C[0] = [1+2+3+4, 0+4+9+16] = [10, 29]
        // C[1] = [-10, -29]
        let b = [1i8, 0, 1, 2, 1, 3, 1, 4];

        let a_p = pack_i8_row_major_k4(&a, 2, 4).unwrap();
        let b_p = pack_i8_b_transposed_k4(&b, 4, 2).unwrap();
        assert_eq!(a_p.len(), 2);
        assert_eq!(b_p.len(), 2);

        let c_ref = ref_gemm_i8_to_i32(&a, &b, 2, 4, 2).unwrap();
        assert_eq!(c_ref, vec![10, 29, -10, -29]);

        let exact = compare_i32_exact(&c_ref, &[10, 29, -10, -29]).unwrap();
        assert!(exact.passed);
        assert_eq!(exact.max_abs_err, 0.0);

        let wrong = compare_i32_exact(&c_ref, &[10, 28, -10, -29]).unwrap();
        assert!(!wrong.passed);

        assert!(matches!(
            pack_i8_row_major_k4(&[1, 2, 3], 1, 3),
            Err(ForgeError::InvalidDimension(_))
        ));
    }

    #[test]
    fn cpu_reference_kernels_and_error_metrics_work() {
        // 2x3 * 3x2
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = ref_gemm_f32_exact_f64(&a, &b, 2, 3, 2).unwrap();
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);

        // RMSNorm on constant vector [2, 2, 2, 2] with weight=1 -> rms=2 -> output ~ [1,1,1,1]
        let x = [2.0f32, 2.0, 2.0, 2.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let y = ref_rmsnorm_f32(&x, &w, 1, 4, 1e-6).unwrap();
        for val in y {
            assert!((val - 1.0).abs() < 1e-5);
        }

        // RoPE at pos=0 rotates by angle 0 (identity)
        let r_in = [1.5f32, -2.5, 3.0, 4.0];
        let r_out = ref_rope_f32(&r_in, 1, 1, 4, 10000.0).unwrap();
        assert_eq!(r_out, r_in);

        let acc = compare_slices(&[1.0, 2.001], &[1.0, 2.0], 1e-2, 1e-2).unwrap();
        assert!(acc.passed && acc.all_finite);
        let nan_acc = compare_slices(&[f32::NAN, 2.0], &[1.0, 2.0], 1e-2, 1e-2).unwrap();
        assert!(!nan_acc.passed && !nan_acc.all_finite);
    }

    #[test]
    fn reference_rejects_invalid_dimensions() {
        assert!(matches!(
            ref_gemm_f32(&[], &[], 0, 1, 1),
            Err(ForgeError::InvalidDimension(_))
        ));
        assert!(matches!(
            ref_rmsnorm_f32(&[1.0], &[1.0], 1, 2, 1e-5),
            Err(ForgeError::InvalidDimension(_))
        ));
        assert!(matches!(
            ref_rope_f32(&[1.0, 2.0, 3.0], 1, 1, 3, 10000.0),
            Err(ForgeError::InvalidDimension(_))
        ));
    }
}
