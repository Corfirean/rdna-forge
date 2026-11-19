pub mod hip_verify;
pub mod probe;
pub mod reference;
pub mod shaders;
pub mod vulkan_compute;

pub use hip_verify::{verify_hip_transfer, HipVerificationResult};
pub use probe::probe_all_backends;
pub use reference::{
    compare_slices, deterministic_f32_data, ref_gemm_f32, ref_gemm_f32_exact_f64, ref_rmsnorm_f32,
    ref_rope_f32, NumericalAccuracy,
};
pub use shaders::{compile_wgsl_to_spirv, GEMM_F32_WGSL, RMSNORM_F32_WGSL, ROPE_F32_WGSL};
pub use vulkan_compute::{GpuTiming, VulkanComputeEngine};

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::ForgeError;

    #[test]
    fn all_embedded_wgsl_shaders_compile_to_spirv() {
        for (name, src) in [
            ("gemm_f32", GEMM_F32_WGSL),
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
