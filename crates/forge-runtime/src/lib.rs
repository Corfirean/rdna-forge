pub mod amf;
pub mod hip_verify;
pub mod probe;
pub mod reference;
pub mod shaders;
pub mod verify;
pub mod vulkan_compute;

pub use amf::{
    amf_surface_format_name, format_amf_version, has_annex_b_start_code, probe_amf_capabilities,
    run_amf_encode_benchmark, AmfAcceleration, AmfCapabilitiesReport, AmfCodecCapability,
    AmfCodecDirection, AmfEncodeBenchConfig, AmfEncodeBenchResult, AmfEncodeCodec,
    AmfQualityPreset, AmfVariantStruct,
};
pub use hip_verify::{verify_hip_transfer, HipVerificationResult};
pub use probe::probe_all_backends;
pub use reference::{
    compare_i32_exact, compare_slices, deterministic_f32_data, deterministic_i8_data,
    deterministic_q4_0_weights, deterministic_q4_k_weights, f16_bits_to_f32, f32_to_f16_bits,
    pack_i8_b_transposed_k4, pack_i8_row_major_k4, pack_q4_k_block, ref_flash_attention_f32,
    ref_gemm_f32, ref_gemm_f32_exact_f64, ref_gemm_i8_to_i32, ref_gemv_q4_0_f32, ref_gemv_q4_k_f32,
    ref_gemv_w8a32_f32, ref_rmsnorm_f32, ref_rmsnorm_rope_f32, ref_rope_f32, unpack_q4_k_scale_min,
    NumericalAccuracy, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_U32S,
};
pub use shaders::{
    compile_wgsl_to_spirv, compile_wgsl_to_spirv_dp4a, compile_wgsl_to_spirv_subgroup,
    spirv_contains_opcode, FLASH_ATTENTION_F32_WAVE32_WGSL, GEMM_F32_TILED_VEC4_WGSL,
    GEMM_F32_WGSL, GEMM_I8_DP4A_WGSL, GEMV_Q4_0_WAVE32_WGSL, GEMV_Q4_K_WAVE32_WGSL,
    GEMV_W8A32_WAVE32_WGSL, RMSNORM_F32_WGSL, RMSNORM_ROPE_F32_SUBGROUP_WGSL, ROPE_F32_WGSL,
    SPIRV_OP_GROUP_NON_UNIFORM_FADD, SPIRV_OP_SDOT,
};
pub use verify::{
    default_untested_capabilities, run_live_capability_checks, VULKAN_VERIFY_CAPABILITIES,
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

        for (name, src) in [
            ("gemv_w8a32_wave32", GEMV_W8A32_WAVE32_WGSL),
            ("gemv_q4_0_wave32", GEMV_Q4_0_WAVE32_WGSL),
            ("gemv_q4_k_wave32", GEMV_Q4_K_WAVE32_WGSL),
            ("rmsnorm_rope_f32_subgroup", RMSNORM_ROPE_F32_SUBGROUP_WGSL),
            (
                "flash_attention_f32_wave32",
                FLASH_ATTENTION_F32_WAVE32_WGSL,
            ),
        ] {
            let sg_spv = compile_wgsl_to_spirv_subgroup(src)
                .unwrap_or_else(|e| panic!("{name} failed subgroup SPIR-V compile: {e}"));
            assert!(sg_spv.len() > 10 && sg_spv[0] == 0x0723_0203);
            assert!(
                spirv_contains_opcode(&sg_spv, SPIRV_OP_GROUP_NON_UNIFORM_FADD),
                "{name} SPIR-V must emit OpGroupNonUniformFAdd (350)"
            );
        }
    }

    #[test]
    fn decode_reference_kernels_and_q4_packing_work() {
        // W8A32 exact row dot check: row0=[1,2,3,4]*0.5 dot [1,1,1,1] = 5.0
        // row1=[-1,-2,-3,-4]*0.25 dot [1,1,1,1] = -2.5
        let w_i8 = [1i8, 2, 3, 4, -1, -2, -3, -4];
        let scales = [0.5f32, 0.25f32];
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let y_w8 = ref_gemv_w8a32_f32(&w_i8, &scales, &x, 2, 4).unwrap();
        assert_eq!(y_w8, vec![5.0, -2.5]);

        // Q4_0 exact single block: nibble 9 -> (9 - 8) = +1; word 0x99999999 = all +1s.
        // Scale 0.5, x = 1.0 -> 32 * 1 * 0.5 = 16.0.
        // Nibble 7 -> (7 - 8) = -1; word 0x77777777 = all -1s -> scale 0.25 -> -8.0.
        let w_q4 = [
            0x9999_9999u32,
            0x9999_9999,
            0x9999_9999,
            0x9999_9999,
            0x7777_7777,
            0x7777_7777,
            0x7777_7777,
            0x7777_7777,
        ];
        let x32 = vec![1.0f32; 32];
        let y_q4 = ref_gemv_q4_0_f32(&w_q4, &[0.5, 0.25], &x32, 2, 32).unwrap();
        assert_eq!(y_q4, vec![16.0, -8.0]);

        let (det_w, det_s) = deterministic_q4_0_weights(3, 64, 42).unwrap();
        assert_eq!(det_w.len(), 3 * 2 * 4);
        assert_eq!(det_s.len(), 3 * 2);

        // Fused RMSNorm+RoPE with pos_offset > 0 must match chained CPU reference at the same positions
        let rx = [1.0f32, -2.0, 0.5, 1.5, -1.0, 2.0, 3.0, -0.5];
        let rw = [1.0f32, 0.5, -1.5, 2.0];
        let fused = ref_rmsnorm_rope_f32(&rx, &rw, 2, 2, 2, 10, 1e-5, 10000.0).unwrap();
        let norm_only = ref_rmsnorm_f32(&rx, &rw, 2, 4, 1e-5).unwrap();
        // Position 10 + 0 vs 10 + 1: embed into 11-token identity to check row 10 & row 11
        let mut padded = vec![0.0f32; 12 * 4];
        padded[10 * 4..12 * 4].copy_from_slice(&norm_only);
        let chained = ref_rope_f32(&padded, 12, 2, 2, 10000.0).unwrap();
        let acc = compare_slices(&fused, &chained[10 * 4..12 * 4], 1e-5, 1e-5).unwrap();
        assert!(acc.passed, "max_abs_err={}", acc.max_abs_err);

        assert!(matches!(
            deterministic_q4_0_weights(2, 30, 1),
            Err(ForgeError::InvalidDimension(_))
        ));
        assert!(matches!(
            pack_i8_row_major_k4(&[1, 2, 3], 1, 3),
            Err(ForgeError::InvalidDimension(_))
        ));
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

    #[test]
    fn amf_version_and_enum_decoders_work() {
        assert_eq!(format_amf_version(0x0001_0004_0025_0000), "1.4.37.0");
        assert_eq!(format_amf_version(0x0001_0005_0003_0000), "1.5.3.0");

        assert_eq!(AmfAcceleration::from_amf(-1), AmfAcceleration::NotSupported);
        assert_eq!(AmfAcceleration::from_amf(0), AmfAcceleration::Hardware);
        assert_eq!(AmfAcceleration::from_amf(1), AmfAcceleration::Gpu);
        assert_eq!(AmfAcceleration::from_amf(2), AmfAcceleration::Software);
        assert_eq!(AmfAcceleration::from_amf(9), AmfAcceleration::Unknown(9));
        assert_eq!(AmfAcceleration::Hardware.to_string(), "HARDWARE");

        assert_eq!(amf_surface_format_name(0), "UNKNOWN");
        assert_eq!(amf_surface_format_name(1), "NV12");
        assert_eq!(amf_surface_format_name(10), "P010");
        assert_eq!(amf_surface_format_name(26), "R32G8X24");
        assert_eq!(amf_surface_format_name(999), "FORMAT_999");
    }

    #[test]
    fn amf_report_json_roundtrip() {
        let report = AmfCapabilitiesReport {
            runtime_version: "1.4.37.0".to_string(),
            dx11_initialized: true,
            codecs: vec![
                AmfCodecCapability {
                    codec: "HEVC/H.265".to_string(),
                    component_id: "AMFVideoEncoderHW_HEVC".to_string(),
                    direction: AmfCodecDirection::Encode,
                    acceleration: AmfAcceleration::Hardware,
                    min_width: Some(128),
                    max_width: Some(8192),
                    min_height: Some(128),
                    max_height: Some(4352),
                    vert_align: Some(16),
                    interlaced_supported: Some(false),
                    native_formats: vec!["NV12".to_string(), "P010".to_string()],
                    supported_formats: vec!["NV12".to_string(), "P010".to_string()],
                    detail: None,
                },
                AmfCodecCapability {
                    codec: "AV1".to_string(),
                    component_id: "AMFVideoEncoderHW_AV1".to_string(),
                    direction: AmfCodecDirection::Encode,
                    acceleration: AmfAcceleration::NotSupported,
                    min_width: None,
                    max_width: None,
                    min_height: None,
                    max_height: None,
                    vert_align: None,
                    interlaced_supported: None,
                    native_formats: Vec::new(),
                    supported_formats: Vec::new(),
                    detail: Some("CreateComponent returned error 30".to_string()),
                },
            ],
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize AMF report");
        let parsed: AmfCapabilitiesReport =
            serde_json::from_str(&json).expect("deserialize AMF report");
        assert_eq!(parsed, report);
    }

    #[test]
    fn default_untested_capabilities_covers_hip_and_vulkan_checks() {
        let caps = default_untested_capabilities();
        assert_eq!(caps.len(), 1 + VULKAN_VERIFY_CAPABILITIES.len());
        assert_eq!(caps[0].name, "HIP transfer");
        assert!(caps
            .iter()
            .all(|cap| cap.status == forge_core::SupportStatus::NotTested));
        for (cap, expected_name) in caps[1..].iter().zip(VULKAN_VERIFY_CAPABILITIES.iter()) {
            assert_eq!(cap.name, *expected_name);
        }
    }

    #[test]
    fn q4_k_and_flash_attention_references_work() {
        // IEEE-754 f16 exact round-trips (normal, zero, subnormal, negative)
        for v in [0.0f32, 1.0, -2.5, 0.5, 0.125, 0.000_976_562_5] {
            let decoded = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!((decoded - v).abs() < 1e-6, "v={v} decoded={decoded}");
        }

        // Exercise >31 6-bit scales & mins across all 8 sub-blocks (indices 4..8 use top 2 bits)
        let scales = [1u8, 2, 17, 31, 35, 48, 61, 63];
        let mins = [0u8, 3, 19, 32, 41, 52, 59, 62];
        let mut qs = [0u8; 128];
        // Set all low & high nibbles in chunk 0 to known values and check exact Q4_K dot product
        qs[0..32].fill(0x21); // lo=1 (weights 0..31), hi=2 (weights 32..63)
        let block = pack_q4_k_block(0.5, 0.25, &scales, &mins, &qs);
        let mut scale_bytes = [0u8; 12];
        for i in 0..3 {
            scale_bytes[i * 4..i * 4 + 4].copy_from_slice(&block[1 + i].to_le_bytes());
        }
        for j in 0..8 {
            assert_eq!(unpack_q4_k_scale_min(&scale_bytes, j), (scales[j], mins[j]));
        }

        // Row dot with x = 1.0:
        // Chunk 0 lo (32 weights): 32 * (0.5 * 1 * 1 - 0.25 * 0) = 16.0
        // Chunk 0 hi (32 weights): 32 * (0.5 * 2 * 2 - 0.25 * 3) = 40.0
        // Chunks 1..3 have q=0, so weight = -0.25 * mins[j] for 32 elements each:
        // 32 * (-0.25) * (19 + 32 + 41 + 52 + 59 + 62) = -8 * 265 = -2120.0
        // Total = 56.0 - 2120.0 = -2064.0
        let x256 = vec![1.0f32; 256];
        let y_q4k = ref_gemv_q4_k_f32(&block, &x256, 1, 256).unwrap();
        assert!((y_q4k[0] - (-2064.0)).abs() < 1e-4, "got {}", y_q4k[0]);

        let det_q4k = deterministic_q4_k_weights(2, 512, 99).unwrap();
        assert_eq!(det_q4k.len(), 2 * 2 * Q4_K_BLOCK_U32S);
        assert!(matches!(
            deterministic_q4_k_weights(1, 128, 1),
            Err(ForgeError::InvalidDimension(_))
        ));

        // FlashAttention analytical 2-token causal vs full test:
        // Q = [[1,0], [0,1]], K = [[1,0], [0,1]], V = [[2,3], [5,7]], scale = 1.0
        // Row 0 causal attends only to key 0 -> out[0] = [2, 3].
        // Row 1 causal attends to [0, 1] with logits [0, 1] -> softmax weights [1/(1+e), e/(1+e)].
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [2.0f32, 3.0, 5.0, 7.0];
        let o_causal = ref_flash_attention_f32(&q, &k, &v, 1, 2, 2, true, 1.0).unwrap();
        assert!((o_causal[0] - 2.0).abs() < 1e-6);
        assert!((o_causal[1] - 3.0).abs() < 1e-6);
        let e1 = 1.0f32.exp();
        let w0 = 1.0 / (1.0 + e1);
        let w1 = e1 / (1.0 + e1);
        assert!((o_causal[2] - (w0 * 2.0 + w1 * 5.0)).abs() < 1e-5);
        assert!((o_causal[3] - (w0 * 3.0 + w1 * 7.0)).abs() < 1e-5);

        let o_full = ref_flash_attention_f32(&q, &k, &v, 1, 2, 2, false, 1.0).unwrap();
        assert!((o_full[0] - (w1 * 2.0 + w0 * 5.0)).abs() < 1e-5);
        assert!((o_full[1] - (w1 * 3.0 + w0 * 7.0)).abs() < 1e-5);
    }

    #[test]
    fn amf_variant_struct_layout_and_annex_b_detector_work() {
        assert_eq!(std::mem::size_of::<AmfVariantStruct>(), 24);
        assert_eq!(std::mem::align_of::<AmfVariantStruct>(), 8);

        let v_i64 = AmfVariantStruct::from_i64(12_000_000);
        assert_eq!(v_i64.ty, AmfVariantStruct::TYPE_INT64);
        assert_eq!(v_i64.data0, 12_000_000);
        assert_eq!(v_i64.data1, 0);

        let v_size = AmfVariantStruct::from_size(1920, 1080);
        assert_eq!(v_size.ty, AmfVariantStruct::TYPE_SIZE);
        assert_eq!((v_size.data0 & 0xffff_ffff) as i32, 1920);
        assert_eq!(((v_size.data0 >> 32) & 0xffff_ffff) as i32, 1080);

        let v_rate = AmfVariantStruct::from_rate(60, 1);
        assert_eq!(v_rate.ty, AmfVariantStruct::TYPE_RATE);
        assert_eq!((v_rate.data0 & 0xffff_ffff) as u32, 60);
        assert_eq!(((v_rate.data0 >> 32) & 0xffff_ffff) as u32, 1);

        assert!(has_annex_b_start_code(&[
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64
        ]));
        assert!(has_annex_b_start_code(&[0x00, 0x00, 0x01, 0x40, 0x01]));
        assert!(!has_annex_b_start_code(&[0x01, 0x02, 0x03, 0x04]));
        assert!(!has_annex_b_start_code(&[]));
    }

    #[test]
    fn amf_encode_config_validation_rejects_invalid_params() {
        let valid = AmfEncodeBenchConfig {
            codec: AmfEncodeCodec::H264Avc,
            width: 1920,
            height: 1080,
            fps: 60,
            target_bitrate_bps: 12_000_000,
            preset: AmfQualityPreset::Speed,
            warmup_frames: 15,
            measured_frames: 120,
        };
        assert!(valid.validate().is_ok());

        // Odd width
        assert!(AmfEncodeBenchConfig {
            width: 1921,
            ..valid.clone()
        }
        .validate()
        .is_err());

        // Odd height
        assert!(AmfEncodeBenchConfig {
            height: 1081,
            ..valid.clone()
        }
        .validate()
        .is_err());

        // Zero FPS
        assert!(AmfEncodeBenchConfig {
            fps: 0,
            ..valid.clone()
        }
        .validate()
        .is_err());

        // Zero measured frames
        assert!(AmfEncodeBenchConfig {
            measured_frames: 0,
            ..valid
        }
        .validate()
        .is_err());
    }
}
