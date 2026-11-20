pub mod llama;
pub mod llm_optimizer;
pub mod llm_recommender;
pub mod memory;
pub mod stats;
pub mod triton;
pub mod video;
pub mod vulkan_suite;

pub use llama::{
    parse_llama_bench_json, run_llama_comparison, BackendRunOutcome, LlamaBenchConfig,
    LlamaBenchEntry, LlamaComparisonReport,
};
pub use llm_optimizer::{
    apply_lm_studio_config, apply_windows_and_ollama_env_tweaks, autotune_llm_model,
    build_lm_studio_model_config_json, discover_local_llm_assets, estimate_kv_and_state_bytes,
    gguf_file_type_name, inspect_gguf_model, plan_llm_configuration,
    plan_llm_configuration_for_device, plan_llm_configuration_for_gpu, rollback_lm_studio_config,
    rollback_lm_studio_config_in_dir, rollback_windows_env_tweaks, EnvAndOllamaPatchResult,
    EnvRollbackItem, EnvRollbackResult, EnvVarChange, GgufModelMetadata, KvCacheMemoryBudget,
    KvCacheQuant, LlmAutotuneCandidateResult, LlmAutotuneConfig, LlmAutotuneReport,
    LlmOptimizationPlan, LlmTargetGpuContext, LlmTweakProfile, LmStudioPatchResult,
    LmStudioRollbackResult, LocalLlmAssets, ProcessVramImpact, RollbackFileAction,
    RDNA_FORGE_MANAGED_ENV_KEYS,
};
pub use llm_recommender::{
    recommend_models_for_device, recommend_models_for_gpu_budget, GpuModelRecommendationReport,
    ModelFitVerdict, ModelRecommendation,
};
pub use memory::peak_process_rss_bytes;
pub use stats::{compute_latency_stats, BenchmarkResult, LatencyStats, StabilityResult};
pub use triton::{
    parse_triton_suite_json, resolve_default_triton_script, run_triton_suite, TritonBenchConfig,
    TritonKernelOutcome, TritonProvenance, TritonSuiteReport,
};
pub use video::{run_video_benchmark_suite, VideoBenchReport, VideoBenchSuiteConfig};
pub use vulkan_suite::{
    run_vulkan_suite, VulkanBenchConfig, VulkanBenchReport, VulkanKernelSelection,
};

pub(crate) fn run_with_pre_post_telemetry<R>(
    pci_device_id: Option<u32>,
    f: impl FnOnce() -> R,
) -> (R, Option<forge_profiler::TelemetrySummary>) {
    let pre = forge_profiler::snapshot_for_device(pci_device_id)
        .ok()
        .flatten();
    let res = f();
    let post = forge_profiler::snapshot_for_device(pci_device_id)
        .ok()
        .flatten();
    let mut samples = Vec::with_capacity(2);
    if let Some(ref s) = pre {
        samples.push(s.clone());
    }
    if let Some(s) = post {
        samples.push(s);
    }
    let summary = forge_profiler::summarize_samples(&samples, 0, pre.as_ref());
    (res, summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::ForgeError;

    #[test]
    fn latency_stats_computes_median_iqr_mad_and_p95() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_latency_stats(10, &samples).unwrap();
        assert_eq!(stats.warmup_iters, 10);
        assert_eq!(stats.measured_iters, 5);
        assert!((stats.min_ms - 1.0).abs() < 1e-9);
        assert!((stats.median_ms - 3.0).abs() < 1e-9);
        assert!((stats.mean_ms - 3.0).abs() < 1e-9);
        assert!((stats.max_ms - 5.0).abs() < 1e-9);
        assert!((stats.iqr_ms - 2.0).abs() < 1e-9);
        assert!((stats.mad_ms - 1.0).abs() < 1e-9);
        assert!((stats.p95_ms - 4.8).abs() < 1e-9);

        assert!(matches!(
            compute_latency_stats(0, &[]),
            Err(ForgeError::InvalidDimension(_))
        ));
        assert!(matches!(
            compute_latency_stats(0, &[1.0, f64::NAN]),
            Err(ForgeError::ExecutionFailed(_))
        ));
    }

    #[test]
    fn llama_bench_json_parses_official_schema() {
        let sample = r#"[
          {
            "build_commit": "abc1234",
            "build_number": 12532,
            "cpu_info": "AMD Ryzen 7 7700X",
            "gpu_info": "AMD Radeon RX 6900 XT",
            "backends": "Vulkan",
            "model_filename": "model.gguf",
            "model_type": "qwen2 1.5B Q4_K - Medium",
            "model_size": 1117000000,
            "model_n_params": 1543714304,
            "n_gpu_layers": 99,
            "main_gpu": 0,
            "n_prompt": 512,
            "n_gen": 0,
            "avg_ts": 1200.0,
            "stddev_ts": 10.0,
            "avg_ns": 426666666,
            "stddev_ns": 3500000,
            "samples_ns": [425000000, 426000000, 429000000],
            "samples_ts": [1204.7, 1201.9, 1193.5]
          },
          {
            "build_commit": "abc1234",
            "build_number": 12532,
            "gpu_info": "AMD Radeon RX 6900 XT",
            "backends": "Vulkan",
            "model_type": "qwen2 1.5B Q4_K - Medium",
            "model_size": 1117000000,
            "n_gpu_layers": 99,
            "main_gpu": 0,
            "n_prompt": 0,
            "n_gen": 128,
            "avg_ts": 110.0,
            "stddev_ts": 1.0,
            "avg_ns": 1163636363,
            "stddev_ns": 10000000,
            "samples_ns": [1160000000, 1163000000, 1168000000],
            "samples_ts": [110.34, 110.06, 109.59]
          }
        ]"#;

        let parsed = parse_llama_bench_json(sample, "vulkan").unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].test_kind, "pp512");
        assert!((parsed[0].median_tok_s - 1201.9).abs() < 1e-6);
        assert_eq!(parsed[1].test_kind, "tg128");
        assert!((parsed[1].median_tok_s - 110.06).abs() < 1e-6);
    }

    #[test]
    fn triton_suite_json_parses_verified_and_unsupported_entries() {
        let sample = r#"{
          "provenance": {
            "python_version": "3.11.9",
            "torch_version": "2.15.0+rocm10.2.0",
            "torch_hip_version": "10.2.0",
            "triton_version": "3.9.0",
            "triton_windows_version": "3.9.0.post30",
            "device_name": "AMD Radeon RX 6900 XT",
            "gcn_arch_name": "gfx1030",
            "total_vram_bytes": 17163091968,
            "os": "Windows 11"
          },
          "benchmarks": [
            {
              "name": "quantized_gemm_int8",
              "backend": "pytorch_int_mm",
              "shape": "2048x2048x2048",
              "dtype": "int8",
              "status": "unsupported",
              "latency": null,
              "throughput_unit": null,
              "throughput_median": null,
              "accuracy": null,
              "peak_vram_allocated_bytes": null,
              "peak_vram_reserved_bytes": null,
              "peak_rss_bytes": null,
              "error": "RuntimeError: _int_mm not supported on gfx1030",
              "notes": null
            }
          ],
          "stability": null
        }"#;

        let parsed = parse_triton_suite_json(sample).unwrap();
        assert_eq!(parsed.provenance.gcn_arch_name.as_deref(), Some("gfx1030"));
        assert_eq!(parsed.benchmarks[0].status, "unsupported");
    }

    #[test]
    fn gguf_file_type_and_kv_budget_estimator_work() {
        assert_eq!(
            gguf_file_type_name(Some(30), "Huihui-Qwen3.8-27B-UD-IQ4_XS.gguf"),
            "UD-IQ4_XS"
        );
        assert_eq!(gguf_file_type_name(Some(15), "model.gguf"), "Q4_K_M");

        let meta = GgufModelMetadata {
            file_path: "Qwen3.8-27B-UD-IQ4_XS.gguf".to_string(),
            file_name: "Qwen3.8-27B-UD-IQ4_XS.gguf".to_string(),
            file_size_bytes: 14_403_496_864,
            gguf_version: 3,
            tensor_count: 866,
            kv_count: 50,
            architecture: "qwen35".to_string(),
            model_name: Some("Qwen3.8-27B".to_string()),
            size_label: Some("27B".to_string()),
            quantized_by: Some("Unsloth".to_string()),
            file_type_code: Some(30),
            quantization_name: "UD-IQ4_XS".to_string(),
            block_count: 65,
            nextn_predict_layers: 1,
            declared_context_length: 262_144,
            embedding_length: 5120,
            feed_forward_length: Some(17408),
            head_count: 24,
            head_count_kv: 4,
            key_length: 256,
            value_length: 256,
            full_attention_interval: Some(4),
            ssm_state_size: Some(128),
            ssm_inner_size: Some(6144),
            ssm_group_count: Some(16),
            expert_count: None,
            expert_used_count: None,
            expert_feed_forward_length: None,
            is_moe: false,
            bytes_per_layer: 14_403_496_864 / 65,
        };

        let f16_bytes = estimate_kv_and_state_bytes(&meta, 4096, KvCacheQuant::F16);
        let q8_bytes = estimate_kv_and_state_bytes(&meta, 4096, KvCacheQuant::Q8_0);
        let q4_bytes = estimate_kv_and_state_bytes(&meta, 4096, KvCacheQuant::Q4_0);
        assert!(f16_bytes > q8_bytes && q8_bytes > q4_bytes);

        let existing = r#"{
            "preset": "",
            "operation": { "fields": [] },
            "load": {
                "fields": [
                    { "key": "llm.load.promptTemplate", "value": "custom-jinja" },
                    { "key": "llm.load.llama.flashAttention", "value": true }
                ]
            }
        }"#;
        let profile = LlmTweakProfile {
            id: "vulkan_wddm_priority".to_string(),
            title: "Vulkan WDDM Priority".to_string(),
            backend: "vulkan".to_string(),
            device_selector: "Vulkan0".to_string(),
            n_gpu_layers: 99,
            n_cpu_moe: 0,
            split_mode: "none".to_string(),
            flash_attn: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            batch_size: 2048,
            ubatch_size: 512,
            threads: 8,
            env_vars: vec![],
            rationale: "test".to_string(),
            llama_server_cmd: "llama-server".to_string(),
        };
        let (merged_json, ratio_str) =
            build_lm_studio_model_config_json(Some(existing), &profile, 65, 4096, &[1]);
        assert_eq!(ratio_str, "1.00 (100% GPU)");
        assert!(merged_json.contains("custom-jinja"));
        assert!(merged_json.contains("\"disabledGpus\": [\n            1\n          ]"));
        assert!(merged_json.contains("\"key\": \"llm.load.contextLength\""));
        assert!(merged_json.contains("\"value\": 4096"));

        let (merged_zero, _) = build_lm_studio_model_config_json(None, &profile, 65, 4096, &[0]);
        assert!(merged_zero.contains("\"disabledGpus\": [\n            0\n          ]"));

        let (merged_empty, _) = build_lm_studio_model_config_json(None, &profile, 65, 4096, &[]);
        assert!(!merged_empty.contains("load.gpuSplitConfig"));
    }

    #[test]
    fn multi_architecture_llm_planner_tailors_profiles_for_rdna1_2_3_35_and_4() {
        use forge_core::GpuArchitecture;

        let sample_27b = GgufModelMetadata {
            file_path: "Qwen3.8-27B-UD-IQ4_XS.gguf".to_string(),
            file_name: "Qwen3.8-27B-UD-IQ4_XS.gguf".to_string(),
            file_size_bytes: 14_403_496_864,
            gguf_version: 3,
            tensor_count: 866,
            kv_count: 50,
            architecture: "qwen35".to_string(),
            model_name: Some("Qwen3.8-27B".to_string()),
            size_label: Some("27B".to_string()),
            quantized_by: Some("Unsloth".to_string()),
            file_type_code: Some(30),
            quantization_name: "UD-IQ4_XS".to_string(),
            block_count: 65,
            nextn_predict_layers: 1,
            declared_context_length: 262_144,
            embedding_length: 5120,
            feed_forward_length: Some(17408),
            head_count: 24,
            head_count_kv: 4,
            key_length: 256,
            value_length: 256,
            full_attention_interval: Some(4),
            ssm_state_size: Some(128),
            ssm_inner_size: Some(6144),
            ssm_group_count: Some(16),
            expert_count: None,
            expert_used_count: None,
            expert_feed_forward_length: None,
            is_moe: false,
            bytes_per_layer: 14_403_496_864 / 65,
        };

        // 1. RX 6900 XT (RDNA2, 16 GiB + iGPU)
        let rx6900_ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "AMD Radeon RX 6900 XT".to_string(),
            gfx_target: "gfx1030".to_string(),
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
            matched_profile_id: Some("rx6900xt".to_string()),
            matched_profile_name: Some("AMD Radeon RX 6900 XT".to_string()),
            vk_visible_device: 0,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![1],
            secondary_igpu_name: Some("AMD Radeon(TM) Graphics".to_string()),
            physical_cpu_cores: 8,
            total_ram_mib: 32_768,
            total_vram_mib: 16_368,
            used_vram_mib: 1_500,
        };
        let p_6900 = plan_llm_configuration_for_gpu(&sample_27b, &rx6900_ctx, 4096, 512);
        assert_eq!(p_6900.gpu_architecture, "RDNA2");
        assert!(p_6900
            .recommended_profiles
            .iter()
            .any(|p| p.id == "rocm_q4_kv_long_ctx"));
        assert!(!p_6900
            .recommended_profiles
            .iter()
            .any(|p| p.id == "rocm_wmma_hipblaslt"));
        assert!(p_6900.has_secondary_igpu);
        assert_eq!(p_6900.disabled_gpu_indices, vec![1]);

        // 2. RX 6600 XT (RDNA2, 8 GiB, partial offload)
        let rx6600_ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "AMD Radeon RX 6600 XT".to_string(),
            gfx_target: "gfx1032".to_string(),
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
            matched_profile_id: Some("rx6600xt".to_string()),
            matched_profile_name: Some("AMD Radeon RX 6600 XT".to_string()),
            vk_visible_device: 0,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![],
            secondary_igpu_name: None,
            physical_cpu_cores: 6,
            total_ram_mib: 32_768,
            total_vram_mib: 8_176,
            used_vram_mib: 800,
        };
        let p_6600 = plan_llm_configuration_for_gpu(&sample_27b, &rx6600_ctx, 4096, 512);
        assert!(!p_6600.kv_budgets[0].fits_in_free_vram);
        assert!(p_6600.kv_budgets[0].strict_zero_spill_ngl < 65);
        assert!(p_6600
            .recommended_profiles
            .iter()
            .any(|p| p.id == "vulkan_strict_zero_spill"));
        assert!(!p_6600.has_secondary_igpu);

        // 3. RX 7900 XTX (RDNA3, 24 GiB, WMMA + hipBLASLt)
        let rx7900_ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "AMD Radeon RX 7900 XTX".to_string(),
            gfx_target: "gfx1100".to_string(),
            architecture: GpuArchitecture::Rdna3,
            discrete: true,
            matched_profile_id: Some("rx7900xtx".to_string()),
            matched_profile_name: Some("AMD Radeon RX 7900 XTX".to_string()),
            vk_visible_device: 0,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![],
            secondary_igpu_name: None,
            physical_cpu_cores: 8,
            total_ram_mib: 65_536,
            total_vram_mib: 24_560,
            used_vram_mib: 1_200,
        };
        let p_7900 = plan_llm_configuration_for_gpu(&sample_27b, &rx7900_ctx, 4096, 512);
        let rocm_7900 = p_7900
            .recommended_profiles
            .iter()
            .find(|p| p.id == "rocm_wmma_hipblaslt")
            .expect("RDNA3 must expose rocm_wmma_hipblaslt");
        assert!(rocm_7900
            .env_vars
            .iter()
            .any(|(k, v)| k == "ROCBLAS_USE_HIPBLASLT" && v == "1"));
        assert_eq!(rocm_7900.cache_type_k, "q8_0");

        // 4. RX 9070 XT (RDNA4, 16 GiB, WMMA + hipBLASLt)
        let rx9070_ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "AMD Radeon RX 9070 XT".to_string(),
            gfx_target: "gfx1201".to_string(),
            architecture: GpuArchitecture::Rdna4,
            discrete: true,
            matched_profile_id: Some("rx9070xt".to_string()),
            matched_profile_name: Some("AMD Radeon RX 9070 XT".to_string()),
            vk_visible_device: 1,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![0],
            secondary_igpu_name: None,
            physical_cpu_cores: 8,
            total_ram_mib: 32_768,
            total_vram_mib: 16_304,
            used_vram_mib: 1_000,
        };
        let p_9070 = plan_llm_configuration_for_gpu(&sample_27b, &rx9070_ctx, 4096, 512);
        assert_eq!(p_9070.gpu_architecture, "RDNA4");
        assert_eq!(p_9070.vk_visible_device, 1);
        assert!(p_9070
            .recommended_profiles
            .iter()
            .any(|p| p.id == "rocm_wmma_hipblaslt"));
        assert!(p_9070
            .recommended_profiles
            .iter()
            .find(|p| p.id == "vulkan_wddm_priority")
            .unwrap()
            .env_vars
            .iter()
            .any(|(k, v)| k == "GGML_VK_VISIBLE_DEVICES" && v == "1"));

        // 5. Radeon 8060S Strix Halo (RDNA3.5 APU, 64 GiB unified pool + 512 MiB carveout test)
        let halo_ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "AMD Radeon 8060S Graphics".to_string(),
            gfx_target: "gfx1151".to_string(),
            architecture: GpuArchitecture::Rdna35,
            discrete: false,
            matched_profile_id: Some("radeon8060s".to_string()),
            matched_profile_name: Some("AMD Radeon 8060S Graphics".to_string()),
            vk_visible_device: 0,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![],
            secondary_igpu_name: None,
            physical_cpu_cores: 16,
            total_ram_mib: 131_072,
            total_vram_mib: 65_536,
            used_vram_mib: 2_048,
        };
        let p_halo = plan_llm_configuration_for_gpu(&sample_27b, &halo_ctx, 8192, 512);
        assert!(p_halo.is_apu_unified_memory);
        assert!(!p_halo.has_secondary_igpu);
        assert!(p_halo.disabled_gpu_indices.is_empty());
        assert!(p_halo.kv_budgets[0].fits_in_free_vram);

        let small_uma_apu = LlmTargetGpuContext {
            total_vram_mib: 512,
            total_ram_mib: 65_536,
            used_vram_mib: 256,
            ..halo_ctx
        };
        let p_apu_scaled = plan_llm_configuration_for_gpu(&sample_27b, &small_uma_apu, 4096, 512);
        assert_eq!(p_apu_scaled.total_vram_mib, 32_768);
        assert!(p_apu_scaled
            .warnings
            .iter()
            .any(|w| w.contains("Variable Graphics Memory")));

        // 6. RX 5700 XT (RDNA1, 8 GiB, Vulkan-only + ROCm warning)
        let rdna1_ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "AMD Radeon RX 5700 XT".to_string(),
            gfx_target: "gfx1010".to_string(),
            architecture: GpuArchitecture::Rdna1,
            discrete: true,
            matched_profile_id: None,
            matched_profile_name: None,
            vk_visible_device: 0,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![],
            secondary_igpu_name: None,
            physical_cpu_cores: 8,
            total_ram_mib: 16_384,
            total_vram_mib: 8_176,
            used_vram_mib: 900,
        };
        let p_5700 = plan_llm_configuration_for_gpu(&sample_27b, &rdna1_ctx, 4096, 512);
        assert_eq!(p_5700.gpu_architecture, "RDNA1");
        assert!(p_5700
            .warnings
            .iter()
            .any(|w| w.contains("RDNA1") && w.contains("ROCm/HIP")));
        assert!(!p_5700
            .recommended_profiles
            .iter()
            .any(|p| p.backend == "hip"));
    }

    #[test]
    fn lm_studio_rollback_previews_and_restores_bak_safely() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rdna_forge_lms_rollback_{uniq}"));
        let internal = root.join(".internal");
        let model_dir = internal
            .join("user-concrete-model-default-config")
            .join("vendor");
        std::fs::create_dir_all(&model_dir).unwrap();

        let prefs = internal.join("backend-preferences-v2.json");
        let prefs_bak = internal.join("backend-preferences-v2.json.bak");
        std::fs::write(&prefs, r#"{"patched":true}"#).unwrap();
        std::fs::write(&prefs_bak, r#"{"original":true}"#).unwrap();

        let model_cfg = model_dir.join("qwen3.gguf.json");
        let model_bak = model_dir.join("qwen3.gguf.json.bak");
        std::fs::write(&model_cfg, r#"{"ctx":4096}"#).unwrap();
        std::fs::write(&model_bak, r#"{"ctx":8192}"#).unwrap();

        // Dry-run must preview both restorations without touching files on disk.
        let dry = rollback_lm_studio_config_in_dir(&root, None, true).unwrap();
        assert!(!dry.applied);
        assert_eq!(dry.actions.len(), 2);
        assert!(dry.actions.iter().all(|a| !a.restored));
        assert_eq!(
            std::fs::read_to_string(&prefs).unwrap(),
            r#"{"patched":true}"#
        );
        assert_eq!(
            std::fs::read_to_string(&model_cfg).unwrap(),
            r#"{"ctx":4096}"#
        );

        // Apply must copy .json.bak contents back over .json cleanly.
        let applied = rollback_lm_studio_config_in_dir(&root, None, false).unwrap();
        assert!(applied.applied);
        assert_eq!(applied.actions.len(), 2);
        assert!(applied.actions.iter().all(|a| a.restored));
        assert_eq!(
            std::fs::read_to_string(&prefs).unwrap(),
            r#"{"original":true}"#
        );
        assert_eq!(
            std::fs::read_to_string(&model_cfg).unwrap(),
            r#"{"ctx":8192}"#
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
