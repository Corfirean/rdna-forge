pub mod llama;
pub mod llm_optimizer;
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
    gguf_file_type_name, inspect_gguf_model, plan_llm_configuration, EnvAndOllamaPatchResult,
    EnvVarChange, GgufModelMetadata, KvCacheMemoryBudget, KvCacheQuant, LlmAutotuneCandidateResult,
    LlmAutotuneConfig, LlmAutotuneReport, LlmOptimizationPlan, LlmTweakProfile,
    LmStudioPatchResult, LocalLlmAssets,
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
            build_lm_studio_model_config_json(Some(existing), &profile, 65, 4096, true);
        assert_eq!(ratio_str, "1.00 (100% GPU)");
        assert!(merged_json.contains("custom-jinja"));
        assert!(merged_json.contains("\"disabledGpus\": [\n            1\n          ]"));
        assert!(merged_json.contains("\"key\": \"llm.load.contextLength\""));
        assert!(merged_json.contains("\"value\": 4096"));
    }
}
