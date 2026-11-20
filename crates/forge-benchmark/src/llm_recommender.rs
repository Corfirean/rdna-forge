use forge_core::{match_profile_for_device, ForgeError, ForgeResult, GpuArchitecture};
use forge_device::discover_all;
use forge_profiler::snapshot_all_gpus;
use serde::{Deserialize, Serialize};

/// Suitability tier of a popular GGUF model on the target AMD GPU's real VRAM budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFitVerdict {
    IdealFreeVram,
    FitsWddmPriority,
    FitsMoeOffload,
    TooLarge,
}

impl ModelFitVerdict {
    pub const fn label(self) -> &'static str {
        match self {
            Self::IdealFreeVram => "IDEAL (Free VRAM)",
            Self::FitsWddmPriority => "FITS (WDDM Priority=1)",
            Self::FitsMoeOffload => "FITS (MoE -ncmoe)",
            Self::TooLarge => "TOO LARGE",
        }
    }

    const fn rank(self) -> u8 {
        match self {
            Self::IdealFreeVram => 0,
            Self::FitsWddmPriority => 1,
            Self::FitsMoeOffload => 2,
            Self::TooLarge => 3,
        }
    }
}

/// Curated GGUF model sizing entry evaluated against the selected GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub name: String,
    pub family: String,
    pub params_label: String,
    pub quant: String,
    pub is_moe: bool,
    pub weights_mib: u64,
    pub kv_q8_mib: u64,
    pub total_required_mib: u64,
    pub verdict: ModelFitVerdict,
    pub recommended_max_context: u32,
    pub suggested_backend: String,
    pub notes: String,
}

/// Complete VRAM-aware popular local LLM compatibility matrix for a single AMD GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuModelRecommendationReport {
    pub gpu_index: usize,
    pub gpu_name: String,
    pub gfx_target: String,
    pub gpu_architecture: String,
    pub matched_profile_id: Option<String>,
    pub total_vram_mib: u64,
    pub used_vram_mib: u64,
    pub free_vram_mib: u64,
    pub priority_budget_mib: u64,
    pub is_apu_unified_memory: bool,
    pub target_context_tokens: u32,
    pub recommendations: Vec<ModelRecommendation>,
}

struct CatalogSpec {
    name: &'static str,
    family: &'static str,
    params_label: &'static str,
    quant: &'static str,
    is_moe: bool,
    weights_mib: u64,
    kv_mib_per_4k_q8: u64,
    scratch_mib: u64,
    notes: &'static str,
}

const POPULAR_LOCAL_LLM_CATALOG: &[CatalogSpec] = &[
    CatalogSpec {
        name: "Qwen3-4B-Instruct",
        family: "Qwen 3",
        params_label: "4B",
        quant: "Q8_0",
        is_moe: false,
        weights_mib: 4_280,
        kv_mib_per_4k_q8: 288,
        scratch_mib: 256,
        notes: "Fast lightweight coding & tool-use model; huge context headroom on 8 GiB cards.",
    },
    CatalogSpec {
        name: "Qwen3-8B",
        family: "Qwen 3",
        params_label: "8B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 4_980,
        kv_mib_per_4k_q8: 576,
        scratch_mib: 256,
        notes: "All-round daily driver for 8-12 GiB GPUs with 16K-32K Q8_0 KV context.",
    },
    CatalogSpec {
        name: "Llama-3.1-8B-Instruct",
        family: "Llama 3.1",
        params_label: "8B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 4_920,
        kv_mib_per_4k_q8: 512,
        scratch_mib: 256,
        notes: "Low VRAM footprint with mature Vulkan/ROCm throughput across RDNA1-RDNA4.",
    },
    CatalogSpec {
        name: "Gemma-3-12B-IT",
        family: "Gemma 3",
        params_label: "12B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 7_450,
        kv_mib_per_4k_q8: 640,
        scratch_mib: 256,
        notes: "Sweet spot for 12 GiB GPUs (RX 6700 XT / RX 7700 XT) and crisp 16 GiB margin.",
    },
    CatalogSpec {
        name: "Qwen3-14B",
        family: "Qwen 3",
        params_label: "14B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 8_960,
        kv_mib_per_4k_q8: 640,
        scratch_mib: 256,
        notes: "Strong reasoning & bilingual quality; fits cleanly in 12 GiB PRIO=1 or 16 GiB Free VRAM.",
    },
    CatalogSpec {
        name: "DeepSeek-R1-Distill-Qwen-14B",
        family: "DeepSeek-R1",
        params_label: "14B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 8_960,
        kv_mib_per_4k_q8: 768,
        scratch_mib: 256,
        notes: "Chain-of-thought reasoning; reserve extra KV headroom for longer thinking traces.",
    },
    CatalogSpec {
        name: "Mistral-Small-3.1-24B-Instruct",
        family: "Mistral",
        params_label: "24B",
        quant: "IQ4_XS",
        is_moe: false,
        weights_mib: 12_950,
        kv_mib_per_4k_q8: 768,
        scratch_mib: 384,
        notes: "Fast 24B workhorse on 16 GiB GPUs (use GGML_VK_ENABLE_MEMORY_PRIORITY=1 on WDDM).",
    },
    CatalogSpec {
        name: "Gemma-3-27B-IT",
        family: "Gemma 3",
        params_label: "27B",
        quant: "IQ4_XS",
        is_moe: false,
        weights_mib: 13_740,
        kv_mib_per_4k_q8: 448,
        scratch_mib: 384,
        notes: "Peak dense 27B quality on 16 GiB GPUs (sliding-window KV); pair with PRIO=1 + Q8_0/Q4_0 KV.",
    },
    CatalogSpec {
        name: "Gemma-3-27B-IT",
        family: "Gemma 3",
        params_label: "27B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 16_450,
        kv_mib_per_4k_q8: 448,
        scratch_mib: 384,
        notes: "Needs 20-24+ GiB VRAM (RX 7900 XT/XTX, AI PRO R9700, or Strix Halo 64 GiB).",
    },
    CatalogSpec {
        name: "Qwen3-30B-A3B",
        family: "Qwen 3 MoE",
        params_label: "30B (3B active)",
        quant: "Q4_K_M",
        is_moe: true,
        weights_mib: 17_700,
        kv_mib_per_4k_q8: 512,
        scratch_mib: 384,
        notes: "Fits 100% on 24 GiB+ VRAM or runs at high tok/s on 12-16 GiB GPUs via -ncmoe expert offload.",
    },
    CatalogSpec {
        name: "Qwen3-32B",
        family: "Qwen 3",
        params_label: "32B",
        quant: "IQ4_XS",
        is_moe: false,
        weights_mib: 17_100,
        kv_mib_per_4k_q8: 1_024,
        scratch_mib: 384,
        notes: "Ideal dense 32B coding/reasoning fit on 20-24 GiB GPUs (RX 7900 XT/XTX).",
    },
    CatalogSpec {
        name: "Qwen3-32B",
        family: "Qwen 3",
        params_label: "32B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 19_450,
        kv_mib_per_4k_q8: 1_024,
        scratch_mib: 384,
        notes: "Comfortable full-VRAM fit on 24-48 GiB cards with 16K+ context.",
    },
    CatalogSpec {
        name: "DeepSeek-R1-Distill-Qwen-32B",
        family: "DeepSeek-R1",
        params_label: "32B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 19_450,
        kv_mib_per_4k_q8: 1_024,
        scratch_mib: 384,
        notes: "Dense 32B reasoning flagship for RX 7900 XTX (24 GiB), PRO R9700 (32 GiB), and Strix Halo.",
    },
    CatalogSpec {
        name: "Llama-3.3-70B-Instruct",
        family: "Llama 3.3",
        params_label: "70B",
        quant: "IQ4_XS",
        is_moe: false,
        weights_mib: 37_400,
        kv_mib_per_4k_q8: 1_280,
        scratch_mib: 512,
        notes: "Requires 48+ GiB Radeon PRO W7900 or high-UMA Strix Halo APU (64+ GiB pool).",
    },
    CatalogSpec {
        name: "Llama-3.3-70B-Instruct",
        family: "Llama 3.3",
        params_label: "70B",
        quant: "Q4_K_M",
        is_moe: false,
        weights_mib: 40_950,
        kv_mib_per_4k_q8: 1_280,
        scratch_mib: 512,
        notes: "Full 70B Q4_K_M fits natively on 48 GiB PRO W7900 and 64+ GiB Strix Halo unified APUs.",
    },
];

fn suggested_backend_for_arch(arch: GpuArchitecture) -> String {
    match arch {
        GpuArchitecture::Rdna1 => "Vulkan (-fa on -ctk q8_0)".to_string(),
        GpuArchitecture::Rdna2 => "Vulkan (PRIO=1) / ROCm (q4_0 KV long ctx)".to_string(),
        GpuArchitecture::Rdna3 | GpuArchitecture::Rdna35 | GpuArchitecture::Rdna4 => {
            "ROCm WMMA + hipBLASLt / Vulkan (PRIO=1)".to_string()
        }
        _ => "Vulkan (PRIO=1)".to_string(),
    }
}

/// Evaluates the curated local LLM catalog against explicit GPU VRAM/RAM parameters.
#[allow(clippy::too_many_arguments)]
pub fn recommend_models_for_gpu_budget(
    gpu_index: usize,
    gpu_name: &str,
    gfx_target: &str,
    architecture: GpuArchitecture,
    matched_profile_id: Option<String>,
    discrete: bool,
    total_ram_mib: u64,
    raw_total_vram_mib: u64,
    raw_used_vram_mib: u64,
    target_context_tokens: u32,
    safety_margin_mib: u64,
) -> GpuModelRecommendationReport {
    let target_ctx = target_context_tokens.clamp(1024, 131_072);
    let is_apu_unified_memory = !discrete;
    let total_vram_mib =
        if is_apu_unified_memory && raw_total_vram_mib < 2_048 && total_ram_mib >= 8_192 {
            (total_ram_mib / 2).max(raw_total_vram_mib)
        } else {
            raw_total_vram_mib
        };
    let used_vram_mib = raw_used_vram_mib.min(total_vram_mib);
    let free_vram_mib = total_vram_mib.saturating_sub(used_vram_mib);
    let usable_free_mib = free_vram_mib.saturating_sub(safety_margin_mib);

    let baseline_os_vram_mib = (if total_vram_mib <= 8_192 {
        768u64
    } else if total_vram_mib <= 12_288 {
        950u64
    } else {
        1_150u64
    })
    .min(total_vram_mib / 4);

    let priority_budget_mib = total_vram_mib
        .saturating_sub(baseline_os_vram_mib + safety_margin_mib / 2)
        .max(usable_free_mib);

    let backend_label = suggested_backend_for_arch(architecture);
    let ctx_scale = (f64::from(target_ctx) / 4096.0).max(0.25);

    let mut recommendations: Vec<ModelRecommendation> = POPULAR_LOCAL_LLM_CATALOG
        .iter()
        .map(|spec| {
            let kv_q8_mib = ((spec.kv_mib_per_4k_q8 as f64) * ctx_scale).ceil() as u64;
            let total_required_mib = spec.weights_mib + kv_q8_mib + spec.scratch_mib;

            // MoE expert offload keeps ~25% (attention + shared weights + active buffers) on GPU.
            let moe_resident_weights_mib = (spec.weights_mib / 4).max(1);
            let moe_vram_mib = moe_resident_weights_mib + kv_q8_mib + spec.scratch_mib;
            let moe_fits_system_ram =
                total_ram_mib == 0 || total_ram_mib >= spec.weights_mib.saturating_add(4_096);

            let verdict = if total_required_mib <= usable_free_mib {
                ModelFitVerdict::IdealFreeVram
            } else if total_required_mib <= priority_budget_mib {
                ModelFitVerdict::FitsWddmPriority
            } else if spec.is_moe && moe_vram_mib <= priority_budget_mib && moe_fits_system_ram {
                ModelFitVerdict::FitsMoeOffload
            } else {
                ModelFitVerdict::TooLarge
            };

            let resident_base_mib = if verdict == ModelFitVerdict::FitsMoeOffload {
                moe_resident_weights_mib
            } else {
                spec.weights_mib
            };

            let recommended_max_context = if verdict == ModelFitVerdict::TooLarge {
                0
            } else {
                let mut best_ctx = 2048u32;
                let mut any_fit = false;
                for &candidate_ctx in &[32_768u32, 16_384, 8_192, 4_096, 2_048] {
                    let kv_c = ((spec.kv_mib_per_4k_q8 as f64)
                        * (f64::from(candidate_ctx) / 4096.0))
                        .ceil() as u64;
                    if resident_base_mib + kv_c + spec.scratch_mib <= priority_budget_mib {
                        best_ctx = candidate_ctx;
                        any_fit = true;
                        break;
                    }
                }
                if any_fit {
                    best_ctx
                } else {
                    target_ctx.min(2048)
                }
            };

            ModelRecommendation {
                name: spec.name.to_string(),
                family: spec.family.to_string(),
                params_label: spec.params_label.to_string(),
                quant: spec.quant.to_string(),
                is_moe: spec.is_moe,
                weights_mib: spec.weights_mib,
                kv_q8_mib,
                total_required_mib,
                verdict,
                recommended_max_context,
                suggested_backend: backend_label.clone(),
                notes: spec.notes.to_string(),
            }
        })
        .collect();

    recommendations.sort_by_key(|r| {
        let rank = r.verdict.rank();
        let weight_order = if rank < 3 {
            u64::MAX - r.weights_mib
        } else {
            r.weights_mib
        };
        (rank, weight_order)
    });

    GpuModelRecommendationReport {
        gpu_index,
        gpu_name: gpu_name.to_string(),
        gfx_target: gfx_target.to_string(),
        gpu_architecture: architecture.to_string(),
        matched_profile_id,
        total_vram_mib,
        used_vram_mib,
        free_vram_mib,
        priority_budget_mib,
        is_apu_unified_memory,
        target_context_tokens: target_ctx,
        recommendations,
    }
}

/// Recommends popular local GGUF models tailored to the live discovered GPU at `device_index`
/// (or primary discrete GPU when `None`).
pub fn recommend_models_for_device(
    device_index: Option<usize>,
    target_context_tokens: u32,
    safety_margin_mib: u64,
) -> ForgeResult<GpuModelRecommendationReport> {
    let snapshot = discover_all();
    let gpu = match device_index {
        Some(idx) => snapshot.gpus.get(idx).ok_or_else(|| {
            ForgeError::DeviceNotFound(format!("GPU device index {idx} not found"))
        })?,
        None => snapshot
            .gpus
            .iter()
            .find(|g| g.discrete)
            .or_else(|| snapshot.gpus.first())
            .ok_or_else(|| ForgeError::DeviceNotFound("no GPU devices discovered".to_string()))?,
    };

    let matched = match_profile_for_device(gpu);
    let gfx_target = gpu
        .gfx_target
        .as_ref()
        .map(|t| t.base.clone())
        .or_else(|| {
            matched
                .as_ref()
                .map(|m| m.profile.device.gfx_target.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let telemetry = snapshot_all_gpus().ok().unwrap_or_default();
    let active_tel = telemetry
        .iter()
        .find(|s| s.pci_device_id == Some(gpu.device_id))
        .or_else(|| telemetry.iter().find(|s| s.discrete == gpu.discrete))
        .or_else(|| telemetry.first());

    let total_vram_mib = active_tel
        .and_then(|s| s.total_vram_mib.map(u64::from))
        .or_else(|| (gpu.vram_bytes > 0).then_some(gpu.vram_bytes / (1024 * 1024)))
        .or_else(|| {
            matched
                .as_ref()
                .map(|m| m.profile.memory.vram_bytes / (1024 * 1024))
        })
        .unwrap_or(8_192);
    let used_vram_mib = active_tel
        .and_then(|s| s.vram_used_mib.map(|v| v.max(0) as u64))
        .unwrap_or(1_024)
        .min(total_vram_mib);
    let total_ram_mib = snapshot.system.total_ram_bytes / (1024 * 1024);

    Ok(recommend_models_for_gpu_budget(
        gpu.index,
        &gpu.name,
        &gfx_target,
        gpu.architecture,
        matched.map(|m| m.id),
        gpu.discrete,
        total_ram_mib,
        total_vram_mib,
        used_vram_mib,
        target_context_tokens,
        safety_margin_mib,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_verdict<'a>(
        rep: &'a GpuModelRecommendationReport,
        name: &str,
        quant: &str,
    ) -> &'a ModelRecommendation {
        rep.recommendations
            .iter()
            .find(|r| r.name == name && r.quant == quant)
            .unwrap_or_else(|| panic!("missing catalog entry {name} ({quant})"))
    }

    #[test]
    fn model_recommender_classifies_8gb_16gb_24gb_48gb_and_strix_halo() {
        // 8 GiB RX 6600 XT: 8B Q4_K_M fits free VRAM, 27B/70B too large
        let rx6600 = recommend_models_for_gpu_budget(
            0,
            "AMD Radeon RX 6600 XT",
            "gfx1032",
            GpuArchitecture::Rdna2,
            Some("rx6600xt".to_string()),
            true,
            32_768,
            8_176,
            900,
            4096,
            512,
        );
        assert_eq!(
            find_verdict(&rx6600, "Qwen3-8B", "Q4_K_M").verdict,
            ModelFitVerdict::IdealFreeVram
        );
        assert_eq!(
            find_verdict(&rx6600, "Gemma-3-27B-IT", "IQ4_XS").verdict,
            ModelFitVerdict::TooLarge
        );

        // 16 GiB RX 6900 XT with 3 GiB desktop usage: 14B Ideal, 27B IQ4_XS FitsWddmPriority, 30B-A3B FitsMoeOffload
        let rx6900 = recommend_models_for_gpu_budget(
            0,
            "AMD Radeon RX 6900 XT",
            "gfx1030",
            GpuArchitecture::Rdna2,
            Some("rx6900xt".to_string()),
            true,
            32_768,
            16_368,
            3_200,
            4096,
            512,
        );
        assert_eq!(
            find_verdict(&rx6900, "Qwen3-14B", "Q4_K_M").verdict,
            ModelFitVerdict::IdealFreeVram
        );
        assert_eq!(
            find_verdict(&rx6900, "Gemma-3-27B-IT", "IQ4_XS").verdict,
            ModelFitVerdict::FitsWddmPriority
        );
        assert_eq!(
            find_verdict(&rx6900, "Qwen3-30B-A3B", "Q4_K_M").verdict,
            ModelFitVerdict::FitsMoeOffload
        );
        assert_eq!(
            find_verdict(&rx6900, "Llama-3.3-70B-Instruct", "Q4_K_M").verdict,
            ModelFitVerdict::TooLarge
        );

        // 24 GiB RX 7900 XTX: Qwen3-32B Q4_K_M fits cleanly; hipBLASLt backend suggested
        let rx7900 = recommend_models_for_gpu_budget(
            0,
            "AMD Radeon RX 7900 XTX",
            "gfx1100",
            GpuArchitecture::Rdna3,
            Some("rx7900xtx".to_string()),
            true,
            65_536,
            24_560,
            1_500,
            4096,
            512,
        );
        let q32 = find_verdict(&rx7900, "Qwen3-32B", "Q4_K_M");
        assert_eq!(q32.verdict, ModelFitVerdict::IdealFreeVram);
        assert!(q32.suggested_backend.contains("hipBLASLt"));

        // 48 GiB PRO W7900 & 64 GiB Strix Halo: Llama 3.3 70B Q4_K_M fits!
        let w7900 = recommend_models_for_gpu_budget(
            0,
            "AMD Radeon PRO W7900",
            "gfx1100",
            GpuArchitecture::Rdna3,
            Some("pro_w7900".to_string()),
            true,
            65_536,
            49_136,
            1_000,
            4096,
            512,
        );
        assert_eq!(
            find_verdict(&w7900, "Llama-3.3-70B-Instruct", "Q4_K_M").verdict,
            ModelFitVerdict::IdealFreeVram
        );

        let strix_halo = recommend_models_for_gpu_budget(
            0,
            "AMD Radeon 8060S Graphics",
            "gfx1151",
            GpuArchitecture::Rdna35,
            Some("radeon8060s".to_string()),
            false,
            131_072,
            512,
            256,
            4096,
            512,
        );
        assert!(strix_halo.is_apu_unified_memory);
        assert_eq!(strix_halo.total_vram_mib, 65_536);
        assert_eq!(
            find_verdict(&strix_halo, "Llama-3.3-70B-Instruct", "Q4_K_M").verdict,
            ModelFitVerdict::IdealFreeVram
        );
    }
}
