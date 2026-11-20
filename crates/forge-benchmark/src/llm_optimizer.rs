use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use forge_core::{match_profile_for_device, ForgeError, ForgeResult, GpuArchitecture};
use forge_device::discover_all;
use forge_profiler::{query_top_vram_processes, snapshot_all_gpus};
use serde::{Deserialize, Serialize};

use crate::gguf_layout::{
    inferred_block_count, read_tensor_records, GgufTensorLayout, DEFAULT_GGUF_ALIGNMENT,
};
use crate::llama::parse_llama_bench_json;
use crate::memory::peak_process_rss_bytes;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_INLINE_ARRAY_LEN: u64 = 32;

/// Parsed metadata and architectural dimensions from a GGUF v2/v3 model header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufModelMetadata {
    pub file_path: String,
    pub file_name: String,
    pub file_size_bytes: u64,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub architecture: String,
    pub model_name: Option<String>,
    pub size_label: Option<String>,
    pub quantized_by: Option<String>,
    pub file_type_code: Option<u32>,
    pub quantization_name: String,
    pub block_count: u32,
    pub nextn_predict_layers: u32,
    pub declared_context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: Option<u32>,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub full_attention_interval: Option<u32>,
    pub ssm_state_size: Option<u32>,
    pub ssm_inner_size: Option<u32>,
    pub ssm_group_count: Option<u32>,
    #[serde(default)]
    pub expert_count: Option<u32>,
    #[serde(default)]
    pub expert_used_count: Option<u32>,
    #[serde(default)]
    pub expert_feed_forward_length: Option<u32>,
    #[serde(default)]
    pub is_moe: bool,
    /// Mean repeating-layer tensor size. Derived from `tensor_layout` when present; otherwise the
    /// legacy `file_size_bytes / block_count` approximation (see [`GgufModelMetadata::tensor_layout_basis`]).
    pub bytes_per_layer: u64,
    /// Exact per-role tensor accounting parsed from the GGUF tensor-info section, if available.
    #[serde(default)]
    pub tensor_layout: Option<GgufTensorLayout>,
    /// `split.count` when the model is a multi-file GGUF (this file holds only one shard).
    #[serde(default)]
    pub split_count: Option<u32>,
    /// Metadata keys that were missing and replaced by a guess; any plan using them is qualified.
    #[serde(default)]
    pub assumed_defaults: Vec<String>,
}

impl GgufModelMetadata {
    /// `"tensor_metadata"` when memory estimates come from parsed tensor sizes, else
    /// `"file_size_approximation"` (uniform file-size / block-count split; not an exact figure).
    pub fn tensor_layout_basis(&self) -> &'static str {
        if self.tensor_layout.is_some() {
            "tensor_metadata"
        } else {
            "file_size_approximation"
        }
    }

    /// Human-readable qualifications that limit how exact the weight estimate is.
    pub fn tensor_layout_notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        match &self.tensor_layout {
            None => notes.push(
                "No tensor metadata was available; weight memory is approximated as file size split evenly across layers and may be wrong for embeddings, output tensors, uneven layers or experts."
                    .to_string(),
            ),
            Some(l) => {
                if l.unknown_type_tensors > 0 {
                    notes.push(format!(
                        "{} tensor(s) use an unknown GGML type; their sizes are upper bounds taken from data offsets.",
                        l.unknown_type_tensors
                    ));
                }
                if l.extra_layer_bytes > 0 {
                    notes.push(
                        "Some blk.N tensors have an index beyond the declared block_count; they are assumed GPU-resident only at full offload."
                            .to_string(),
                    );
                }
            }
        }
        if let Some(n) = self.split_count.filter(|&n| n > 1) {
            notes.push(format!(
                "Model is split across {n} GGUF files; only this file's tensors are counted, so weight memory is under-reported."
            ));
        }
        for d in &self.assumed_defaults {
            notes.push(format!("Assumed default: {d}."));
        }
        notes
    }
}

/// Supported `llama.cpp` KV cache quantization types (`-ctk` / `-ctv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheQuant {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheQuant {
    pub const fn as_cli_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    pub const fn bytes_per_element(self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8_0 => 34.0 / 32.0, // 32 int8 + 1 fp16 scale = 34 B per 32 elems
            Self::Q4_0 => 18.0 / 32.0, // 16 byte nibbles + 1 fp16 scale = 18 B per 32 elems
        }
    }

    pub const fn requires_flash_attn_for_v(self) -> bool {
        !matches!(self, Self::F16)
    }
}

/// Memory budget breakdown for a specific KV cache quantization mode at a target context length.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheMemoryBudget {
    pub kv_quant: String,
    pub requires_flash_attn: bool,
    pub context_tokens: u32,
    /// Estimated GPU-resident weight memory at full offload (MiB). Computed from GGUF tensor
    /// sizes when `weights_estimate_basis == "tensor_metadata"`, otherwise it is the whole file
    /// size and therefore an upper-bound approximation. Never an exact allocator prediction.
    pub model_weights_mib: u64,
    /// On-disk model file size (MiB), reported separately from the GPU weight estimate.
    #[serde(default)]
    pub model_file_mib: u64,
    /// `"tensor_metadata"` or `"file_size_approximation"`.
    #[serde(default)]
    pub weights_estimate_basis: String,
    pub kv_and_state_mib: u64,
    pub compute_scratch_mib: u64,
    pub total_required_mib: u64,
    pub fits_in_free_vram: bool,
    pub fits_with_wddm_memory_priority: bool,
    pub strict_zero_spill_ngl: u32,
    pub memory_priority_ngl: u32,
    #[serde(default)]
    pub recommended_n_cpu_moe: u32,
}

/// A concrete, hardware-tested `llama.cpp` execution profile for AMD RDNA2 (`gfx1030`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmTweakProfile {
    pub id: String,
    pub title: String,
    pub backend: String,
    pub device_selector: String,
    pub n_gpu_layers: u32,
    #[serde(default)]
    pub n_cpu_moe: u32,
    pub split_mode: String,
    pub flash_attn: bool,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u32,
    pub env_vars: Vec<(String, String)>,
    pub rationale: String,
    pub llama_server_cmd: String,
}

/// Per-process VRAM consumption impact relative to the target LLM model's layer size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessVramImpact {
    pub pid: u32,
    pub process_name: String,
    pub dedicated_vram_mib: u64,
    pub shared_vram_mib: u64,
    pub recoverable_gpu_layers: u32,
}

/// Complete static analysis, VRAM anti-spill budget, and hardware-aware tweak recommendations for a GGUF model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmOptimizationPlan {
    pub model: GgufModelMetadata,
    #[serde(default)]
    pub gpu_index: usize,
    pub gpu_name: String,
    pub gfx_target: String,
    #[serde(default)]
    pub gpu_architecture: String,
    #[serde(default)]
    pub matched_profile_id: Option<String>,
    #[serde(default)]
    pub matched_profile_name: Option<String>,
    #[serde(default)]
    pub vk_visible_device: usize,
    #[serde(default)]
    pub hip_visible_device: usize,
    #[serde(default)]
    pub disabled_gpu_indices: Vec<usize>,
    #[serde(default)]
    pub is_apu_unified_memory: bool,
    pub has_secondary_igpu: bool,
    pub secondary_igpu_name: Option<String>,
    pub physical_cpu_cores: u32,
    pub total_vram_mib: u64,
    pub used_vram_mib: u64,
    pub free_vram_mib: u64,
    pub safety_margin_mib: u64,
    pub target_context_tokens: u32,
    pub recoverable_layers_if_apps_closed: u32,
    #[serde(default)]
    pub top_vram_processes: Vec<ProcessVramImpact>,
    pub warnings: Vec<String>,
    pub kv_budgets: Vec<KvCacheMemoryBudget>,
    pub recommended_profiles: Vec<LlmTweakProfile>,
    pub lm_studio_tips: Vec<String>,
}

/// Target GPU properties supplied to the pure multi-GPU LLM planner.
#[derive(Debug, Clone)]
pub struct LlmTargetGpuContext {
    pub gpu_index: usize,
    pub gpu_name: String,
    pub gfx_target: String,
    pub architecture: GpuArchitecture,
    pub discrete: bool,
    pub matched_profile_id: Option<String>,
    pub matched_profile_name: Option<String>,
    pub vk_visible_device: usize,
    pub hip_visible_device: usize,
    pub disabled_gpu_indices: Vec<usize>,
    pub secondary_igpu_name: Option<String>,
    pub physical_cpu_cores: u32,
    pub total_ram_mib: u64,
    pub total_vram_mib: u64,
    pub used_vram_mib: u64,
}

/// Auto-discovered local GGUF models and `llama.cpp` binaries on the host system.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalLlmAssets {
    pub gguf_models: Vec<String>,
    pub vulkan_bench_bin: Option<String>,
    pub hip_bench_bin: Option<String>,
    pub vulkan_server_bin: Option<String>,
    pub hip_server_bin: Option<String>,
    pub rocm_lib_dir: Option<String>,
}

/// Parameters for running the live empirical `llama-bench` autotuner on a GGUF model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmAutotuneConfig {
    pub model_path: PathBuf,
    #[serde(default)]
    pub device_index: Option<usize>,
    pub vulkan_bin: Option<PathBuf>,
    pub hip_bin: Option<PathBuf>,
    pub rocm_lib_dir: Option<PathBuf>,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub context_depth: u32,
    pub reps: u32,
}

/// Empirical result for a single candidate profile tested by `autotune_llm_model`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmAutotuneCandidateResult {
    pub profile_id: String,
    pub title: String,
    pub backend: String,
    pub n_gpu_layers: u32,
    pub flash_attn: bool,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub env_summary: String,
    pub succeeded: bool,
    pub pp_tok_s: Option<f64>,
    pub tg_tok_s: Option<f64>,
    pub tg_speedup_vs_baseline: Option<f64>,
    pub pp_speedup_vs_baseline: Option<f64>,
    pub llama_server_cmd: String,
    pub error: Option<String>,
}

/// Full report produced by `autotune_llm_model`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmAutotuneReport {
    pub plan: LlmOptimizationPlan,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub context_depth: u32,
    pub candidates: Vec<LlmAutotuneCandidateResult>,
    pub best_decode_profile_id: Option<String>,
    pub best_prefill_profile_id: Option<String>,
    pub best_balanced_profile_id: Option<String>,
    /// Profile every speedup is relative to. `None` when no designated baseline ran successfully
    /// (speedups are then absent rather than measured against an arbitrary candidate).
    #[serde(default)]
    pub baseline_profile_id: Option<String>,
    /// Human-readable definition of the balanced score used for `best_balanced_profile_id`.
    #[serde(default)]
    pub balanced_score_formula: String,
    pub peak_rss_bytes: Option<u64>,
}

/// Result of previewing (`dry_run = true`) or writing (`dry_run = false`) optimized LM Studio config files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LmStudioPatchResult {
    pub applied: bool,
    pub profile_id: String,
    pub profile_title: String,
    pub model_config_path: String,
    pub model_config_backup_path: Option<String>,
    pub backend_prefs_path: String,
    pub backend_prefs_backup_path: Option<String>,
    pub selected_lmstudio_backend: String,
    pub offload_ratio: String,
    pub context_length: u32,
    pub flash_attention: bool,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub cpu_threads: u32,
    pub disabled_igpu: bool,
    pub patched_model_json_preview: String,
    pub patched_backend_json_preview: String,
}

/// Single Windows User Environment (`HKCU\Environment`) variable change record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvVarChange {
    pub key: String,
    pub previous_value: Option<String>,
    pub new_value: String,
    #[serde(default)]
    pub purpose: String,
}

/// Result of previewing (`dry_run = true`) or applying (`dry_run = false`) Windows & Ollama RDNA2 environment tweaks and `Modelfile`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvAndOllamaPatchResult {
    pub applied: bool,
    pub env_changes: Vec<EnvVarChange>,
    pub ollama_modelfile_content: String,
    pub ollama_modelfile_path: Option<String>,
    pub ollama_create_cmd: String,
}

/// Environment variable names in `HKCU\Environment` managed by RDNA Forge LLM tweaks.
pub const RDNA_FORGE_MANAGED_ENV_KEYS: &[&str] = &[
    "GGML_VK_ENABLE_MEMORY_PRIORITY",
    "GGML_VK_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCBLAS_USE_HIPBLASLT",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_NUM_PARALLEL",
];

/// Single LM Studio JSON config file restore action from its `.bak` backup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackFileAction {
    pub target_path: String,
    pub backup_path: String,
    pub restored: bool,
}

/// Result of previewing (`dry_run = true`) or restoring (`dry_run = false`) LM Studio `.json.bak` backups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LmStudioRollbackResult {
    pub applied: bool,
    pub actions: Vec<RollbackFileAction>,
}

/// Status of a single RDNA Forge-managed `HKCU\Environment` variable during rollback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvRollbackItem {
    pub key: String,
    pub current_value: Option<String>,
    pub removed: bool,
}

/// Result of previewing (`dry_run = true`) or deleting (`dry_run = false`) RDNA Forge `HKCU\Environment` variables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvRollbackResult {
    pub applied: bool,
    pub keys: Vec<EnvRollbackItem>,
}

#[derive(Debug, Clone)]
enum GgufValue {
    UInt(u64),
    Int(i64),
    Float,
    Bool,
    Str(String),
    SkippedArray,
}

impl GgufValue {
    fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::UInt(v) => u32::try_from(v).ok(),
            Self::Int(v) if v >= 0 => u32::try_from(v).ok(),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64<R: Read>(r: &mut R) -> std::io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_gguf_string<R: Read + Seek>(r: &mut R, max_keep: usize) -> ForgeResult<String> {
    let len = read_u64(r).map_err(ForgeError::Io)?;
    if len > 64 * 1024 * 1024 {
        return Err(ForgeError::ParseError(format!(
            "GGUF string length {len} exceeds 64 MiB safety limit"
        )));
    }
    if (len as usize) <= max_keep {
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf).map_err(ForgeError::Io)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    } else {
        r.seek(SeekFrom::Current(len as i64))
            .map_err(ForgeError::Io)?;
        Ok(format!("<truncated {len}B string>"))
    }
}

fn scalar_type_size(vtype: u32) -> Option<u64> {
    match vtype {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_gguf_value<R: Read + Seek>(r: &mut R, vtype: u32) -> ForgeResult<GgufValue> {
    match vtype {
        0 => Ok(GgufValue::UInt(u64::from(
            read_u8(r).map_err(ForgeError::Io)?,
        ))),
        1 => Ok(GgufValue::Int(i64::from(
            read_u8(r).map_err(ForgeError::Io)? as i8,
        ))),
        2 => Ok(GgufValue::UInt(u64::from(
            read_u16(r).map_err(ForgeError::Io)?,
        ))),
        3 => Ok(GgufValue::Int(i64::from(
            read_u16(r).map_err(ForgeError::Io)? as i16,
        ))),
        4 => Ok(GgufValue::UInt(u64::from(
            read_u32(r).map_err(ForgeError::Io)?,
        ))),
        5 => Ok(GgufValue::Int(i64::from(
            read_i32(r).map_err(ForgeError::Io)?,
        ))),
        6 => {
            let _ = read_f32(r).map_err(ForgeError::Io)?;
            Ok(GgufValue::Float)
        }
        7 => {
            let _ = read_u8(r).map_err(ForgeError::Io)?;
            Ok(GgufValue::Bool)
        }
        8 => Ok(GgufValue::Str(read_gguf_string(r, 4096)?)),
        9 => {
            let elem_type = read_u32(r).map_err(ForgeError::Io)?;
            let len = read_u64(r).map_err(ForgeError::Io)?;
            if let Some(elem_size) = scalar_type_size(elem_type) {
                let skip_bytes = len.checked_mul(elem_size).ok_or_else(|| {
                    ForgeError::ParseError("GGUF array byte size overflow".to_string())
                })?;
                r.seek(SeekFrom::Current(skip_bytes as i64))
                    .map_err(ForgeError::Io)?;
            } else if elem_type == 8 {
                for _ in 0..len {
                    let slen = read_u64(r).map_err(ForgeError::Io)?;
                    r.seek(SeekFrom::Current(slen as i64))
                        .map_err(ForgeError::Io)?;
                }
            } else if len <= MAX_INLINE_ARRAY_LEN {
                for _ in 0..len {
                    let _ = read_gguf_value(r, elem_type)?;
                }
            } else {
                return Err(ForgeError::ParseError(format!(
                    "unsupported nested GGUF array element type {elem_type}"
                )));
            }
            Ok(GgufValue::SkippedArray)
        }
        10 => Ok(GgufValue::UInt(read_u64(r).map_err(ForgeError::Io)?)),
        11 => Ok(GgufValue::Int(read_i64(r).map_err(ForgeError::Io)?)),
        12 => {
            let _ = read_f64(r).map_err(ForgeError::Io)?;
            Ok(GgufValue::Float)
        }
        other => Err(ForgeError::ParseError(format!(
            "unknown GGUF metadata value type {other}"
        ))),
    }
}

/// Maps a GGUF `general.file_type` integer code (and filename hint) to a canonical quantization label.
pub fn gguf_file_type_name(file_type: Option<u32>, file_name: &str) -> String {
    let upper = file_name.to_ascii_uppercase();
    for tag in [
        "UD-IQ4_XS",
        "IQ4_XS",
        "IQ4_NL",
        "Q4_K_M",
        "Q4_K_S",
        "Q4_K_L",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
        "Q4_0",
        "Q3_K_M",
        "Q3_K_L",
        "Q3_K_S",
        "IQ3_M",
        "IQ3_XS",
        "IQ2_M",
        "BF16",
        "F16",
    ] {
        if upper.contains(tag) {
            return tag.to_string();
        }
    }

    match file_type {
        Some(0) => "F32".to_string(),
        Some(1) => "F16".to_string(),
        Some(2) => "Q4_0".to_string(),
        Some(3) => "Q4_1".to_string(),
        Some(7) => "Q8_0".to_string(),
        Some(10) => "Q2_K".to_string(),
        Some(11) => "Q3_K_S".to_string(),
        Some(12) => "Q3_K_M".to_string(),
        Some(13) => "Q3_K_L".to_string(),
        Some(14) => "Q4_K_S".to_string(),
        Some(15) => "Q4_K_M".to_string(),
        Some(16) => "Q5_K_S".to_string(),
        Some(17) => "Q5_K_M".to_string(),
        Some(18) => "Q6_K".to_string(),
        Some(19) => "IQ2_XXS".to_string(),
        Some(20) => "IQ2_XS".to_string(),
        Some(21) => "Q2_K_S".to_string(),
        Some(22) => "IQ3_XS".to_string(),
        Some(23) => "IQ3_XXS".to_string(),
        Some(24) => "IQ1_S".to_string(),
        Some(25) => "IQ4_NL".to_string(),
        Some(26) => "IQ3_S".to_string(),
        Some(27) => "IQ3_M".to_string(),
        Some(28) => "IQ2_S".to_string(),
        Some(29) => "IQ2_M".to_string(),
        Some(30) => "IQ4_XS".to_string(),
        Some(31) => "IQ1_M".to_string(),
        Some(32) => "BF16".to_string(),
        Some(code) => format!("FTYPE_{code}"),
        None => "UNKNOWN".to_string(),
    }
}

/// Parses a GGUF v2/v3 file header and extracts model architecture and memory sizing metadata.
pub fn inspect_gguf_model(path: &Path) -> ForgeResult<GgufModelMetadata> {
    let file = File::open(path).map_err(ForgeError::Io)?;
    let file_size_bytes = file.metadata().map_err(ForgeError::Io)?.len();
    inspect_gguf_reader(BufReader::new(file), file_size_bytes, path)
}

/// Parses a GGUF stream of `file_size_bytes` bytes. `path` is only used for names in the result
/// and error messages. Exposed so synthetic fixtures can be inspected without touching disk.
pub fn inspect_gguf_reader<R: Read + Seek>(
    mut reader: R,
    file_size_bytes: u64,
    path: &Path,
) -> ForgeResult<GgufModelMetadata> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(ForgeError::Io)?;
    if &magic != GGUF_MAGIC {
        return Err(ForgeError::ParseError(format!(
            "file '{}' does not start with GGUF magic header",
            path.display()
        )));
    }

    let gguf_version = read_u32(&mut reader).map_err(ForgeError::Io)?;
    if !(2..=3).contains(&gguf_version) {
        return Err(ForgeError::ParseError(format!(
            "unsupported GGUF version {gguf_version} (expected 2 or 3)"
        )));
    }

    let tensor_count = read_u64(&mut reader).map_err(ForgeError::Io)?;
    let kv_count = read_u64(&mut reader).map_err(ForgeError::Io)?;
    if kv_count > 100_000 {
        return Err(ForgeError::ParseError(format!(
            "GGUF kv_count {kv_count} exceeds 100,000 safety limit"
        )));
    }

    let mut kvs = HashMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader, 512)?;
        let vtype = read_u32(&mut reader).map_err(ForgeError::Io)?;
        let val = read_gguf_value(&mut reader, vtype)?;
        if !key.starts_with("tokenizer.") {
            kvs.insert(key, val);
        }
    }

    let alignment = kvs
        .get("general.alignment")
        .and_then(GgufValue::as_u32)
        .map_or(DEFAULT_GGUF_ALIGNMENT, u64::from);
    let (tensor_records, data_offset) =
        read_tensor_records(&mut reader, tensor_count, file_size_bytes, alignment)?;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf")
        .to_string();
    let architecture = kvs
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("llama")
        .to_string();

    let get_arch_u32 = |suffix: &str| -> Option<u32> {
        let key = format!("{architecture}.{suffix}");
        kvs.get(&key).and_then(GgufValue::as_u32)
    };

    // Missing keys fall back to a guess, but every guess is recorded so callers can qualify
    // any estimate that depends on it instead of presenting it as exact.
    let mut assumed_defaults: Vec<String> = Vec::new();
    let mut get_or_assume = |suffix: &str, default: u32| -> u32 {
        get_arch_u32(suffix).unwrap_or_else(|| {
            assumed_defaults.push(format!(
                "{architecture}.{suffix} missing, assumed {default}"
            ));
            default
        })
    };

    let nextn_predict_layers = get_arch_u32("nextn_predict_layers").unwrap_or(0);
    let declared_context_length = get_or_assume("context_length", 4096);
    let embedding_length = get_or_assume("embedding_length", 4096);
    let feed_forward_length = get_arch_u32("feed_forward_length");
    let head_count = get_or_assume("attention.head_count", 32).max(1);
    let head_count_kv = get_arch_u32("attention.head_count_kv")
        .unwrap_or(head_count)
        .max(1);
    let default_head_dim = (embedding_length / head_count).max(64);
    let key_length = get_arch_u32("attention.key_length").unwrap_or(default_head_dim);
    let value_length = get_arch_u32("attention.value_length").unwrap_or(key_length);
    let full_attention_interval = get_arch_u32("full_attention_interval");
    let ssm_state_size = get_arch_u32("ssm.state_size");
    let ssm_inner_size = get_arch_u32("ssm.inner_size");
    let ssm_group_count = get_arch_u32("ssm.group_count");
    let expert_count = get_arch_u32("expert_count");
    let expert_used_count = get_arch_u32("expert_used_count");
    let expert_feed_forward_length = get_arch_u32("expert_feed_forward_length");
    let is_moe = expert_count.unwrap_or(0) > 1;

    let block_count = match (
        get_arch_u32("block_count"),
        inferred_block_count(&tensor_records),
    ) {
        (Some(declared), _) => declared.max(1),
        (None, Some(inferred)) => {
            assumed_defaults.push(format!(
                "{architecture}.block_count missing, inferred {inferred} from blk.N tensors"
            ));
            inferred
        }
        (None, None) => {
            assumed_defaults.push(format!("{architecture}.block_count missing, assumed 32"));
            32
        }
    };

    let tensor_layout = if tensor_records.is_empty() {
        None
    } else {
        Some(GgufTensorLayout::from_records(
            &tensor_records,
            block_count,
            alignment,
            data_offset,
        )?)
    };
    let split_count = kvs.get("split.count").and_then(GgufValue::as_u32);

    let file_type_code = kvs.get("general.file_type").and_then(GgufValue::as_u32);
    let quantization_name = gguf_file_type_name(file_type_code, &file_name);
    let bytes_per_layer = match &tensor_layout {
        Some(layout) => layout.mean_layer_bytes(),
        // Legacy approximation, only used when no tensor metadata exists (see `tensor_layout_basis`).
        None => file_size_bytes / u64::from(block_count.max(1)),
    };

    Ok(GgufModelMetadata {
        file_path: path.display().to_string(),
        file_name,
        file_size_bytes,
        gguf_version,
        tensor_count,
        kv_count,
        architecture,
        model_name: kvs
            .get("general.name")
            .and_then(GgufValue::as_str)
            .map(ToString::to_string),
        size_label: kvs
            .get("general.size_label")
            .and_then(GgufValue::as_str)
            .map(ToString::to_string),
        quantized_by: kvs
            .get("general.quantized_by")
            .and_then(GgufValue::as_str)
            .map(ToString::to_string),
        file_type_code,
        quantization_name,
        block_count,
        nextn_predict_layers,
        declared_context_length,
        embedding_length,
        feed_forward_length,
        head_count,
        head_count_kv,
        key_length,
        value_length,
        full_attention_interval,
        ssm_state_size,
        ssm_inner_size,
        ssm_group_count,
        expert_count,
        expert_used_count,
        expert_feed_forward_length,
        is_moe,
        bytes_per_layer,
        tensor_layout,
        split_count,
        assumed_defaults,
    })
}

/// Estimates total KV cache + recurrent SSM state bytes for `context_tokens` at `kv_quant`.
pub fn estimate_kv_and_state_bytes(
    meta: &GgufModelMetadata,
    context_tokens: u32,
    kv_quant: KvCacheQuant,
) -> u64 {
    let (attn_layers, ssm_layers) = match meta.full_attention_interval {
        Some(interval) if interval > 1 => {
            let attn = meta.block_count.div_ceil(interval);
            (attn, meta.block_count.saturating_sub(attn))
        }
        _ => (meta.block_count, 0),
    };

    // Header values are untrusted u32s; saturate instead of wrapping so a corrupt header can only
    // inflate the estimate (and therefore fail the "fits" check), never shrink it.
    let kv_elems_per_token_per_layer = u64::from(meta.head_count_kv)
        .saturating_mul(u64::from(meta.key_length) + u64::from(meta.value_length));
    let total_kv_elems = kv_elems_per_token_per_layer
        .saturating_mul(u64::from(attn_layers))
        .saturating_mul(u64::from(context_tokens));
    // `as u64` on an f64 saturates, which is the intended behavior here.
    let kv_bytes = (total_kv_elems as f64 * kv_quant.bytes_per_element()).ceil() as u64;

    let ssm_bytes_per_layer = match (meta.ssm_inner_size, meta.ssm_state_size) {
        (Some(inner), Some(state)) => u64::from(inner)
            .saturating_mul(u64::from(state))
            .saturating_mul(4)
            .saturating_add(512 * 1024),
        _ => 4 * 1024 * 1024,
    };
    let ssm_bytes = u64::from(ssm_layers).saturating_mul(ssm_bytes_per_layer);

    kv_bytes.saturating_add(ssm_bytes)
}

/// Auto-discovers local `.gguf` model files and `llama.cpp` Vulkan/ROCm binaries on Windows.
pub fn discover_local_llm_assets() -> LocalLlmAssets {
    let mut assets = LocalLlmAssets::default();
    let mut model_dirs = Vec::new();
    let mut backend_dirs = Vec::new();

    if let Ok(home) = std::env::var("USERPROFILE") {
        let home_path = PathBuf::from(&home);
        model_dirs.push(home_path.join(".lmstudio").join("models"));
        model_dirs.push(home_path.join("Models"));
        backend_dirs.push(
            home_path
                .join(".lmstudio")
                .join("extensions")
                .join("backends"),
        );
    }
    if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
        let local_path = PathBuf::from(&local_app);
        model_dirs.push(local_path.join("LM-Studio").join("models"));
    }

    for dir in model_dirs {
        collect_gguf_files(&dir, 0, 5, &mut assets.gguf_models);
    }
    assets.gguf_models.sort();
    assets.gguf_models.dedup();

    for dir in backend_dirs {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut subdirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            subdirs.sort();
            for sub in subdirs {
                let name = sub
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let server = sub.join("llama-server.exe");
                let bench = sub.join("llama-bench.exe");
                if name.contains("vulkan") {
                    if server.exists() {
                        assets.vulkan_server_bin = Some(server.display().to_string());
                    }
                    if bench.exists() {
                        assets.vulkan_bench_bin = Some(bench.display().to_string());
                    }
                } else if name.contains("rocm") || name.contains("hip") {
                    if server.exists() {
                        assets.hip_server_bin = Some(server.display().to_string());
                    }
                    if bench.exists() {
                        assets.hip_bench_bin = Some(bench.display().to_string());
                    }
                    if sub.join("hipblas.dll").exists() && assets.rocm_lib_dir.is_none() {
                        assets.rocm_lib_dir = Some(sub.display().to_string());
                    }
                }
            }
        }
    }

    // Check common Python ROCm venv locations for `_rocm_sdk_libraries\bin\hipblas.dll`
    if assets.rocm_lib_dir.is_none() {
        let mut candidate_rocm_bins = vec![PathBuf::from(
            r"C:\venvs\rdna-forge\Lib\site-packages\_rocm_sdk_libraries\bin",
        )];
        if let Ok(path_var) = std::env::var("PATH") {
            for entry in std::env::split_paths(&path_var) {
                candidate_rocm_bins.push(entry);
            }
        }
        for bin_dir in candidate_rocm_bins {
            if bin_dir.join("hipblas.dll").exists() {
                assets.rocm_lib_dir = Some(bin_dir.display().to_string());
                break;
            }
        }
    }

    assets
}

fn collect_gguf_files(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<String>) {
    if depth > max_depth || !dir.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, depth + 1, max_depth, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            let fname = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !fname.starts_with("mmproj") && !fname.contains("imatrix") {
                out.push(path.display().to_string());
            }
        }
    }
}

fn gpu_names_roughly_match(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.to_ascii_lowercase()
            .replace("(tm)", "")
            .replace("(r)", "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let na = norm(a);
    let nb = norm(b);
    !na.is_empty() && !nb.is_empty() && (na == nb || na.contains(&nb) || nb.contains(&na))
}

fn resolve_vulkan_visible_index(
    primary_gpu: &forge_core::GpuDeviceInfo,
    vulkan_devices: &[forge_device::VulkanDeviceProbe],
) -> usize {
    if let Some(ref pci) = primary_gpu.pci_bus_id {
        if let Some(found) = vulkan_devices
            .iter()
            .find(|v| v.pci_bus_id.as_deref() == Some(pci.as_str()))
        {
            return found.index;
        }
    }
    if let Some(found) = vulkan_devices.iter().find(|v| {
        v.vendor_id == primary_gpu.vendor_id
            && v.device_id == primary_gpu.device_id
            && v.discrete == primary_gpu.discrete
            && v.name == primary_gpu.name
    }) {
        return found.index;
    }
    vulkan_devices
        .iter()
        .find(|v| {
            v.vendor_id == primary_gpu.vendor_id
                && v.device_id == primary_gpu.device_id
                && v.discrete == primary_gpu.discrete
        })
        .map(|v| v.index)
        .unwrap_or(0)
}

fn resolve_hip_visible_index(
    primary_gpu: &forge_core::GpuDeviceInfo,
    hip_devices: &[forge_device::HipDeviceProbe],
) -> usize {
    if let Some(ref pci) = primary_gpu.pci_bus_id {
        if let Some(found) = hip_devices
            .iter()
            .find(|h| h.pci_bus_id.as_deref() == Some(pci.as_str()))
        {
            return found.hip_index.max(0) as usize;
        }
    }
    if let Some(found) = hip_devices.iter().find(|h| {
        gpu_names_roughly_match(&primary_gpu.name, &h.name)
            && h.integrated
                .map(|int| int != primary_gpu.discrete)
                .unwrap_or(true)
    }) {
        return found.hip_index.max(0) as usize;
    }
    hip_devices
        .iter()
        .find(|h| gpu_names_roughly_match(&primary_gpu.name, &h.name))
        .map(|h| h.hip_index.max(0) as usize)
        .unwrap_or(0)
}

const MIB: u64 = 1024 * 1024;

/// Weight-memory model used by the planner.
///
/// With GGUF tensor metadata it uses exact per-layer sizes and llama.cpp layer-offload
/// semantics. Without it, it falls back to the legacy uniform `file_size / block_count`
/// split, which is only an approximation (embeddings, output tensors, uneven layers and MoE
/// experts are not distinguished) and is labelled as such via [`Self::basis`].
struct WeightModel<'a> {
    model: &'a GgufModelMetadata,
}

impl<'a> WeightModel<'a> {
    fn new(model: &'a GgufModelMetadata) -> Self {
        Self { model }
    }

    fn basis(&self) -> &'static str {
        self.model.tensor_layout_basis()
    }

    fn file_mib(&self) -> u64 {
        self.model.file_size_bytes.div_ceil(MIB)
    }

    /// GPU weight MiB with every layer offloaded and no CPU experts.
    fn full_offload_mib(&self) -> u64 {
        match &self.model.tensor_layout {
            Some(l) => l.full_offload_bytes().div_ceil(MIB),
            None => self.file_mib(),
        }
    }

    /// Mean repeating-layer size in MiB (a mean over uneven layers, used only for coarse conversions).
    fn mean_layer_mib(&self) -> f64 {
        match &self.model.tensor_layout {
            Some(l) => (l.mean_layer_bytes() as f64 / MIB as f64).max(1.0),
            None => (self.file_mib() as f64 / f64::from(self.model.block_count.max(1))).max(1.0),
        }
    }

    /// Largest `-ngl` (<= block count) whose weights fit in `budget_mib`.
    fn max_ngl(&self, budget_mib: u64) -> u32 {
        match &self.model.tensor_layout {
            Some(l) => l.max_ngl_within(budget_mib.saturating_mul(MIB)),
            None => ((budget_mib as f64 / self.mean_layer_mib()).floor() as u32)
                .min(self.model.block_count),
        }
    }

    /// Smallest `--n-cpu-moe` that brings full-offload weights within `budget_mib`.
    fn min_cpu_moe(&self, budget_mib: u64) -> u32 {
        match &self.model.tensor_layout {
            Some(l) => l.min_cpu_moe_within(budget_mib.saturating_mul(MIB)),
            None => {
                let overflow = self.full_offload_mib().saturating_sub(budget_mib) as f64;
                // Legacy heuristic: experts assumed to be ~84% of each layer.
                let expert_mib_per_layer = (self.mean_layer_mib() * 0.84).max(1.0);
                ((overflow / expert_mib_per_layer).ceil() as u32).min(self.model.block_count)
            }
        }
    }
}

/// Builds a hardware-aware anti-WDDM-spill memory plan and architecture tweak matrix
/// for the primary (discrete preferred) GPU.
pub fn plan_llm_configuration(
    model: &GgufModelMetadata,
    target_context_tokens: u32,
    safety_margin_mib: u64,
) -> ForgeResult<LlmOptimizationPlan> {
    plan_llm_configuration_for_device(model, None, target_context_tokens, safety_margin_mib)
}

/// Builds a hardware-aware anti-WDDM-spill memory plan and architecture tweak matrix
/// for the discovered GPU at `device_index` (or primary GPU when `None`).
pub fn plan_llm_configuration_for_device(
    model: &GgufModelMetadata,
    device_index: Option<usize>,
    target_context_tokens: u32,
    safety_margin_mib: u64,
) -> ForgeResult<LlmOptimizationPlan> {
    let snapshot = discover_all();
    let primary_gpu = match device_index {
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

    let matched = match_profile_for_device(primary_gpu);
    let gpu_name = primary_gpu.name.clone();
    let gfx_target = primary_gpu
        .gfx_target
        .as_ref()
        .map(|t| t.base.clone())
        .or_else(|| {
            matched
                .as_ref()
                .map(|m| m.profile.device.gfx_target.clone())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let vk_visible_device = resolve_vulkan_visible_index(primary_gpu, &snapshot.vulkan.devices);
    let hip_visible_device = resolve_hip_visible_index(primary_gpu, &snapshot.hip.devices);
    let disabled_gpu_indices: Vec<usize> = snapshot
        .vulkan
        .devices
        .iter()
        .filter(|v| v.index != vk_visible_device)
        .map(|v| v.index)
        .collect();

    let secondary_igpu = snapshot
        .gpus
        .iter()
        .find(|g| g.index != primary_gpu.index && !g.discrete);
    let secondary_igpu_name = secondary_igpu.map(|g| g.name.clone());

    let logical_cores = snapshot.system.logical_cores;
    let physical_cpu_cores = (logical_cores / 2).clamp(4, 32) as u32;
    let total_ram_mib = snapshot.system.total_ram_bytes / (1024 * 1024);

    let telemetry = snapshot_all_gpus().ok().unwrap_or_default();
    let active_tel = telemetry
        .iter()
        .find(|s| s.pci_device_id == Some(primary_gpu.device_id))
        .or_else(|| {
            telemetry
                .iter()
                .find(|s| s.discrete == primary_gpu.discrete)
        })
        .or_else(|| telemetry.first());

    let mut capacity_warnings: Vec<String> = Vec::new();
    let total_vram_mib: u64 = active_tel
        .and_then(|s| s.total_vram_mib.map(u64::from))
        .or_else(|| (primary_gpu.vram_bytes > 0).then_some(primary_gpu.vram_bytes / (1024 * 1024)))
        .or_else(|| {
            matched
                .as_ref()
                .map(|m| m.profile.memory.vram_bytes / (1024 * 1024))
        })
        .unwrap_or_else(|| {
            capacity_warnings.push(
                "GPU VRAM capacity could not be read from telemetry, Vulkan or a hardware profile; 8192 MiB is a placeholder, not a measurement. Plan figures are unreliable."
                    .to_string(),
            );
            8_192
        });
    let used_vram_mib: u64 = active_tel
        .and_then(|s| s.vram_used_mib.map(|v| v.max(0) as u64))
        .unwrap_or_else(|| {
            capacity_warnings.push(
                "Current VRAM usage could not be read; 1024 MiB is an assumed placeholder, not a measurement."
                    .to_string(),
            );
            1_024
        })
        .min(total_vram_mib);

    let ctx = LlmTargetGpuContext {
        gpu_index: primary_gpu.index,
        gpu_name,
        gfx_target,
        architecture: primary_gpu.architecture,
        discrete: primary_gpu.discrete,
        matched_profile_id: matched.as_ref().map(|m| m.id.clone()),
        matched_profile_name: matched.as_ref().map(|m| m.profile.device.name.clone()),
        vk_visible_device,
        hip_visible_device,
        disabled_gpu_indices,
        secondary_igpu_name,
        physical_cpu_cores,
        total_ram_mib,
        total_vram_mib,
        used_vram_mib,
    };

    let mut plan =
        plan_llm_configuration_for_gpu(model, &ctx, target_context_tokens, safety_margin_mib);
    plan.warnings.extend(capacity_warnings);

    let mib_per_layer = WeightModel::new(model).mean_layer_mib();
    plan.top_vram_processes = query_top_vram_processes(6)
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.dedicated_vram_mib >= 16)
        .map(|p| ProcessVramImpact {
            pid: p.pid,
            process_name: p.process_name,
            dedicated_vram_mib: p.dedicated_vram_mib,
            shared_vram_mib: p.shared_vram_mib,
            recoverable_gpu_layers: ((p.dedicated_vram_mib as f64 / mib_per_layer).floor() as u32)
                .min(model.block_count),
        })
        .collect();

    Ok(plan)
}

/// Pure multi-GPU LLM memory & launch planner over an explicit `LlmTargetGpuContext`.
pub fn plan_llm_configuration_for_gpu(
    model: &GgufModelMetadata,
    ctx: &LlmTargetGpuContext,
    target_context_tokens: u32,
    safety_margin_mib: u64,
) -> LlmOptimizationPlan {
    let target_ctx = target_context_tokens.clamp(512, 262_144);
    let is_apu_unified_memory = !ctx.discrete;

    let mut warnings = Vec::new();
    let (total_vram_mib, apu_small_carveout_scaled) =
        if is_apu_unified_memory && ctx.total_vram_mib < 2_048 && ctx.total_ram_mib >= 8_192 {
            ((ctx.total_ram_mib / 2).max(ctx.total_vram_mib), true)
        } else {
            (ctx.total_vram_mib, false)
        };
    let used_vram_mib = ctx.used_vram_mib.min(total_vram_mib);
    let free_vram_mib = total_vram_mib.saturating_sub(used_vram_mib);

    if apu_small_carveout_scaled {
        warnings.push(format!(
            "Integrated APU ({}) reports {} MiB dedicated UMA carveout out of {} MiB system RAM. Windows WDDM allocates shared system RAM dynamically (~{} MiB usable half-RAM ceiling). On RDNA3.5 / Strix Halo APUs, increase UMA Frame Buffer Size in BIOS or Adrenalin Variable Graphics Memory (VGM) for optimal ROCm/Vulkan stability with large models.",
            ctx.gpu_name, ctx.total_vram_mib, ctx.total_ram_mib, total_vram_mib
        ));
    }

    let weights = WeightModel::new(model);
    let model_weights_mib = weights.full_offload_mib();
    let model_file_mib = weights.file_mib();
    let mib_per_layer = weights.mean_layer_mib();
    for note in model.tensor_layout_notes() {
        warnings.push(format!("Estimate qualification: {note}"));
    }
    let compute_scratch_mib = if model.embedding_length <= 4096 && target_ctx <= 8192 {
        256u64
    } else if model.embedding_length >= 6144 || target_ctx >= 32_768 {
        512u64
    } else {
        384u64
    };

    let baseline_os_vram_mib = (if total_vram_mib <= 8_192 {
        768u64
    } else if total_vram_mib <= 12_288 {
        950u64
    } else {
        1_150u64
    })
    .min(total_vram_mib / 4);

    let mut kv_budgets = Vec::with_capacity(3);
    for kv_q in [KvCacheQuant::F16, KvCacheQuant::Q8_0, KvCacheQuant::Q4_0] {
        let kv_bytes = estimate_kv_and_state_bytes(model, target_ctx, kv_q);
        let kv_and_state_mib = kv_bytes.div_ceil(1024 * 1024);
        let total_required_mib = model_weights_mib
            .saturating_add(kv_and_state_mib)
            .saturating_add(compute_scratch_mib);

        let usable_free_mib = free_vram_mib.saturating_sub(safety_margin_mib);
        let fits_in_free_vram = total_required_mib <= usable_free_mib;

        // With GGML_VK_ENABLE_MEMORY_PRIORITY=1 (VK_EXT_memory_priority), Windows WDDM VidMm
        // can page out non-pinned background app VRAM down to OS baseline.
        let priority_budget_mib = total_vram_mib
            .saturating_sub(baseline_os_vram_mib.saturating_add(safety_margin_mib / 2))
            .max(usable_free_mib);
        let fits_with_wddm_memory_priority = total_required_mib <= priority_budget_mib;

        let budget_for_weights_strict =
            usable_free_mib.saturating_sub(kv_and_state_mib.saturating_add(compute_scratch_mib));
        let strict_zero_spill_ngl = weights.max_ngl(budget_for_weights_strict);

        let budget_for_weights_prio = priority_budget_mib
            .saturating_sub(kv_and_state_mib.saturating_add(compute_scratch_mib));
        let memory_priority_ngl = weights.max_ngl(budget_for_weights_prio);

        // For MoE models, `--n-cpu-moe N` keeps `-ngl 99` (attention, shared tensors and KV cache
        // on GPU) and moves only the routed-expert tensors of the first N layers to system RAM.
        // With tensor metadata N is computed from the real expert bytes of those layers; without it
        // the legacy "experts are ~84% of a layer" heuristic is used (see `WeightModel`).
        let recommended_n_cpu_moe = if model.is_moe && total_required_mib > usable_free_mib {
            let weights_budget_mib = usable_free_mib
                .saturating_sub(kv_and_state_mib.saturating_add(compute_scratch_mib));
            weights.min_cpu_moe(weights_budget_mib)
        } else {
            0
        };

        kv_budgets.push(KvCacheMemoryBudget {
            kv_quant: kv_q.as_cli_str().to_string(),
            requires_flash_attn: kv_q.requires_flash_attn_for_v(),
            context_tokens: target_ctx,
            model_weights_mib,
            model_file_mib,
            weights_estimate_basis: weights.basis().to_string(),
            kv_and_state_mib,
            compute_scratch_mib,
            total_required_mib,
            fits_in_free_vram,
            fits_with_wddm_memory_priority,
            strict_zero_spill_ngl,
            memory_priority_ngl,
            recommended_n_cpu_moe,
        });
    }

    let reclaimable_vram_mib = used_vram_mib.saturating_sub(baseline_os_vram_mib);
    let recoverable_layers_if_apps_closed =
        ((reclaimable_vram_mib as f64 / mib_per_layer).floor() as u32).min(model.block_count);

    if model.declared_context_length > 16_384 {
        warnings.push(format!(
            "Model metadata declares a default context of {} tokens ({}K). Loading without an explicit context cap (-c {}) will over-allocate VRAM and spill into system RAM.",
            model.declared_context_length,
            model.declared_context_length / 1024,
            target_ctx
        ));
    }
    if let Some(ref igpu_name) = ctx.secondary_igpu_name {
        warnings.push(format!(
            "Secondary integrated GPU ({igpu_name}) detected. On Windows ROCm/Vulkan, this iGPU exposes shared system RAM as 'VRAM', causing llama.cpp's default '-sm layer' to split layers onto the slow iGPU unless GGML_VK_VISIBLE_DEVICES={} / HIP_VISIBLE_DEVICES={} and '-sm none' are set.",
            ctx.vk_visible_device, ctx.hip_visible_device
        ));
    }
    if ctx.architecture == GpuArchitecture::Rdna1 {
        warnings.push(format!(
            "{} uses RDNA1 ({}), where ROCm/HIP lacks official Windows support and hardware INT8 DP4A instructions are unavailable. Use the Vulkan backend.",
            ctx.gpu_name, ctx.gfx_target
        ));
    }
    if !kv_budgets[0].fits_in_free_vram {
        warnings.push(format!(
            "Model weights ({} MiB) + KV/scratch ({} MiB) = {} MiB exceed currently free {} VRAM ({} MiB; {} MiB currently occupied by Windows/apps). Enable GGML_VK_ENABLE_MEMORY_PRIORITY=1 and/or quantized KV cache (-ctk q4_0 -ctv q4_0) or cap -ngl to {} to prevent WDDM PCIe thrashing.",
            model_weights_mib,
            kv_budgets[0].kv_and_state_mib + compute_scratch_mib,
            kv_budgets[0].total_required_mib,
            if is_apu_unified_memory { "unified" } else { "physical" },
            free_vram_mib,
            used_vram_mib,
            kv_budgets[0].strict_zero_spill_ngl
        ));
    }

    let is_hybrid_ssm = model.full_attention_interval.unwrap_or(1) > 1;
    let physical_cpu_cores = ctx.physical_cpu_cores.clamp(1, 128);
    let vk_dev_str = ctx.vk_visible_device.to_string();
    let hip_dev_str = ctx.hip_visible_device.to_string();
    let arch_label = ctx.architecture.to_string();

    let f16_budget = &kv_budgets[0];
    let q8_budget = &kv_budgets[1];
    let q4_budget = &kv_budgets[2];
    let vk_prio_ngl = if f16_budget.fits_with_wddm_memory_priority {
        99
    } else {
        f16_budget.memory_priority_ngl
    };
    let vk_q4_ngl = if q4_budget.fits_with_wddm_memory_priority {
        99
    } else {
        q4_budget.memory_priority_ngl
    };

    let mut recommended_profiles = Vec::new();

    // Profile 1: Vulkan WDDM High-Priority Interactive
    let vk_prio_fa = match ctx.architecture {
        GpuArchitecture::Rdna1 | GpuArchitecture::Rdna2 => !is_hybrid_ssm,
        _ => !is_hybrid_ssm,
    };
    let vk_prio_rationale = if is_hybrid_ssm
        && matches!(
            ctx.architecture,
            GpuArchitecture::Rdna1 | GpuArchitecture::Rdna2
        ) {
        if ctx.architecture == GpuArchitecture::Rdna2 {
            format!(
                "Activates VK_EXT_memory_priority (1.0) so Windows WDDM keeps weights resident in {} ({}) VRAM, pins Vulkan device {}, and disables Vulkan FlashAttention (-fa off) which suffers a severe prefill slowdown at depth on hybrid qwen35 layers. Verified on RX 6900 XT (RDNA2) + Qwen3.8-27B: 343.2 tok/s prefill (+108%) and 26.81 tok/s generation (+63%) at 2048 context.",
                ctx.gpu_name, ctx.gfx_target, vk_dev_str
            )
        } else {
            format!(
                "Activates VK_EXT_memory_priority (1.0) on {} ({}), pins Vulkan device {}, and disables Vulkan FlashAttention (-fa off) for hybrid SSM/attention layers.",
                ctx.gpu_name, ctx.gfx_target, vk_dev_str
            )
        }
    } else {
        format!(
            "Activates VK_EXT_memory_priority (1.0) to prevent Windows WDDM from paging LLM weights out to system RAM and isolates execution strictly to {} ({}, Vulkan index {}).",
            ctx.gpu_name, ctx.gfx_target, vk_dev_str
        )
    };
    recommended_profiles.push(LlmTweakProfile {
        id: "vulkan_wddm_priority".to_string(),
        title: "Vulkan WDDM High-Priority (Fastest Interactive Chat <= 2K-3K Context)".to_string(),
        backend: "vulkan".to_string(),
        device_selector: "Vulkan0".to_string(),
        n_gpu_layers: vk_prio_ngl,
        n_cpu_moe: 0,
        split_mode: "none".to_string(),
        flash_attn: vk_prio_fa,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        batch_size: 2048,
        ubatch_size: 512,
        threads: physical_cpu_cores,
        env_vars: vec![
            ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
            ("GGML_VK_VISIBLE_DEVICES".to_string(), vk_dev_str.clone()),
        ],
        rationale: vk_prio_rationale,
        llama_server_cmd: format!(
            "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"{}\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev Vulkan0 -sm none -mg 0 -fa {} -ctk f16 -ctv f16 -b 2048 -ub 512 -t {}",
            vk_dev_str,
            model.file_path,
            target_ctx,
            vk_prio_ngl,
            if vk_prio_fa { "on" } else { "off" },
            physical_cpu_cores
        ),
    });

    // Profile 2: ROCm/HIP (WMMA + hipBLASLt on RDNA3/3.5/4; standard ROCm on RDNA2; skipped on RDNA1)
    if ctx.architecture != GpuArchitecture::Rdna1 {
        if ctx.architecture.supports_hipblaslt() {
            let use_q8 = q8_budget.fits_in_free_vram || q8_budget.fits_with_wddm_memory_priority;
            let (rocm_kv_enum, rocm_budget) = if use_q8 {
                (KvCacheQuant::Q8_0, q8_budget)
            } else {
                (KvCacheQuant::Q4_0, q4_budget)
            };
            let rocm_kv = rocm_kv_enum.as_cli_str();
            let rocm_ngl = if rocm_budget.fits_in_free_vram {
                99
            } else {
                rocm_budget
                    .strict_zero_spill_ngl
                    .max(rocm_budget.memory_priority_ngl.saturating_sub(2))
            };
            recommended_profiles.push(LlmTweakProfile {
                id: "rocm_wmma_hipblaslt".to_string(),
                title: format!(
                    "ROCm / HIP WMMA + hipBLASLt + FlashAttention (Fastest Prefill on {arch_label})"
                ),
                backend: "hip".to_string(),
                device_selector: "ROCm0".to_string(),
                n_gpu_layers: rocm_ngl,
                n_cpu_moe: 0,
                split_mode: "none".to_string(),
                flash_attn: true,
                cache_type_k: rocm_kv.to_string(),
                cache_type_v: rocm_kv.to_string(),
                batch_size: 2048,
                ubatch_size: 512,
                threads: physical_cpu_cores,
                env_vars: vec![
                    ("HIP_VISIBLE_DEVICES".to_string(), hip_dev_str.clone()),
                    ("ROCBLAS_USE_HIPBLASLT".to_string(), "1".to_string()),
                ],
                rationale: format!(
                    "Isolates HIP device {hip_dev_str} ({} / {}), enables ROCBLAS_USE_HIPBLASLT=1 for {arch_label} matrix-core paths alongside native rocWMMA FlashAttention (-fa on), and uses {rocm_kv} KV cache quantization to maximize prefill speed and context capacity.",
                    ctx.gpu_name, ctx.gfx_target
                ),
                llama_server_cmd: format!(
                    "$env:HIP_VISIBLE_DEVICES=\"{}\"; $env:ROCBLAS_USE_HIPBLASLT=\"1\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev ROCm0 -sm none -mg 0 -fa on -ctk {} -ctv {} -b 2048 -ub 512 -t {}",
                    hip_dev_str, model.file_path, target_ctx, rocm_ngl, rocm_kv, rocm_kv, physical_cpu_cores
                ),
            });
        } else {
            let rocm_q4_ngl = if q4_budget.fits_in_free_vram {
                99
            } else {
                q4_budget
                    .strict_zero_spill_ngl
                    .max(q4_budget.memory_priority_ngl.saturating_sub(2))
            };
            let rocm_rationale = if ctx.architecture == GpuArchitecture::Rdna2 {
                format!(
                    "Pins HIP execution to {} ({}) via HIP_VISIBLE_DEVICES={} (preventing any integrated GPU shared-RAM layer split), enables native ROCm FlashAttention (-fa on), and compresses the KV cache 3.5x (-ctk q4_0 -ctv q4_0). Verified on RX 6900 XT (RDNA2) + Qwen3.8-27B: 522.6 tok/s initial prefill and 246.2 tok/s prefill at 4096 context (20.3x faster than Vulkan -fa 1).",
                    ctx.gpu_name, ctx.gfx_target, hip_dev_str
                )
            } else {
                format!(
                    "Pins HIP execution to {} ({}) via HIP_VISIBLE_DEVICES={}, enables native ROCm FlashAttention (-fa on), and uses Q4_0 KV cache quantization (-ctk q4_0 -ctv q4_0) to fit 4K-16K context cleanly.",
                    ctx.gpu_name, ctx.gfx_target, hip_dev_str
                )
            };
            recommended_profiles.push(LlmTweakProfile {
                id: "rocm_q4_kv_long_ctx".to_string(),
                title: "ROCm / HIP FlashAttention + Q4_0 KV Cache (Fastest Prefill & 4K-16K Context)".to_string(),
                backend: "hip".to_string(),
                device_selector: "ROCm0".to_string(),
                n_gpu_layers: rocm_q4_ngl,
                n_cpu_moe: 0,
                split_mode: "none".to_string(),
                flash_attn: true,
                cache_type_k: "q4_0".to_string(),
                cache_type_v: "q4_0".to_string(),
                batch_size: 2048,
                ubatch_size: 512,
                threads: physical_cpu_cores,
                env_vars: vec![("HIP_VISIBLE_DEVICES".to_string(), hip_dev_str.clone())],
                rationale: rocm_rationale,
                llama_server_cmd: format!(
                    "$env:HIP_VISIBLE_DEVICES=\"{}\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev ROCm0 -sm none -mg 0 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t {}",
                    hip_dev_str, model.file_path, target_ctx, rocm_q4_ngl, physical_cpu_cores
                ),
            });
        }
    }

    // Profile 3: Vulkan Long-Context Decode with Q4_0 KV Cache
    let vk_q4_rationale = if ctx.architecture == GpuArchitecture::Rdna2 {
        format!(
            "Combines VK_EXT_memory_priority with 4-bit KV cache quantization (-ctk q4_0 -ctv q4_0 -fa on) on {} ({}, Vulkan index {}) to keep 4K+ context inside VRAM. Verified on RX 6900 XT (RDNA2) + Qwen3.8-27B: sustains 25.63 tok/s token generation at 4096 context (+61% vs f16 KV).",
            ctx.gpu_name, ctx.gfx_target, vk_dev_str
        )
    } else {
        format!(
            "Combines VK_EXT_memory_priority with 4-bit KV cache quantization (-ctk q4_0 -ctv q4_0 -fa on) on {} ({}, Vulkan index {}) to keep 4K+ context inside VRAM.",
            ctx.gpu_name, ctx.gfx_target, vk_dev_str
        )
    };
    recommended_profiles.push(LlmTweakProfile {
        id: "vulkan_q4_kv_decode".to_string(),
        title: "Vulkan WDDM Priority + Q4_0 KV Cache (Max Token Generation Speed at 4K+ Context)"
            .to_string(),
        backend: "vulkan".to_string(),
        device_selector: "Vulkan0".to_string(),
        n_gpu_layers: vk_q4_ngl,
        n_cpu_moe: 0,
        split_mode: "none".to_string(),
        flash_attn: true,
        cache_type_k: "q4_0".to_string(),
        cache_type_v: "q4_0".to_string(),
        batch_size: 2048,
        ubatch_size: 512,
        threads: physical_cpu_cores,
        env_vars: vec![
            ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
            ("GGML_VK_VISIBLE_DEVICES".to_string(), vk_dev_str.clone()),
        ],
        rationale: vk_q4_rationale,
        llama_server_cmd: format!(
            "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"{}\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev Vulkan0 -sm none -mg 0 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t {}",
            vk_dev_str, model.file_path, target_ctx, vk_q4_ngl, physical_cpu_cores
        ),
    });

    // Optional MoE Expert-Offload Profile (-ncmoe) when model.is_moe is true
    if model.is_moe && q4_budget.recommended_n_cpu_moe > 0 {
        let ncmoe = q4_budget.recommended_n_cpu_moe;
        recommended_profiles.push(LlmTweakProfile {
            id: "moe_expert_offload".to_string(),
            title: format!(
                "MoE Smart Expert Offload (100% Attention on GPU + -ncmoe {ncmoe} Experts on Host RAM)"
            ),
            backend: "vulkan".to_string(),
            device_selector: "Vulkan0".to_string(),
            n_gpu_layers: 99,
            n_cpu_moe: ncmoe,
            split_mode: "none".to_string(),
            flash_attn: true,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            batch_size: 2048,
            ubatch_size: 512,
            threads: physical_cpu_cores,
            env_vars: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
                ("GGML_VK_VISIBLE_DEVICES".to_string(), vk_dev_str.clone()),
            ],
            rationale: format!(
                "MoE model detected ({:?} routed experts, {:?} active per token). Keeps -ngl 99 so 100% of attention layers and KV cache remain in {} VRAM while offloading {}/{} layers' sparse expert FFN blocks (-ncmoe {}) to host RAM.",
                model.expert_count, model.expert_used_count, ctx.gpu_name, ncmoe, model.block_count, ncmoe
            ),
            llama_server_cmd: format!(
                "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"{}\"; llama-server.exe -m \"{}\" -c {} -ngl 99 -ncmoe {} -dev Vulkan0 -sm none -mg 0 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -t {}",
                vk_dev_str, model.file_path, target_ctx, ncmoe, physical_cpu_cores
            ),
        });
    }

    // Profile 4: Strict Zero-Spill Hybrid GPU+CPU Split
    if f16_budget.strict_zero_spill_ngl < model.block_count {
        let safe_ngl = q8_budget.strict_zero_spill_ngl;
        recommended_profiles.push(LlmTweakProfile {
            id: "vulkan_strict_zero_spill".to_string(),
            title: format!(
                "Strict Zero-Spill Hybrid ({}/{} GPU Layers + {} CPU Threads)",
                safe_ngl, model.block_count, physical_cpu_cores
            ),
            backend: "vulkan".to_string(),
            device_selector: "Vulkan0".to_string(),
            n_gpu_layers: safe_ngl,
            n_cpu_moe: 0,
            split_mode: "none".to_string(),
            flash_attn: false,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "f16".to_string(),
            batch_size: 2048,
            ubatch_size: 512,
            threads: physical_cpu_cores,
            env_vars: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
                ("GGML_VK_VISIBLE_DEVICES".to_string(), vk_dev_str.clone()),
            ],
            rationale: format!(
                "Caps GPU layers to exactly {}/{} on {} ({} MiB safety margin) so the model fits strictly inside currently unallocated VRAM ({} MiB) while background apps remain open.",
                safe_ngl, model.block_count, ctx.gpu_name, safety_margin_mib, free_vram_mib
            ),
            llama_server_cmd: format!(
                "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"{}\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev Vulkan0 -sm none -mg 0 -fa off -ctk q8_0 -ctv f16 -b 2048 -ub 512 -t {}",
                vk_dev_str, model.file_path, target_ctx, safe_ngl, physical_cpu_cores
            ),
        });
    }

    let mut lm_studio_tips = vec![
        if ctx.architecture.supports_hipblaslt() {
            format!(
                "1. Set Windows User Environment Variables: GGML_VK_ENABLE_MEMORY_PRIORITY=1, GGML_VK_VISIBLE_DEVICES={vk_dev_str}, HIP_VISIBLE_DEVICES={hip_dev_str}, and ROCBLAS_USE_HIPBLASLT=1 (enabling WDDM residency priority on Vulkan and WMMA/hipBLASLt paths on {arch_label} {}).",
                ctx.gpu_name
            )
        } else if ctx.architecture == GpuArchitecture::Rdna1 {
            format!(
                "1. Set Windows User Environment Variables: GGML_VK_ENABLE_MEMORY_PRIORITY=1 and GGML_VK_VISIBLE_DEVICES={vk_dev_str}, and select the Vulkan llama.cpp runtime in LM Studio for {} ({}).",
                ctx.gpu_name, ctx.gfx_target
            )
        } else {
            format!(
                "1. Set Windows User Environment Variables: GGML_VK_ENABLE_MEMORY_PRIORITY=1, GGML_VK_VISIBLE_DEVICES={vk_dev_str}, and HIP_VISIBLE_DEVICES={hip_dev_str} (boosts Windows WDDM VRAM residency priority and pins execution to {}).",
                ctx.gpu_name
            )
        },
        format!(
            "2. Context Length: Change from the GGUF metadata default ({} tokens) to 4096 or 8192 tokens so the KV cache does not overflow the remaining {} MiB of free {} VRAM.",
            model.declared_context_length,
            free_vram_mib,
            if is_apu_unified_memory { "unified" } else { "physical" }
        ),
        if is_hybrid_ssm && matches!(ctx.architecture, GpuArchitecture::Rdna1 | GpuArchitecture::Rdna2) {
            "3. Flash Attention & KV Cache (hybrid SSM+Attention architecture): On RDNA1/RDNA2 Vulkan at <= 3072 context, turn Flash Attention OFF to avoid prefill slowdowns. For >= 4096 context, use ROCm with Flash Attention ON + Q4_0 KV Cache, or Vulkan with Flash Attention ON + Q4_0 KV Cache.".to_string()
        } else {
            "3. Flash Attention & KV Cache: Enable Flash Attention and set K/V Cache Quantization to Q8_0 (or Q4_0 for very long contexts) to save 50%-72% of context VRAM.".to_string()
        },
        format!(
            "4. CPU Thread Count & Device Isolation: Set CPU Threads to {} (matching physical cores rather than SMT threads) and target {}.",
            physical_cpu_cores, ctx.gpu_name
        ),
    ];

    if is_apu_unified_memory {
        lm_studio_tips.push(format!(
            "5. APU Unified Memory Tuning ({}): Configure BIOS UMA Frame Buffer Size or Adrenalin Variable Graphics Memory (VGM) large enough to hold model weights + KV cache without Windows paging.",
            ctx.gpu_name
        ));
    }

    LlmOptimizationPlan {
        model: model.clone(),
        gpu_index: ctx.gpu_index,
        gpu_name: ctx.gpu_name.clone(),
        gfx_target: ctx.gfx_target.clone(),
        gpu_architecture: arch_label,
        matched_profile_id: ctx.matched_profile_id.clone(),
        matched_profile_name: ctx.matched_profile_name.clone(),
        vk_visible_device: ctx.vk_visible_device,
        hip_visible_device: ctx.hip_visible_device,
        disabled_gpu_indices: ctx.disabled_gpu_indices.clone(),
        is_apu_unified_memory,
        has_secondary_igpu: ctx.secondary_igpu_name.is_some(),
        secondary_igpu_name: ctx.secondary_igpu_name.clone(),
        physical_cpu_cores,
        total_vram_mib,
        used_vram_mib,
        free_vram_mib,
        safety_margin_mib,
        target_context_tokens: target_ctx,
        recoverable_layers_if_apps_closed,
        top_vram_processes: Vec::new(),
        warnings,
        kv_budgets,
        recommended_profiles,
        lm_studio_tips,
    }
}

/// Runs an empirical `llama-bench` autotuning sweep across architecture-aware tweak candidates on the target GGUF model.
pub fn autotune_llm_model(config: &LlmAutotuneConfig) -> ForgeResult<LlmAutotuneReport> {
    if !config.model_path.exists() {
        return Err(ForgeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("GGUF model file not found: {}", config.model_path.display()),
        )));
    }
    if config.vulkan_bin.is_none() && config.hip_bin.is_none() {
        return Err(ForgeError::RuntimeUnavailable {
            runtime: "llama-bench".to_string(),
            reason: "provide at least one of --vulkan-bin or --hip-bin for empirical autotuning"
                .to_string(),
        });
    }

    let meta = inspect_gguf_model(&config.model_path)?;
    let plan = plan_llm_configuration_for_device(
        &meta,
        config.device_index,
        config.context_depth.max(2048),
        512,
    )?;
    let reps = config.reps.clamp(1, 5);
    let prompt_tokens = config.prompt_tokens.clamp(32, 2048);
    let gen_tokens = config.gen_tokens.clamp(16, 512);

    let f16_safe_ngl = plan
        .kv_budgets
        .first()
        .map(|b| {
            if b.fits_with_wddm_memory_priority {
                99
            } else {
                b.memory_priority_ngl.max(1)
            }
        })
        .unwrap_or(99);
    let q4_safe_ngl = plan
        .kv_budgets
        .get(2)
        .map(|b| {
            if b.fits_with_wddm_memory_priority {
                99
            } else {
                b.memory_priority_ngl.max(1)
            }
        })
        .unwrap_or(99);

    let vk_dev_str = plan.vk_visible_device.to_string();
    let hip_dev_str = plan.hip_visible_device.to_string();
    let supports_hipblaslt = matches!(
        plan.gpu_architecture.as_str(),
        "RDNA3" | "RDNA3.5" | "RDNA4" | "CDNA2" | "CDNA3" | "CDNA4"
    );

    struct ProbeSpec {
        id: &'static str,
        title: &'static str,
        backend: &'static str,
        dev: &'static str,
        ngl: u32,
        fa: bool,
        ctk: &'static str,
        ctv: &'static str,
        envs: Vec<(&'static str, String)>,
    }

    let mut specs = Vec::new();
    if config.vulkan_bin.is_some() {
        specs.push(ProbeSpec {
            id: "vulkan_naive_fa1",
            title: "Baseline Vulkan (-fa 1, f16 KV, default WDDM priority)",
            backend: "vulkan",
            dev: "Vulkan0",
            ngl: f16_safe_ngl,
            fa: true,
            ctk: "f16",
            ctv: "f16",
            envs: vec![("GGML_VK_VISIBLE_DEVICES", vk_dev_str.clone())],
        });
        specs.push(ProbeSpec {
            id: "vulkan_wddm_priority_fa0",
            title: "Tuned Vulkan (GGML_VK_ENABLE_MEMORY_PRIORITY=1 + -fa 0 + GPU isolated)",
            backend: "vulkan",
            dev: "Vulkan0",
            ngl: f16_safe_ngl,
            fa: false,
            ctk: "f16",
            ctv: "f16",
            envs: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY", "1".to_string()),
                ("GGML_VK_VISIBLE_DEVICES", vk_dev_str.clone()),
            ],
        });
        specs.push(ProbeSpec {
            id: "vulkan_wddm_priority_q4_kv",
            title: "Tuned Vulkan Long-Context (MEMORY_PRIORITY=1 + -fa 1 + q4_0 KV cache)",
            backend: "vulkan",
            dev: "Vulkan0",
            ngl: q4_safe_ngl,
            fa: true,
            ctk: "q4_0",
            ctv: "q4_0",
            envs: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY", "1".to_string()),
                ("GGML_VK_VISIBLE_DEVICES", vk_dev_str),
            ],
        });
    }

    if config.hip_bin.is_some() && plan.gpu_architecture != "RDNA1" {
        let mut hip_envs = vec![("HIP_VISIBLE_DEVICES", hip_dev_str)];
        if supports_hipblaslt {
            hip_envs.push(("ROCBLAS_USE_HIPBLASLT", "1".to_string()));
        }
        specs.push(ProbeSpec {
            id: if supports_hipblaslt {
                "rocm_wmma_hipblaslt"
            } else {
                "rocm_isolated_q4_kv"
            },
            title: if supports_hipblaslt {
                "Tuned ROCm/HIP (HIP_VISIBLE_DEVICES + hipBLASLt + -fa 1 + q4_0 KV cache)"
            } else {
                "Tuned ROCm/HIP (HIP_VISIBLE_DEVICES + -fa 1 + q4_0 KV cache)"
            },
            backend: "hip",
            dev: "ROCm0",
            ngl: q4_safe_ngl,
            fa: true,
            ctk: "q4_0",
            ctv: "q4_0",
            envs: hip_envs,
        });
    }

    let mut candidates = Vec::with_capacity(specs.len());

    for spec in specs {
        let bin_path = if spec.backend == "vulkan" {
            config.vulkan_bin.as_ref().unwrap()
        } else {
            config.hip_bin.as_ref().unwrap()
        };

        let mut cmd = Command::new(bin_path);
        if let Some(parent) = bin_path.parent() {
            cmd.current_dir(parent);
        }
        cmd.arg("-m")
            .arg(&config.model_path)
            .arg("-p")
            .arg(prompt_tokens.to_string())
            .arg("-n")
            .arg(gen_tokens.to_string())
            .arg("-d")
            .arg(config.context_depth.to_string())
            .arg("-r")
            .arg(reps.to_string())
            .arg("-ngl")
            .arg(spec.ngl.to_string())
            .arg("-dev")
            .arg(spec.dev)
            .arg("-sm")
            .arg("none")
            .arg("-mg")
            .arg("0")
            .arg("-fa")
            .arg(if spec.fa { "1" } else { "0" })
            .arg("-ctk")
            .arg(spec.ctk)
            .arg("-ctv")
            .arg(spec.ctv)
            .arg("-t")
            .arg(plan.physical_cpu_cores.to_string())
            .arg("-o")
            .arg("json");

        for (k, v) in &spec.envs {
            cmd.env(k, v);
        }

        if spec.backend == "hip" {
            if let Some(ref rocm_dir) = config.rocm_lib_dir {
                let current_path = std::env::var_os("PATH").unwrap_or_default();
                let mut paths = vec![rocm_dir.clone()];
                if let Some(parent) = rocm_dir.parent().and_then(|p| p.parent()) {
                    let core_bin = parent.join("_rocm_sdk_core").join("bin");
                    if core_bin.exists() {
                        paths.push(core_bin);
                    }
                }
                paths.extend(std::env::split_paths(&current_path));
                if let Ok(joined) = std::env::join_paths(paths) {
                    cmd.env("PATH", joined);
                }
            }
        }

        let env_summary = if spec.envs.is_empty() {
            "default".to_string()
        } else {
            spec.envs
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let env_prefix = spec
            .envs
            .iter()
            .map(|(k, v)| format!("$env:{k}=\"{v}\"; "))
            .collect::<String>();
        let llama_server_cmd = format!(
            "{env_prefix}llama-server.exe -m \"{}\" -c {} -ngl {} -dev {} -sm none -mg 0 -fa {} -ctk {} -ctv {} -b 2048 -ub 512 -t {}",
            config.model_path.display(),
            config.context_depth.max(4096),
            spec.ngl,
            spec.dev,
            if spec.fa { "on" } else { "off" },
            spec.ctk,
            spec.ctv,
            plan.physical_cpu_cores
        );

        match cmd.output() {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                match parse_llama_bench_json(&stdout, spec.backend) {
                    Ok(entries) => {
                        let pp = valid_rate(
                            entries
                                .iter()
                                .find(|e| e.n_prompt > 0 && e.n_gen == 0)
                                .map(|e| e.median_tok_s),
                        );
                        let tg = valid_rate(
                            entries
                                .iter()
                                .find(|e| e.n_gen > 0 && e.n_prompt == 0)
                                .map(|e| e.median_tok_s),
                        );
                        // A run that produced no usable rate at all is a failure, not a success.
                        let error = (pp.is_none() && tg.is_none()).then(|| {
                            "llama-bench produced no finite, positive prompt-processing or token-generation rate"
                                .to_string()
                        });

                        candidates.push(LlmAutotuneCandidateResult {
                            profile_id: spec.id.to_string(),
                            title: spec.title.to_string(),
                            backend: spec.backend.to_string(),
                            n_gpu_layers: spec.ngl,
                            flash_attn: spec.fa,
                            cache_type_k: spec.ctk.to_string(),
                            cache_type_v: spec.ctv.to_string(),
                            env_summary,
                            succeeded: error.is_none(),
                            pp_tok_s: pp,
                            tg_tok_s: tg,
                            tg_speedup_vs_baseline: None,
                            pp_speedup_vs_baseline: None,
                            llama_server_cmd,
                            error,
                        });
                    }
                    Err(e) => {
                        candidates.push(LlmAutotuneCandidateResult {
                            profile_id: spec.id.to_string(),
                            title: spec.title.to_string(),
                            backend: spec.backend.to_string(),
                            n_gpu_layers: spec.ngl,
                            flash_attn: spec.fa,
                            cache_type_k: spec.ctk.to_string(),
                            cache_type_v: spec.ctv.to_string(),
                            env_summary,
                            succeeded: false,
                            pp_tok_s: None,
                            tg_tok_s: None,
                            tg_speedup_vs_baseline: None,
                            pp_speedup_vs_baseline: None,
                            llama_server_cmd,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                candidates.push(LlmAutotuneCandidateResult {
                    profile_id: spec.id.to_string(),
                    title: spec.title.to_string(),
                    backend: spec.backend.to_string(),
                    n_gpu_layers: spec.ngl,
                    flash_attn: spec.fa,
                    cache_type_k: spec.ctk.to_string(),
                    cache_type_v: spec.ctv.to_string(),
                    env_summary,
                    succeeded: false,
                    pp_tok_s: None,
                    tg_tok_s: None,
                    tg_speedup_vs_baseline: None,
                    pp_speedup_vs_baseline: None,
                    llama_server_cmd,
                    error: Some(stderr),
                });
            }
            Err(e) => {
                candidates.push(LlmAutotuneCandidateResult {
                    profile_id: spec.id.to_string(),
                    title: spec.title.to_string(),
                    backend: spec.backend.to_string(),
                    n_gpu_layers: spec.ngl,
                    flash_attn: spec.fa,
                    cache_type_k: spec.ctk.to_string(),
                    cache_type_v: spec.ctv.to_string(),
                    env_summary,
                    succeeded: false,
                    pp_tok_s: None,
                    tg_tok_s: None,
                    tg_speedup_vs_baseline: None,
                    pp_speedup_vs_baseline: None,
                    llama_server_cmd,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    // The designated baseline is the stock Vulkan profile. If it did not run or failed there is no
    // baseline: speedups stay empty instead of being measured against whichever candidate
    // happened to succeed first.
    let baseline_profile_id = BASELINE_PROFILE_ID
        .filter(|id| {
            candidates
                .iter()
                .any(|c| c.profile_id == *id && c.succeeded)
        })
        .map(str::to_string);
    if let Some(id) = &baseline_profile_id {
        apply_baseline_speedups(&mut candidates, id);
    }
    let (best_decode_profile_id, best_prefill_profile_id, best_balanced_profile_id) =
        select_autotune_winners(&candidates);

    Ok(LlmAutotuneReport {
        plan,
        prompt_tokens,
        gen_tokens,
        context_depth: config.context_depth,
        candidates,
        best_decode_profile_id,
        best_prefill_profile_id,
        best_balanced_profile_id,
        baseline_profile_id,
        balanced_score_formula: BALANCED_SCORE_FORMULA.to_string(),
        peak_rss_bytes: peak_process_rss_bytes(),
    })
}

/// Profile used as the speedup reference when the Vulkan backend is benchmarked.
const BASELINE_PROFILE_ID: Option<&str> = Some("vulkan_naive_fa1");

/// Weight of decode (token generation) in the balanced score.
///
/// Rationale: interactive chat spends most wall-clock time in decode, and prefill rates are
/// typically an order of magnitude higher, so an unweighted product would be dominated by
/// prefill. A weighted geometric mean `tg^w * pp^(1-w)` is scale-free (a 2x gain in either rate
/// moves the score by the same factor regardless of units). The 70/30 split is a documented
/// default, not a derived optimum: callers with prompt-heavy workloads should read the separate
/// `best_prefill_profile_id` instead.
pub const BALANCED_DECODE_WEIGHT: f64 = 0.7;
pub const BALANCED_PREFILL_WEIGHT: f64 = 1.0 - BALANCED_DECODE_WEIGHT;
const BALANCED_SCORE_FORMULA: &str =
    "score = tg_tok_s^0.7 * pp_tok_s^0.3 (weighted geometric mean, 70% decode / 30% prefill); only candidates with both rates, finite and > 0, are eligible";

/// Keeps only finite, strictly positive rates; NaN, infinity, zero and negatives are measurement failures.
fn valid_rate(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite() && *x > 0.0)
}

/// Weighted geometric mean of decode and prefill rates (see [`BALANCED_DECODE_WEIGHT`]).
pub fn balanced_score(tg_tok_s: f64, pp_tok_s: f64) -> f64 {
    tg_tok_s.powf(BALANCED_DECODE_WEIGHT) * pp_tok_s.powf(BALANCED_PREFILL_WEIGHT)
}

/// Fills `*_speedup_vs_baseline` relative to the successful candidate `baseline_id`.
/// Speedups are only computed from rates that exist (and are valid) in both candidates.
pub fn apply_baseline_speedups(candidates: &mut [LlmAutotuneCandidateResult], baseline_id: &str) {
    let Some(base) = candidates
        .iter()
        .find(|c| c.profile_id == baseline_id && c.succeeded)
    else {
        return;
    };
    let (base_pp, base_tg) = (valid_rate(base.pp_tok_s), valid_rate(base.tg_tok_s));
    for c in candidates.iter_mut().filter(|c| c.succeeded) {
        c.pp_speedup_vs_baseline = valid_rate(c.pp_tok_s)
            .zip(base_pp)
            .map(|(cur, base)| cur / base);
        c.tg_speedup_vs_baseline = valid_rate(c.tg_tok_s)
            .zip(base_tg)
            .map(|(cur, base)| cur / base);
    }
}

/// Picks `(best_decode, best_prefill, best_balanced)` profile ids.
///
/// Failed candidates and non-finite / non-positive rates are never eligible. Ties resolve to the
/// earliest candidate in probe order, so the result is deterministic.
pub fn select_autotune_winners(
    candidates: &[LlmAutotuneCandidateResult],
) -> (Option<String>, Option<String>, Option<String>) {
    fn best_by(
        candidates: &[LlmAutotuneCandidateResult],
        score: impl Fn(&LlmAutotuneCandidateResult) -> Option<f64>,
    ) -> Option<String> {
        let mut best: Option<(&LlmAutotuneCandidateResult, f64)> = None;
        for c in candidates.iter().filter(|c| c.succeeded) {
            if let Some(s) = score(c).filter(|s| s.is_finite()) {
                if best.is_none_or(|(_, b)| s > b) {
                    best = Some((c, s));
                }
            }
        }
        best.map(|(c, _)| c.profile_id.clone())
    }
    (
        best_by(candidates, |c| valid_rate(c.tg_tok_s)),
        best_by(candidates, |c| valid_rate(c.pp_tok_s)),
        best_by(candidates, |c| {
            valid_rate(c.tg_tok_s)
                .zip(valid_rate(c.pp_tok_s))
                .map(|(tg, pp)| balanced_score(tg, pp))
        }),
    )
}

/// Builds a merged LM Studio per-model JSON configuration (`user-concrete-model-default-config`),
/// preserving any unrelated user fields (such as custom Jinja prompt templates).
pub fn build_lm_studio_model_config_json(
    existing_json: Option<&str>,
    profile: &LlmTweakProfile,
    block_count: u32,
    context_length: u32,
    disabled_gpus: &[usize],
) -> (String, String) {
    let mut root = existing_json
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .filter(|v| v.is_object())
        .unwrap_or_else(|| {
            serde_json::json!({
                "preset": "",
                "operation": { "fields": [] },
                "load": { "fields": [] }
            })
        });

    let managed_keys = [
        "load.gpuSplitConfig",
        "llm.load.contextLength",
        "llm.load.llama.cpuThreadPoolSize",
        "llm.load.llama.flashAttention",
        "llm.load.llama.kCacheQuantizationType",
        "llm.load.llama.vCacheQuantizationType",
        "llm.load.llama.acceleration.offloadRatio",
        "llm.load.llama.autoFit",
        "llm.load.numParallelSessions",
    ];

    let mut preserved_fields: Vec<serde_json::Value> = root
        .get("load")
        .and_then(|l| l.get("fields"))
        .and_then(|f| f.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|entry| {
            entry
                .get("key")
                .and_then(|k| k.as_str())
                .is_none_or(|k| !managed_keys.contains(&k))
        })
        .collect();

    let (offload_val, offload_str) = if profile.n_gpu_layers >= block_count {
        (serde_json::json!(1), "1.00 (100% GPU)".to_string())
    } else {
        let r = ((profile.n_gpu_layers as f64 / f64::from(block_count.max(1))) * 10_000.0).round()
            / 10_000.0;
        (
            serde_json::json!(r),
            format!("{r:.4} ({}/{} layers)", profile.n_gpu_layers, block_count),
        )
    };

    if !disabled_gpus.is_empty() {
        preserved_fields.push(serde_json::json!({
            "key": "load.gpuSplitConfig",
            "value": {
                "strategy": "evenly",
                "disabledGpus": disabled_gpus,
                "priority": [],
                "customRatio": []
            }
        }));
    }

    preserved_fields.extend([
        serde_json::json!({
            "key": "llm.load.contextLength",
            "value": context_length
        }),
        serde_json::json!({
            "key": "llm.load.llama.cpuThreadPoolSize",
            "value": profile.threads
        }),
        serde_json::json!({
            "key": "llm.load.llama.flashAttention",
            "value": profile.flash_attn
        }),
        serde_json::json!({
            "key": "llm.load.llama.kCacheQuantizationType",
            "value": {
                "checked": profile.cache_type_k != "f16",
                "value": profile.cache_type_k
            }
        }),
        serde_json::json!({
            "key": "llm.load.llama.vCacheQuantizationType",
            "value": {
                "checked": profile.cache_type_v != "f16",
                "value": profile.cache_type_v
            }
        }),
        serde_json::json!({
            "key": "llm.load.llama.acceleration.offloadRatio",
            "value": offload_val
        }),
        serde_json::json!({
            "key": "llm.load.llama.autoFit",
            "value": false
        }),
        serde_json::json!({
            "key": "llm.load.numParallelSessions",
            "value": 1
        }),
    ]);

    if let Some(obj) = root.as_object_mut() {
        if !obj.contains_key("preset") {
            obj.insert("preset".to_string(), serde_json::json!(""));
        }
        if !obj.contains_key("operation") {
            obj.insert("operation".to_string(), serde_json::json!({ "fields": [] }));
        }
        obj.insert(
            "load".to_string(),
            serde_json::json!({ "fields": preserved_fields }),
        );
    }

    let formatted = serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".to_string());
    (formatted, offload_str)
}

fn detect_lmstudio_backend_version(lms_root: &Path, backend_id: &str) -> Option<String> {
    let backends_dir = lms_root.join("extensions").join("backends");
    let entries = std::fs::read_dir(backends_dir).ok()?;
    let prefix = format!("{backend_id}-");
    let mut best: Option<String> = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix(&prefix) {
            let trimmed = rest.trim_start_matches('v').trim();
            if !trimmed.is_empty() {
                best = Some(trimmed.to_string());
            }
        } else if name == backend_id {
            let manifest = entry.path().join("manifest.json");
            if let Ok(raw) = std::fs::read_to_string(manifest) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(ver) = val.get("version").and_then(|v| v.as_str()) {
                        best = Some(ver.to_string());
                    }
                }
            }
        }
    }
    best
}

/// Previews (`dry_run = true`) or writes (`dry_run = false`, with `.bak` backups) optimized
/// LM Studio per-model load settings and backend selection for the given GGUF model and target GPU.
pub fn apply_lm_studio_config(
    model_path: &Path,
    device_index: Option<usize>,
    profile_id: Option<&str>,
    context_tokens: u32,
    safety_margin_mib: u64,
    dry_run: bool,
) -> ForgeResult<LmStudioPatchResult> {
    let meta = inspect_gguf_model(model_path)?;
    let plan =
        plan_llm_configuration_for_device(&meta, device_index, context_tokens, safety_margin_mib)?;

    let profile = match profile_id.map(str::trim).filter(|s| !s.is_empty()) {
        Some(want) => plan
            .recommended_profiles
            .iter()
            .find(|p| p.id == want)
            .ok_or_else(|| {
                ForgeError::InvalidParameter(format!(
                    "unknown LLM tweak profile '{want}' (available: {})",
                    plan.recommended_profiles
                        .iter()
                        .map(|p| p.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?,
        None => plan.recommended_profiles.first().ok_or_else(|| {
            ForgeError::ExecutionFailed("no recommended LLM profiles generated".to_string())
        })?,
    };

    let home = std::env::var("USERPROFILE").map_err(|_| ForgeError::RuntimeUnavailable {
        runtime: "LM Studio".to_string(),
        reason: "USERPROFILE environment variable is not set".to_string(),
    })?;
    let lms_root = PathBuf::from(home).join(".lmstudio");
    let internal_dir = lms_root.join(".internal");
    if !internal_dir.is_dir() {
        return Err(ForgeError::RuntimeUnavailable {
            runtime: "LM Studio".to_string(),
            reason: format!(
                "LM Studio config directory not found at '{}'",
                internal_dir.display()
            ),
        });
    }

    let models_root = lms_root.join("models");
    let rel_model_path = model_path
        .strip_prefix(&models_root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| PathBuf::from(&meta.file_name));

    let model_config_path = internal_dir
        .join("user-concrete-model-default-config")
        .join(format!("{}.json", rel_model_path.display()));
    let backend_prefs_path = internal_dir.join("backend-preferences-v2.json");

    let existing_model_json = std::fs::read_to_string(&model_config_path).ok();
    let (patched_model_json_preview, offload_ratio) = build_lm_studio_model_config_json(
        existing_model_json.as_deref(),
        profile,
        meta.block_count,
        plan.target_context_tokens,
        &plan.disabled_gpu_indices,
    );

    let selected_lmstudio_backend = if profile.backend == "hip" {
        "llama.cpp-win-x86_64-amd-rocm-avx2".to_string()
    } else {
        "llama.cpp-win-x86_64-vulkan-avx2".to_string()
    };

    let mut backend_root = std::fs::read_to_string(&backend_prefs_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .filter(|v| v.is_object())
        .unwrap_or_else(|| {
            serde_json::json!({
                "formatDefaults": {},
                "versions": {}
            })
        });

    if let Some(obj) = backend_root.as_object_mut() {
        let mut fmt_defaults = obj
            .get("formatDefaults")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        fmt_defaults.insert(
            "gguf".to_string(),
            serde_json::Value::String(selected_lmstudio_backend.clone()),
        );
        obj.insert(
            "formatDefaults".to_string(),
            serde_json::Value::Object(fmt_defaults),
        );

        let mut versions = obj
            .get("versions")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        if !versions.contains_key(&selected_lmstudio_backend) {
            if let Some(detected_ver) =
                detect_lmstudio_backend_version(&lms_root, &selected_lmstudio_backend)
            {
                versions.insert(
                    selected_lmstudio_backend.clone(),
                    serde_json::Value::String(detected_ver),
                );
            }
        }
        obj.insert("versions".to_string(), serde_json::Value::Object(versions));
    }

    let patched_backend_json_preview =
        serde_json::to_string_pretty(&backend_root).unwrap_or_else(|_| "{}".to_string());

    let mut model_config_backup_path = None;
    let mut backend_prefs_backup_path = None;

    if !dry_run {
        if let Some(parent) = model_config_path.parent() {
            std::fs::create_dir_all(parent).map_err(ForgeError::Io)?;
        }
        if model_config_path.exists() {
            let bak = model_config_path.with_extension("json.bak");
            std::fs::copy(&model_config_path, &bak).map_err(ForgeError::Io)?;
            model_config_backup_path = Some(bak.display().to_string());
        }
        std::fs::write(&model_config_path, &patched_model_json_preview).map_err(ForgeError::Io)?;

        if backend_prefs_path.exists() {
            let bak = backend_prefs_path.with_extension("json.bak");
            std::fs::copy(&backend_prefs_path, &bak).map_err(ForgeError::Io)?;
            backend_prefs_backup_path = Some(bak.display().to_string());
        }
        std::fs::write(&backend_prefs_path, &patched_backend_json_preview)
            .map_err(ForgeError::Io)?;
    }

    Ok(LmStudioPatchResult {
        applied: !dry_run,
        profile_id: profile.id.clone(),
        profile_title: profile.title.clone(),
        model_config_path: model_config_path.display().to_string(),
        model_config_backup_path,
        backend_prefs_path: backend_prefs_path.display().to_string(),
        backend_prefs_backup_path,
        selected_lmstudio_backend,
        offload_ratio,
        context_length: plan.target_context_tokens,
        flash_attention: profile.flash_attn,
        cache_type_k: profile.cache_type_k.clone(),
        cache_type_v: profile.cache_type_v.clone(),
        cpu_threads: profile.threads,
        disabled_igpu: !plan.disabled_gpu_indices.is_empty(),
        patched_model_json_preview,
        patched_backend_json_preview,
    })
}

fn query_hkcu_env_var(key: &str) -> Option<String> {
    let out = Command::new("reg.exe")
        .args(["query", r"HKCU\Environment", "/v", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(key) {
            let parts: Vec<&str> = trimmed.splitn(3, char::is_whitespace).collect();
            if let Some(last) = trimmed.split("REG_SZ").nth(1).or_else(|| {
                let _ = parts;
                trimmed.split("REG_EXPAND_SZ").nth(1)
            }) {
                return Some(last.trim().to_string());
            }
        }
    }
    None
}

/// Previews (`dry_run = true`) or writes (`dry_run = false`) hardware-tailored Windows User Environment
/// variables (`HKCU\Environment`, inherited by both LM Studio and Ollama) and generates an optimized Ollama `Modelfile`.
pub fn apply_windows_and_ollama_env_tweaks(
    model_path: &Path,
    device_index: Option<usize>,
    context_tokens: u32,
    kv_quant: KvCacheQuant,
    modelfile_out: Option<&Path>,
    dry_run: bool,
) -> ForgeResult<EnvAndOllamaPatchResult> {
    let meta = inspect_gguf_model(model_path)?;
    let plan = plan_llm_configuration_for_device(&meta, device_index, context_tokens, 512)?;

    let supports_hipblaslt = matches!(
        plan.gpu_architecture.as_str(),
        "RDNA3" | "RDNA3.5" | "RDNA4" | "CDNA2" | "CDNA3" | "CDNA4"
    );

    let mut desired: Vec<(&'static str, String, String)> = vec![
        (
            "GGML_VK_ENABLE_MEMORY_PRIORITY",
            "1".to_string(),
            "Enables VK_EXT_memory_priority so Windows WDDM keeps LLM weights resident in physical VRAM".to_string(),
        ),
        (
            "GGML_VK_VISIBLE_DEVICES",
            plan.vk_visible_device.to_string(),
            format!(
                "Pins Vulkan llama.cpp / LM Studio execution to {} (Vulkan index {})",
                plan.gpu_name, plan.vk_visible_device
            ),
        ),
    ];

    if plan.gpu_architecture != "RDNA1" {
        desired.push((
            "HIP_VISIBLE_DEVICES",
            plan.hip_visible_device.to_string(),
            format!(
                "Pins ROCm/HIP execution to {} (HIP index {}) and prevents integrated GPU layer split",
                plan.gpu_name, plan.hip_visible_device
            ),
        ));
    }

    if supports_hipblaslt {
        desired.push((
            "ROCBLAS_USE_HIPBLASLT",
            "1".to_string(),
            format!(
                "Enables hipBLASLt WMMA matrix-core paths on {} ({})",
                plan.gpu_architecture, plan.gfx_target
            ),
        ));
    }

    desired.extend([
        (
            "OLLAMA_FLASH_ATTENTION",
            "1".to_string(),
            "Enables Flash Attention in Ollama to reduce long-context VRAM usage".to_string(),
        ),
        (
            "OLLAMA_KV_CACHE_TYPE",
            kv_quant.as_cli_str().to_string(),
            "Quantizes Ollama KV cache precision to save 50%-72% context VRAM".to_string(),
        ),
        (
            "OLLAMA_NUM_PARALLEL",
            "1".to_string(),
            "Prevents Ollama from multiplying KV cache allocation across parallel slots"
                .to_string(),
        ),
    ]);

    let mut env_changes = Vec::with_capacity(desired.len());
    for (k, v, purpose) in desired {
        let prev = query_hkcu_env_var(k);
        if !dry_run {
            let status = Command::new("reg.exe")
                .args([
                    "add",
                    r"HKCU\Environment",
                    "/v",
                    k,
                    "/t",
                    "REG_SZ",
                    "/d",
                    &v,
                    "/f",
                ])
                .status()
                .map_err(ForgeError::Io)?;
            if !status.success() {
                return Err(ForgeError::ExecutionFailed(format!(
                    "reg.exe failed to set HKCU\\Environment\\{k}"
                )));
            }
        }
        env_changes.push(EnvVarChange {
            key: k.to_string(),
            previous_value: prev,
            new_value: v,
            purpose,
        });
    }

    let rec_ngl = plan
        .recommended_profiles
        .first()
        .map(|p| p.n_gpu_layers)
        .unwrap_or(99);
    let ollama_modelfile_content = format!(
        "# Generated by RDNA Forge for {} ({}, {})\n# Model: {} ({}, {} layers)\nFROM \"{}\"\nPARAMETER num_ctx {}\nPARAMETER num_gpu {}\nPARAMETER num_thread {}\nPARAMETER num_batch 512\n",
        plan.gpu_name,
        plan.gpu_architecture,
        plan.gfx_target,
        meta.file_name,
        meta.quantization_name,
        meta.block_count,
        meta.file_path,
        plan.target_context_tokens,
        rec_ngl,
        plan.physical_cpu_cores
    );

    let mut written_path = None;
    if let Some(out_path) = modelfile_out {
        written_path = Some(out_path.display().to_string());
        if !dry_run {
            if let Some(parent) = out_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(ForgeError::Io)?;
                }
            }
            std::fs::write(out_path, &ollama_modelfile_content).map_err(ForgeError::Io)?;
        }
    }

    let model_slug = meta
        .model_name
        .as_deref()
        .unwrap_or(&meta.architecture)
        .to_ascii_lowercase()
        .replace(|c: char| !c.is_ascii_alphanumeric(), "-");
    let mf_arg = written_path.as_deref().unwrap_or("Modelfile.rdna-forge");
    let ollama_create_cmd = format!("ollama create {model_slug}-rdna -f \"{mf_arg}\"");

    Ok(EnvAndOllamaPatchResult {
        applied: !dry_run,
        env_changes,
        ollama_modelfile_content,
        ollama_modelfile_path: written_path,
        ollama_create_cmd,
    })
}

fn collect_json_bak_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_bak_files(&path, out);
        } else if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".json.bak"))
        {
            out.push(path);
        }
    }
}

fn target_path_from_json_bak(bak_path: &Path) -> Option<PathBuf> {
    let raw = bak_path.to_string_lossy();
    raw.strip_suffix(".bak").map(PathBuf::from)
}

/// Previews (`dry_run = true`) or restores (`dry_run = false`) LM Studio `.json.bak` backups
/// under the provided LM Studio root directory (typically `%USERPROFILE%\.lmstudio`).
pub fn rollback_lm_studio_config_in_dir(
    lms_root: &Path,
    model_path: Option<&Path>,
    dry_run: bool,
) -> ForgeResult<LmStudioRollbackResult> {
    let internal_dir = lms_root.join(".internal");
    if !internal_dir.is_dir() {
        return Err(ForgeError::RuntimeUnavailable {
            runtime: "LM Studio".to_string(),
            reason: format!(
                "LM Studio config directory not found at '{}'",
                internal_dir.display()
            ),
        });
    }

    let mut backups: Vec<PathBuf> = Vec::new();
    let backend_bak = internal_dir.join("backend-preferences-v2.json.bak");
    if backend_bak.is_file() {
        backups.push(backend_bak);
    }

    let model_cfg_root = internal_dir.join("user-concrete-model-default-config");
    if let Some(mp) = model_path {
        let models_root = lms_root.join("models");
        let file_name = mp
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let rel = mp
            .strip_prefix(&models_root)
            .map(Path::to_path_buf)
            .unwrap_or_else(|_| PathBuf::from(&file_name));
        let candidate = model_cfg_root.join(format!("{}.json.bak", rel.display()));
        if candidate.is_file() {
            backups.push(candidate);
        } else if !file_name.is_empty() {
            let flat = model_cfg_root.join(format!("{file_name}.json.bak"));
            if flat.is_file() {
                backups.push(flat);
            }
        }
    } else {
        collect_json_bak_files(&model_cfg_root, &mut backups);
    }
    backups.sort();
    backups.dedup();

    let mut actions = Vec::with_capacity(backups.len());
    for bak in backups {
        let Some(target) = target_path_from_json_bak(&bak) else {
            continue;
        };
        let restored = if !dry_run {
            std::fs::copy(&bak, &target).map_err(ForgeError::Io)?;
            true
        } else {
            false
        };
        actions.push(RollbackFileAction {
            target_path: target.display().to_string(),
            backup_path: bak.display().to_string(),
            restored,
        });
    }

    Ok(LmStudioRollbackResult {
        applied: !dry_run,
        actions,
    })
}

/// Previews (`dry_run = true`) or restores (`dry_run = false`) LM Studio `.json.bak` backups
/// under `%USERPROFILE%\.lmstudio`.
pub fn rollback_lm_studio_config(
    model_path: Option<&Path>,
    dry_run: bool,
) -> ForgeResult<LmStudioRollbackResult> {
    let home = std::env::var("USERPROFILE").map_err(|_| ForgeError::RuntimeUnavailable {
        runtime: "LM Studio".to_string(),
        reason: "USERPROFILE environment variable is not set".to_string(),
    })?;
    let lms_root = PathBuf::from(home).join(".lmstudio");
    rollback_lm_studio_config_in_dir(&lms_root, model_path, dry_run)
}

/// Previews (`dry_run = true`) or deletes (`dry_run = false`) RDNA Forge-managed keys
/// from `HKCU\Environment` (`GGML_VK_*`, `HIP_VISIBLE_DEVICES`, `ROCBLAS_USE_HIPBLASLT`, `OLLAMA_*`).
pub fn rollback_windows_env_tweaks(dry_run: bool) -> ForgeResult<EnvRollbackResult> {
    let mut keys = Vec::with_capacity(RDNA_FORGE_MANAGED_ENV_KEYS.len());
    for &key in RDNA_FORGE_MANAGED_ENV_KEYS {
        let current_value = query_hkcu_env_var(key);
        let removed = if !dry_run && current_value.is_some() {
            let status = Command::new("reg.exe")
                .args(["delete", r"HKCU\Environment", "/v", key, "/f"])
                .status()
                .map_err(ForgeError::Io)?;
            if !status.success() {
                return Err(ForgeError::ExecutionFailed(format!(
                    "reg.exe failed to delete HKCU\\Environment\\{key}"
                )));
            }
            true
        } else {
            false
        };
        keys.push(EnvRollbackItem {
            key: key.to_string(),
            current_value,
            removed,
        });
    }

    Ok(EnvRollbackResult {
        applied: !dry_run,
        keys,
    })
}
