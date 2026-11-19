use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use forge_core::{ForgeError, ForgeResult};
use forge_device::discover_all;
use forge_profiler::snapshot_all_gpus;
use serde::{Deserialize, Serialize};

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
    pub bytes_per_layer: u64,
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
    pub model_weights_mib: u64,
    pub kv_and_state_mib: u64,
    pub compute_scratch_mib: u64,
    pub total_required_mib: u64,
    pub fits_in_free_vram: bool,
    pub fits_with_wddm_memory_priority: bool,
    pub strict_zero_spill_ngl: u32,
    pub memory_priority_ngl: u32,
}

/// A concrete, hardware-tested `llama.cpp` execution profile for AMD RDNA2 (`gfx1030`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmTweakProfile {
    pub id: String,
    pub title: String,
    pub backend: String,
    pub device_selector: String,
    pub n_gpu_layers: u32,
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

/// Complete static analysis, VRAM anti-spill budget, and RDNA2 tweak recommendations for a GGUF model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmOptimizationPlan {
    pub model: GgufModelMetadata,
    pub gpu_name: String,
    pub gfx_target: String,
    pub has_secondary_igpu: bool,
    pub secondary_igpu_name: Option<String>,
    pub physical_cpu_cores: u32,
    pub total_vram_mib: u64,
    pub used_vram_mib: u64,
    pub free_vram_mib: u64,
    pub safety_margin_mib: u64,
    pub target_context_tokens: u32,
    pub recoverable_layers_if_apps_closed: u32,
    pub warnings: Vec<String>,
    pub kv_budgets: Vec<KvCacheMemoryBudget>,
    pub recommended_profiles: Vec<LlmTweakProfile>,
    pub lm_studio_tips: Vec<String>,
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
    pub peak_rss_bytes: Option<u64>,
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
    let mut reader = BufReader::new(file);

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

    let block_count = get_arch_u32("block_count").unwrap_or(32).max(1);
    let nextn_predict_layers = get_arch_u32("nextn_predict_layers").unwrap_or(0);
    let declared_context_length = get_arch_u32("context_length").unwrap_or(4096);
    let embedding_length = get_arch_u32("embedding_length").unwrap_or(4096);
    let feed_forward_length = get_arch_u32("feed_forward_length");
    let head_count = get_arch_u32("attention.head_count").unwrap_or(32).max(1);
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

    let file_type_code = kvs.get("general.file_type").and_then(GgufValue::as_u32);
    let quantization_name = gguf_file_type_name(file_type_code, &file_name);
    let total_layers = u64::from(block_count.max(1));
    let bytes_per_layer = file_size_bytes / total_layers;

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
        bytes_per_layer,
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

    let kv_elems_per_token_per_layer =
        u64::from(meta.head_count_kv) * u64::from(meta.key_length + meta.value_length);
    let total_kv_elems =
        kv_elems_per_token_per_layer * u64::from(attn_layers) * u64::from(context_tokens);
    let kv_bytes = (total_kv_elems as f64 * kv_quant.bytes_per_element()).ceil() as u64;

    let ssm_bytes_per_layer = match (meta.ssm_inner_size, meta.ssm_state_size) {
        (Some(inner), Some(state)) => u64::from(inner) * u64::from(state) * 4 + 512 * 1024,
        _ => 4 * 1024 * 1024,
    };
    let ssm_bytes = u64::from(ssm_layers) * ssm_bytes_per_layer;

    kv_bytes + ssm_bytes
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

/// Builds a hardware-aware anti-WDDM-spill memory plan and RDNA2 tweak matrix for `model`.
pub fn plan_llm_configuration(
    model: &GgufModelMetadata,
    target_context_tokens: u32,
    safety_margin_mib: u64,
) -> ForgeResult<LlmOptimizationPlan> {
    let target_ctx = target_context_tokens.clamp(512, 262_144);
    let snapshot = discover_all();
    let primary_gpu = snapshot
        .gpus
        .iter()
        .find(|g| g.discrete)
        .or_else(|| snapshot.gpus.first());

    let gpu_name = primary_gpu
        .map(|g| g.name.clone())
        .unwrap_or_else(|| "AMD Radeon GPU".to_string());
    let gfx_target = primary_gpu
        .and_then(|g| g.gfx_target.as_ref().map(|t| t.base.clone()))
        .unwrap_or_else(|| "gfx1030".to_string());
    let secondary_igpu = snapshot.gpus.iter().find(|g| !g.discrete);
    let has_secondary_igpu = secondary_igpu.is_some();
    let secondary_igpu_name = secondary_igpu.map(|g| g.name.clone());

    let logical_cores = snapshot.system.logical_cores;
    let physical_cpu_cores = (logical_cores / 2).clamp(4, 16) as u32;

    let telemetry = snapshot_all_gpus().ok().unwrap_or_default();
    let active_tel = telemetry
        .iter()
        .find(|s| s.discrete)
        .or_else(|| telemetry.first());

    let total_vram_mib: u64 = active_tel
        .and_then(|s| s.total_vram_mib.map(u64::from))
        .or_else(|| primary_gpu.map(|g| g.vram_bytes / (1024 * 1024)))
        .unwrap_or(16_368);
    let used_vram_mib: u64 = active_tel
        .and_then(|s| s.vram_used_mib.map(|v| v.max(0) as u64))
        .unwrap_or(2_048)
        .min(total_vram_mib);
    let free_vram_mib: u64 = total_vram_mib.saturating_sub(used_vram_mib);

    let model_weights_mib = model.file_size_bytes.div_ceil(1024 * 1024);
    let mib_per_layer = (model_weights_mib as f64 / f64::from(model.block_count.max(1))).max(1.0);
    let compute_scratch_mib = 384u64;

    let mut kv_budgets = Vec::with_capacity(3);
    for kv_q in [KvCacheQuant::F16, KvCacheQuant::Q8_0, KvCacheQuant::Q4_0] {
        let kv_bytes = estimate_kv_and_state_bytes(model, target_ctx, kv_q);
        let kv_and_state_mib = kv_bytes.div_ceil(1024 * 1024);
        let total_required_mib = model_weights_mib + kv_and_state_mib + compute_scratch_mib;

        let usable_free_mib = free_vram_mib.saturating_sub(safety_margin_mib);
        let fits_in_free_vram = total_required_mib <= usable_free_mib;

        // With GGML_VK_ENABLE_MEMORY_PRIORITY=1 (VK_EXT_memory_priority), Windows WDDM VidMm
        // can page out non-pinned background app VRAM down to ~1100 MiB OS baseline.
        let priority_budget_mib = total_vram_mib.saturating_sub(1_150 + safety_margin_mib / 2);
        let fits_with_wddm_memory_priority = total_required_mib <= priority_budget_mib;

        let budget_for_weights_strict =
            usable_free_mib.saturating_sub(kv_and_state_mib + compute_scratch_mib);
        let strict_zero_spill_ngl = ((budget_for_weights_strict as f64 / mib_per_layer).floor()
            as u32)
            .min(model.block_count);

        let budget_for_weights_prio =
            priority_budget_mib.saturating_sub(kv_and_state_mib + compute_scratch_mib);
        let memory_priority_ngl = ((budget_for_weights_prio as f64 / mib_per_layer).floor() as u32)
            .min(model.block_count);

        kv_budgets.push(KvCacheMemoryBudget {
            kv_quant: kv_q.as_cli_str().to_string(),
            requires_flash_attn: kv_q.requires_flash_attn_for_v(),
            context_tokens: target_ctx,
            model_weights_mib,
            kv_and_state_mib,
            compute_scratch_mib,
            total_required_mib,
            fits_in_free_vram,
            fits_with_wddm_memory_priority,
            strict_zero_spill_ngl,
            memory_priority_ngl,
        });
    }

    let baseline_os_vram_mib = 1_150u64;
    let reclaimable_vram_mib = used_vram_mib.saturating_sub(baseline_os_vram_mib);
    let recoverable_layers_if_apps_closed =
        ((reclaimable_vram_mib as f64 / mib_per_layer).floor() as u32).min(model.block_count);

    let mut warnings = Vec::new();
    if model.declared_context_length > 16_384 {
        warnings.push(format!(
            "Model metadata declares a default context of {} tokens ({}K). Loading without an explicit context cap (-c {}) will over-allocate VRAM and spill into system RAM.",
            model.declared_context_length,
            model.declared_context_length / 1024,
            target_ctx
        ));
    }
    if has_secondary_igpu {
        warnings.push(format!(
            "Secondary integrated GPU ({}) detected. On Windows ROCm, this iGPU reports ~12 GiB of shared DDR5 RAM as 'VRAM', causing llama.cpp's default '-sm layer' to split layers onto the slow iGPU unless HIP_VISIBLE_DEVICES=0 / GGML_VK_VISIBLE_DEVICES=0 and '-sm none' are set.",
            secondary_igpu_name.as_deref().unwrap_or("Radeon iGPU")
        ));
    }
    if !kv_budgets[0].fits_in_free_vram {
        warnings.push(format!(
            "Model weights ({} MiB) + KV/scratch ({} MiB) = {} MiB exceed currently free physical VRAM ({} MiB; {} MiB currently occupied by Windows/apps). Enable GGML_VK_ENABLE_MEMORY_PRIORITY=1 and/or quantized KV cache (-ctk q4_0 -ctv q4_0) or cap -ngl to {} to prevent WDDM PCIe thrashing.",
            model_weights_mib,
            kv_budgets[0].kv_and_state_mib + compute_scratch_mib,
            kv_budgets[0].total_required_mib,
            free_vram_mib,
            used_vram_mib,
            kv_budgets[0].strict_zero_spill_ngl
        ));
    }

    let is_hybrid_ssm = model.full_attention_interval.unwrap_or(1) > 1;
    let q4_budget = &kv_budgets[2];
    let f16_budget = &kv_budgets[0];
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
    let rocm_q4_ngl = if q4_budget.fits_in_free_vram {
        99
    } else {
        q4_budget
            .strict_zero_spill_ngl
            .max(q4_budget.memory_priority_ngl.saturating_sub(2))
    };

    let mut recommended_profiles = Vec::new();

    // Profile 1: Vulkan WDDM High-Priority Interactive (Verified fastest on Qwen3.8-27B up to 2K-3K context)
    recommended_profiles.push(LlmTweakProfile {
        id: "vulkan_wddm_priority".to_string(),
        title: "Vulkan WDDM High-Priority (Fastest Interactive Chat <= 2K-3K Context)".to_string(),
        backend: "vulkan".to_string(),
        device_selector: "Vulkan0".to_string(),
        n_gpu_layers: vk_prio_ngl,
        split_mode: "none".to_string(),
        flash_attn: !is_hybrid_ssm,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        batch_size: 2048,
        ubatch_size: 512,
        threads: physical_cpu_cores,
        env_vars: vec![
            ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
            ("GGML_VK_VISIBLE_DEVICES".to_string(), "0".to_string()),
        ],
        rationale: if is_hybrid_ssm {
            "Activates VK_EXT_memory_priority (1.0) so Windows WDDM keeps all 27B weights resident in RX 6900 XT VRAM (evicting background desktop allocations first), hides the Ryzen iGPU, and disables Vulkan FlashAttention (-fa 0) which suffers a 15x prefill slowdown at depth on hybrid qwen35 layers. Verified on RX 6900 XT + Qwen3.8-27B: 343.2 tok/s prefill (+108%) and 26.81 tok/s generation (+63%) at 2048 context.".to_string()
        } else {
            "Activates VK_EXT_memory_priority (1.0) to prevent Windows WDDM from paging LLM weights out to system RAM and isolates execution strictly to the discrete RX 6900 XT.".to_string()
        },
        llama_server_cmd: format!(
            "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"0\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev Vulkan0 -sm none -mg 0 -fa {} -ctk f16 -ctv f16 -b 2048 -ub 512 -t {}",
            model.file_path,
            target_ctx,
            vk_prio_ngl,
            if is_hybrid_ssm { "off" } else { "on" },
            physical_cpu_cores
        ),
    });

    // Profile 2: ROCm/HIP Long-Context & Heavy Prompt Ingestion (q4_0 KV Cache + FlashAttention)
    recommended_profiles.push(LlmTweakProfile {
        id: "rocm_q4_kv_long_ctx".to_string(),
        title: "ROCm / HIP FlashAttention + Q4_0 KV Cache (Fastest Prefill & 4K-16K Context)".to_string(),
        backend: "hip".to_string(),
        device_selector: "ROCm0".to_string(),
        n_gpu_layers: rocm_q4_ngl,
        split_mode: "none".to_string(),
        flash_attn: true,
        cache_type_k: "q4_0".to_string(),
        cache_type_v: "q4_0".to_string(),
        batch_size: 2048,
        ubatch_size: 512,
        threads: physical_cpu_cores,
        env_vars: vec![("HIP_VISIBLE_DEVICES".to_string(), "0".to_string())],
        rationale: "Hides the Ryzen 7700X iGPU (which ROCm misreports as a 12.4 GiB GPU), enables native ROCm FlashAttention (-fa 1), and compresses the KV cache 3.5x (-ctk q4_0 -ctv q4_0). Verified on RX 6900 XT + Qwen3.8-27B: 522.6 tok/s initial prefill, 246.2 tok/s prefill at 4096 context (20.3x faster than Vulkan -fa 1), and 27.2 -> 22.1 -> 19.0 tok/s generation.".to_string(),
        llama_server_cmd: format!(
            "$env:HIP_VISIBLE_DEVICES=\"0\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev ROCm0 -sm none -mg 0 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t {}",
            model.file_path, target_ctx, rocm_q4_ngl, physical_cpu_cores
        ),
    });

    // Profile 3: Vulkan Long-Context Decode with Q4_0 KV Cache
    recommended_profiles.push(LlmTweakProfile {
        id: "vulkan_q4_kv_decode".to_string(),
        title: "Vulkan WDDM Priority + Q4_0 KV Cache (Max Token Generation Speed at 4K+ Context)".to_string(),
        backend: "vulkan".to_string(),
        device_selector: "Vulkan0".to_string(),
        n_gpu_layers: vk_q4_ngl,
        split_mode: "none".to_string(),
        flash_attn: true,
        cache_type_k: "q4_0".to_string(),
        cache_type_v: "q4_0".to_string(),
        batch_size: 2048,
        ubatch_size: 512,
        threads: physical_cpu_cores,
        env_vars: vec![
            ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
            ("GGML_VK_VISIBLE_DEVICES".to_string(), "0".to_string()),
        ],
        rationale: "Combines VK_EXT_memory_priority with 4-bit KV cache quantization (-ctk q4_0 -ctv q4_0 -fa 1) to keep 4K+ context entirely inside RX 6900 XT physical VRAM. Verified on RX 6900 XT + Qwen3.8-27B: sustains 25.63 tok/s token generation at 4096 context (+61% faster than f16 KV cache at 4K depth).".to_string(),
        llama_server_cmd: format!(
            "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"0\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev Vulkan0 -sm none -mg 0 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t {}",
            model.file_path, target_ctx, vk_q4_ngl, physical_cpu_cores
        ),
    });

    // Profile 4: Strict Zero-Spill Hybrid GPU+CPU Split (Guaranteed fit without evicting background apps)
    if f16_budget.strict_zero_spill_ngl < model.block_count {
        let safe_ngl = kv_budgets[1].strict_zero_spill_ngl; // q8_0 strict fit
        recommended_profiles.push(LlmTweakProfile {
            id: "vulkan_strict_zero_spill".to_string(),
            title: format!(
                "Strict Zero-Spill Hybrid ({}/{} GPU Layers + Ryzen {} Threads)",
                safe_ngl, model.block_count, physical_cpu_cores
            ),
            backend: "vulkan".to_string(),
            device_selector: "Vulkan0".to_string(),
            n_gpu_layers: safe_ngl,
            split_mode: "none".to_string(),
            flash_attn: false,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "f16".to_string(),
            batch_size: 2048,
            ubatch_size: 512,
            threads: physical_cpu_cores,
            env_vars: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY".to_string(), "1".to_string()),
                ("GGML_VK_VISIBLE_DEVICES".to_string(), "0".to_string()),
            ],
            rationale: format!(
                "Caps GPU layers to exactly {}/{} ({} MiB safety margin) so the model fits strictly inside currently unallocated VRAM ({} MiB) while heavy background apps remain open.",
                safe_ngl, model.block_count, safety_margin_mib, free_vram_mib
            ),
            llama_server_cmd: format!(
                "$env:GGML_VK_ENABLE_MEMORY_PRIORITY=\"1\"; $env:GGML_VK_VISIBLE_DEVICES=\"0\"; llama-server.exe -m \"{}\" -c {} -ngl {} -dev Vulkan0 -sm none -mg 0 -fa off -ctk q8_0 -ctv f16 -b 2048 -ub 512 -t {}",
                model.file_path, target_ctx, safe_ngl, physical_cpu_cores
            ),
        });
    }

    let lm_studio_tips = vec![
        "1. Set Windows User Environment Variables: GGML_VK_ENABLE_MEMORY_PRIORITY=1, GGML_VK_VISIBLE_DEVICES=0, and HIP_VISIBLE_DEVICES=0 (LM Studio's llama.cpp 2.73.0 backend reads these directly, boosting WDDM VRAM residency priority and hiding the Ryzen 7700X iGPU).".to_string(),
        format!(
            "2. Context Length: Change from the GGUF metadata default ({} tokens) to 4096 or 8192 tokens so the KV cache does not overflow the remaining {} MiB of free physical VRAM.",
            model.declared_context_length, free_vram_mib
        ),
        if is_hybrid_ssm {
            "3. Flash Attention & KV Cache (qwen35 hybrid architecture): On the Vulkan backend at <= 3072 context, turn Flash Attention OFF (2.08x-15x faster prefill and 26.8 tok/s generation). For >= 4096 context, either switch to the ROCm backend with Flash Attention ON + K/V Cache Quantization = Q4_0 (246 tok/s prefill at 4K), or use Vulkan with Flash Attention ON + K/V Cache Quantization = Q4_0 (25.6 tok/s decode at 4K).".to_string()
        } else {
            "3. Flash Attention & KV Cache: Enable Flash Attention and set K/V Cache Quantization to Q8_0 or Q4_0 to save 50%-72% of context VRAM.".to_string()
        },
        format!(
            "4. CPU Thread Count: Set CPU Threads to {} (matching physical Zen 4 cores rather than 16 SMT threads) and pin GPU Offload strictly to GPU 0 (AMD Radeon RX 6900 XT).",
            physical_cpu_cores
        ),
    ];

    Ok(LlmOptimizationPlan {
        model: model.clone(),
        gpu_name,
        gfx_target,
        has_secondary_igpu,
        secondary_igpu_name,
        physical_cpu_cores,
        total_vram_mib,
        used_vram_mib,
        free_vram_mib,
        safety_margin_mib,
        target_context_tokens: target_ctx,
        recoverable_layers_if_apps_closed,
        warnings,
        kv_budgets,
        recommended_profiles,
        lm_studio_tips,
    })
}

/// Runs an empirical `llama-bench` autotuning sweep across RDNA2 tweak candidates on the target GGUF model.
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
    let plan = plan_llm_configuration(&meta, config.context_depth.max(2048), 512)?;
    let reps = config.reps.clamp(1, 5);
    let prompt_tokens = config.prompt_tokens.clamp(32, 2048);
    let gen_tokens = config.gen_tokens.clamp(16, 512);

    struct ProbeSpec {
        id: &'static str,
        title: &'static str,
        backend: &'static str,
        dev: &'static str,
        ngl: u32,
        fa: bool,
        ctk: &'static str,
        ctv: &'static str,
        envs: Vec<(&'static str, &'static str)>,
    }

    let mut specs = Vec::new();
    if config.vulkan_bin.is_some() {
        specs.push(ProbeSpec {
            id: "vulkan_naive_fa1",
            title: "Baseline Vulkan (-fa 1, f16 KV, default WDDM priority)",
            backend: "vulkan",
            dev: "Vulkan0",
            ngl: 99,
            fa: true,
            ctk: "f16",
            ctv: "f16",
            envs: vec![],
        });
        specs.push(ProbeSpec {
            id: "vulkan_wddm_priority_fa0",
            title: "Tuned Vulkan (GGML_VK_ENABLE_MEMORY_PRIORITY=1 + -fa 0 + iGPU isolated)",
            backend: "vulkan",
            dev: "Vulkan0",
            ngl: 99,
            fa: false,
            ctk: "f16",
            ctv: "f16",
            envs: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY", "1"),
                ("GGML_VK_VISIBLE_DEVICES", "0"),
            ],
        });
        specs.push(ProbeSpec {
            id: "vulkan_wddm_priority_q4_kv",
            title: "Tuned Vulkan Long-Context (MEMORY_PRIORITY=1 + -fa 1 + q4_0 KV cache)",
            backend: "vulkan",
            dev: "Vulkan0",
            ngl: 99,
            fa: true,
            ctk: "q4_0",
            ctv: "q4_0",
            envs: vec![
                ("GGML_VK_ENABLE_MEMORY_PRIORITY", "1"),
                ("GGML_VK_VISIBLE_DEVICES", "0"),
            ],
        });
    }

    if config.hip_bin.is_some() {
        specs.push(ProbeSpec {
            id: "rocm_isolated_q4_kv",
            title: "Tuned ROCm/HIP (HIP_VISIBLE_DEVICES=0 + -fa 1 + q4_0 KV cache)",
            backend: "hip",
            dev: "ROCm0",
            ngl: 99,
            fa: true,
            ctk: "q4_0",
            ctv: "q4_0",
            envs: vec![("HIP_VISIBLE_DEVICES", "0")],
        });
    }

    let mut candidates = Vec::with_capacity(specs.len());
    let mut baseline_pp: Option<f64> = None;
    let mut baseline_tg: Option<f64> = None;

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

        for &(k, v) in &spec.envs {
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
                        let pp = entries
                            .iter()
                            .find(|e| e.n_prompt > 0 && e.n_gen == 0)
                            .map(|e| e.median_tok_s);
                        let tg = entries
                            .iter()
                            .find(|e| e.n_gen > 0 && e.n_prompt == 0)
                            .map(|e| e.median_tok_s);

                        if baseline_pp.is_none() {
                            baseline_pp = pp;
                        }
                        if baseline_tg.is_none() {
                            baseline_tg = tg;
                        }

                        let pp_speedup = match (pp, baseline_pp) {
                            (Some(cur), Some(base)) if base > 0.0 => Some(cur / base),
                            _ => None,
                        };
                        let tg_speedup = match (tg, baseline_tg) {
                            (Some(cur), Some(base)) if base > 0.0 => Some(cur / base),
                            _ => None,
                        };

                        candidates.push(LlmAutotuneCandidateResult {
                            profile_id: spec.id.to_string(),
                            title: spec.title.to_string(),
                            backend: spec.backend.to_string(),
                            n_gpu_layers: spec.ngl,
                            flash_attn: spec.fa,
                            cache_type_k: spec.ctk.to_string(),
                            cache_type_v: spec.ctv.to_string(),
                            env_summary,
                            succeeded: true,
                            pp_tok_s: pp,
                            tg_tok_s: tg,
                            tg_speedup_vs_baseline: tg_speedup,
                            pp_speedup_vs_baseline: pp_speedup,
                            llama_server_cmd,
                            error: None,
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

    let best_decode_profile_id = candidates
        .iter()
        .filter(|c| c.succeeded && c.tg_tok_s.is_some())
        .max_by(|a, b| {
            a.tg_tok_s
                .unwrap_or(0.0)
                .total_cmp(&b.tg_tok_s.unwrap_or(0.0))
        })
        .map(|c| c.profile_id.clone());

    let best_prefill_profile_id = candidates
        .iter()
        .filter(|c| c.succeeded && c.pp_tok_s.is_some())
        .max_by(|a, b| {
            a.pp_tok_s
                .unwrap_or(0.0)
                .total_cmp(&b.pp_tok_s.unwrap_or(0.0))
        })
        .map(|c| c.profile_id.clone());

    // Geometric mean of prefill and decode (weighted 70% decode, 30% prefill)
    let best_balanced_profile_id = candidates
        .iter()
        .filter(|c| c.succeeded && c.tg_tok_s.is_some() && c.pp_tok_s.is_some())
        .max_by(|a, b| {
            let score_a = a.tg_tok_s.unwrap_or(0.0).powf(0.7) * a.pp_tok_s.unwrap_or(0.0).powf(0.3);
            let score_b = b.tg_tok_s.unwrap_or(0.0).powf(0.7) * b.pp_tok_s.unwrap_or(0.0).powf(0.3);
            score_a.total_cmp(&score_b)
        })
        .map(|c| c.profile_id.clone());

    Ok(LlmAutotuneReport {
        plan,
        prompt_tokens,
        gen_tokens,
        context_depth: config.context_depth,
        candidates,
        best_decode_profile_id,
        best_prefill_profile_id,
        best_balanced_profile_id,
        peak_rss_bytes: peak_process_rss_bytes(),
    })
}
