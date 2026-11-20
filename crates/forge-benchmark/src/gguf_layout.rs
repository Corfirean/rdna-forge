//! GGUF tensor-info parsing and weight-placement modelling.
//!
//! The GGUF header lists every tensor with its dimensions, GGML storage type and
//! data offset. This module turns that metadata into an exact per-role byte
//! accounting (per-layer, expert, embedding, output, shared) so that the memory
//! planner does not have to divide the whole file size by the block count.
//!
//! What is exact and what is an estimate:
//!
//! * **Exact (given a well-formed header):** the raw byte size of each tensor,
//!   derived from `dims` and the GGML type's block layout.
//! * **Modelled:** which tensors a llama.cpp-style `-ngl N` / `--n-cpu-moe N`
//!   run places on the GPU. This follows llama.cpp's documented layer-offload
//!   semantics (last N repeating layers; output tensors only when `N` exceeds
//!   the layer count; token embeddings stay on the CPU), but the real allocator
//!   is not observable from the header, so callers must present the result as
//!   an estimate.
//! * **Not modelled:** backend-specific sub-allocation granularity, compute
//!   buffers and driver overhead. Those are budgeted separately by the planner.

// `%` rather than `is_multiple_of` keeps the code within the workspace's declared MSRV (1.87).
#![allow(clippy::manual_is_multiple_of)]

use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Seek};

use forge_core::{ForgeError, ForgeResult};
use serde::{Deserialize, Serialize};

/// Default `general.alignment` when the key is absent (GGUF specification).
pub const DEFAULT_GGUF_ALIGNMENT: u64 = 32;

/// GGML supports at most four tensor dimensions.
const MAX_TENSOR_DIMS: u32 = 4;
/// Tensor names in real models are < 100 bytes; anything larger is corrupt.
const MAX_TENSOR_NAME_LEN: u64 = 4096;
/// Smallest possible tensor-info record: empty name (8) + n_dims (4) + one dim (8) + type (4) + offset (8).
const MIN_TENSOR_INFO_BYTES: u64 = 32;
/// Upper bound on transformer block count accepted for per-layer accounting.
pub const MAX_ACCOUNTED_BLOCKS: u32 = 16_384;

/// Returns `(name, elements_per_block, bytes_per_block)` for a GGML tensor type code.
///
/// Returns `None` for codes this build does not know (new or removed GGML types).
pub fn ggml_type_traits(code: u32) -> Option<(&'static str, u64, u64)> {
    Some(match code {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        16 => ("IQ2_XXS", 256, 66),
        17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98),
        19 => ("IQ1_S", 256, 50),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82),
        23 => ("IQ4_XS", 256, 136),
        24 => ("I8", 1, 1),
        25 => ("I16", 1, 2),
        26 => ("I32", 1, 4),
        27 => ("I64", 1, 8),
        28 => ("F64", 1, 8),
        29 => ("IQ1_M", 256, 56),
        30 => ("BF16", 1, 2),
        34 => ("TQ1_0", 256, 54),
        35 => ("TQ2_0", 256, 66),
        39 => ("MXFP4", 32, 17),
        _ => return None,
    })
}

/// Computes the raw storage size of one tensor, with checked arithmetic.
///
/// Fails on empty/oversized dimension lists, zero-sized dimensions, element-count or
/// byte-size overflow, and `dims[0]` not being a multiple of the type's block size.
pub fn tensor_size_bytes(name: &str, dims: &[u64], ggml_type: u32) -> ForgeResult<Option<u64>> {
    if dims.is_empty() || dims.len() > MAX_TENSOR_DIMS as usize {
        return Err(ForgeError::ParseError(format!(
            "tensor '{name}' has {} dimensions (expected 1..={MAX_TENSOR_DIMS})",
            dims.len()
        )));
    }
    let mut elements: u64 = 1;
    for &d in dims {
        if d == 0 {
            return Err(ForgeError::ParseError(format!(
                "tensor '{name}' has a zero-sized dimension"
            )));
        }
        elements = elements.checked_mul(d).ok_or_else(|| {
            ForgeError::ParseError(format!("tensor '{name}' element count overflows u64"))
        })?;
    }
    let Some((type_name, block_elems, block_bytes)) = ggml_type_traits(ggml_type) else {
        return Ok(None);
    };
    if dims[0] % block_elems != 0 {
        return Err(ForgeError::ParseError(format!(
            "tensor '{name}' ({type_name}) has ne[0]={} which is not a multiple of block size {block_elems}",
            dims[0]
        )));
    }
    let blocks = elements / block_elems;
    let bytes = blocks.checked_mul(block_bytes).ok_or_else(|| {
        ForgeError::ParseError(format!("tensor '{name}' byte size overflows u64"))
    })?;
    Ok(Some(bytes))
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment > 0);
    let rem = value % alignment;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - rem)
    }
}

/// Role a tensor plays when layers are distributed across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// `blk.N.*`; `expert` is true for routed-expert tensors (`*_exps*`).
    Layer { index: u32, expert: bool },
    /// Token / input embeddings (`token_embd*`, `per_layer_token_embd*`).
    Embedding,
    /// Output head and final norm (`output*`).
    Output,
    /// Everything else (e.g. `rope_freqs.weight`).
    Shared,
}

/// Classifies a GGUF tensor by name using the llama.cpp naming convention.
pub fn classify_tensor(name: &str) -> TensorRole {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((idx, tail)) = rest.split_once('.') {
            if let Ok(index) = idx.parse::<u32>() {
                return TensorRole::Layer {
                    index,
                    expert: tail.contains("_exps"),
                };
            }
        }
        return TensorRole::Shared;
    }
    if name.starts_with("token_embd") || name.starts_with("per_layer_token_embd") {
        TensorRole::Embedding
    } else if name.starts_with("output") {
        TensorRole::Output
    } else {
        TensorRole::Shared
    }
}

/// One parsed tensor-info record with its computed size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRecord {
    pub name: String,
    pub ggml_type: u32,
    pub offset: u64,
    /// Raw storage size. For unknown GGML types this is the offset gap to the next tensor.
    pub size_bytes: u64,
    pub size_is_exact: bool,
    pub role: TensorRole,
}

/// Exact per-role tensor accounting for one GGUF file.
///
/// All byte counts include per-tensor rounding up to [`Self::alignment`], which
/// approximates sub-allocation granularity; `total_tensor_bytes` is the raw sum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufTensorLayout {
    pub tensors_parsed: u64,
    pub alignment: u64,
    pub data_offset: u64,
    /// Sum of raw tensor sizes (no alignment padding).
    pub total_tensor_bytes: u64,
    /// Sum of tensor sizes with each tensor rounded up to `alignment`.
    pub total_tensor_bytes_aligned: u64,
    /// Aligned bytes per transformer block index (`blk.N.*`), length = block count.
    pub layer_bytes: Vec<u64>,
    /// Portion of `layer_bytes[i]` that is routed-expert FFN weights.
    pub layer_expert_bytes: Vec<u64>,
    pub embedding_bytes: u64,
    pub output_bytes: u64,
    /// Tensors that are neither per-layer, embedding nor output.
    pub shared_bytes: u64,
    /// `blk.N.*` tensors whose index is >= the declared block count.
    pub extra_layer_bytes: u64,
    /// True when no `output.weight` exists, i.e. the output head reuses `token_embd.weight`.
    pub tied_output_embedding: bool,
    /// Tensors whose GGML type is unknown to this build (size taken from offset gaps).
    pub unknown_type_tensors: u64,
    /// Raw bytes per GGML type name (shows mixed quantization). Deterministic order.
    pub bytes_by_type: BTreeMap<String, u64>,
}

impl GgufTensorLayout {
    /// Folds parsed tensor records into a layout for a model with `block_count` blocks.
    pub fn from_records(
        records: &[TensorRecord],
        block_count: u32,
        alignment: u64,
        data_offset: u64,
    ) -> ForgeResult<Self> {
        if block_count == 0 || block_count > MAX_ACCOUNTED_BLOCKS {
            return Err(ForgeError::ParseError(format!(
                "block_count {block_count} is outside the supported accounting range 1..={MAX_ACCOUNTED_BLOCKS}"
            )));
        }
        let overflow =
            || ForgeError::ParseError("GGUF tensor byte accounting overflows u64".to_string());
        let n = block_count as usize;
        let mut layout = Self {
            tensors_parsed: records.len() as u64,
            alignment,
            data_offset,
            total_tensor_bytes: 0,
            total_tensor_bytes_aligned: 0,
            layer_bytes: vec![0; n],
            layer_expert_bytes: vec![0; n],
            embedding_bytes: 0,
            output_bytes: 0,
            shared_bytes: 0,
            extra_layer_bytes: 0,
            tied_output_embedding: true,
            unknown_type_tensors: 0,
            bytes_by_type: BTreeMap::new(),
        };

        for rec in records {
            let padded = align_up(rec.size_bytes, alignment).ok_or_else(overflow)?;
            layout.total_tensor_bytes = layout
                .total_tensor_bytes
                .checked_add(rec.size_bytes)
                .ok_or_else(overflow)?;
            layout.total_tensor_bytes_aligned = layout
                .total_tensor_bytes_aligned
                .checked_add(padded)
                .ok_or_else(overflow)?;

            let type_label = ggml_type_traits(rec.ggml_type)
                .map(|(name, _, _)| name.to_string())
                .unwrap_or_else(|| format!("UNKNOWN_{}", rec.ggml_type));
            if !rec.size_is_exact {
                layout.unknown_type_tensors += 1;
            }
            let entry = layout.bytes_by_type.entry(type_label).or_insert(0);
            *entry = entry.checked_add(rec.size_bytes).ok_or_else(overflow)?;

            match rec.role {
                TensorRole::Layer { index, expert } => {
                    if (index as usize) < n {
                        let i = index as usize;
                        layout.layer_bytes[i] = layout.layer_bytes[i]
                            .checked_add(padded)
                            .ok_or_else(overflow)?;
                        if expert {
                            layout.layer_expert_bytes[i] = layout.layer_expert_bytes[i]
                                .checked_add(padded)
                                .ok_or_else(overflow)?;
                        }
                    } else {
                        layout.extra_layer_bytes = layout
                            .extra_layer_bytes
                            .checked_add(padded)
                            .ok_or_else(overflow)?;
                    }
                }
                TensorRole::Embedding => {
                    layout.embedding_bytes = layout
                        .embedding_bytes
                        .checked_add(padded)
                        .ok_or_else(overflow)?;
                }
                TensorRole::Output => {
                    if rec.name == "output.weight" {
                        layout.tied_output_embedding = false;
                    }
                    layout.output_bytes = layout
                        .output_bytes
                        .checked_add(padded)
                        .ok_or_else(overflow)?;
                }
                TensorRole::Shared => {
                    layout.shared_bytes = layout
                        .shared_bytes
                        .checked_add(padded)
                        .ok_or_else(overflow)?;
                }
            }
        }
        Ok(layout)
    }

    /// Number of transformer blocks accounted for.
    pub fn block_count(&self) -> u32 {
        self.layer_bytes.len() as u32
    }

    /// Sum of all per-layer bytes.
    pub fn repeating_layer_bytes(&self) -> u64 {
        self.layer_bytes
            .iter()
            .fold(0u64, |acc, &b| acc.saturating_add(b))
    }

    /// Mean size of a repeating layer. This is a mean over uneven layers, not a per-layer truth.
    pub fn mean_layer_bytes(&self) -> u64 {
        match self.layer_bytes.len() as u64 {
            0 => 0,
            n => self.repeating_layer_bytes() / n,
        }
    }

    /// True when any layer carries routed-expert weights.
    pub fn has_experts(&self) -> bool {
        self.layer_expert_bytes.iter().any(|&b| b > 0)
    }

    /// Bytes of the output head that move to the GPU. A tied head is duplicated there.
    fn output_head_gpu_bytes(&self) -> u64 {
        if self.tied_output_embedding {
            self.output_bytes.saturating_add(self.embedding_bytes)
        } else {
            self.output_bytes
        }
    }

    /// Estimated GPU-resident weight bytes for llama.cpp `-ngl ngl --n-cpu-moe n_cpu_moe`.
    ///
    /// Model: the *last* `ngl` repeating layers are offloaded; the output head (plus shared
    /// tensors) only when `ngl` exceeds the layer count; token embeddings stay on the CPU.
    /// `n_cpu_moe` keeps routed-expert tensors of the *first* `n_cpu_moe` layers in system RAM.
    /// Saturating arithmetic is used so an adversarial header can only make the estimate larger.
    pub fn gpu_weight_bytes(&self, ngl: u32, n_cpu_moe: u32) -> u64 {
        if ngl == 0 {
            return 0;
        }
        let n = self.layer_bytes.len();
        let on_gpu = (ngl as usize).min(n);
        let first_gpu = n - on_gpu;
        let mut total = self.shared_bytes;
        for i in first_gpu..n {
            let mut b = self.layer_bytes[i];
            if (i as u32) < n_cpu_moe {
                b = b.saturating_sub(self.layer_expert_bytes[i]);
            }
            total = total.saturating_add(b);
        }
        if ngl as usize > n {
            total = total
                .saturating_add(self.output_head_gpu_bytes())
                .saturating_add(self.extra_layer_bytes);
        }
        total
    }

    /// Estimated GPU weight bytes with everything offloaded and no CPU experts.
    pub fn full_offload_bytes(&self) -> u64 {
        self.gpu_weight_bytes(u32::MAX, 0)
    }

    /// Largest `-ngl` (capped at the block count) whose weights fit in `budget_bytes`.
    pub fn max_ngl_within(&self, budget_bytes: u64) -> u32 {
        let n = self.block_count();
        let mut best = 0;
        for ngl in 1..=n {
            if self.gpu_weight_bytes(ngl, 0) <= budget_bytes {
                best = ngl;
            } else {
                break;
            }
        }
        best
    }

    /// Smallest `--n-cpu-moe` such that fully offloaded weights fit in `budget_bytes`.
    /// Returns the block count when even moving every layer's experts to the CPU is not enough.
    pub fn min_cpu_moe_within(&self, budget_bytes: u64) -> u32 {
        let n = self.block_count();
        (0..=n)
            .find(|&k| self.gpu_weight_bytes(u32::MAX, k) <= budget_bytes)
            .unwrap_or(n)
    }
}

fn read_exact_array<R: Read, const N: usize>(r: &mut R) -> ForgeResult<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b).map_err(ForgeError::Io)?;
    Ok(b)
}

fn read_u32<R: Read>(r: &mut R) -> ForgeResult<u32> {
    Ok(u32::from_le_bytes(read_exact_array::<R, 4>(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> ForgeResult<u64> {
    Ok(u64::from_le_bytes(read_exact_array::<R, 8>(r)?))
}

fn read_tensor_name<R: Read>(r: &mut R) -> ForgeResult<String> {
    let len = read_u64(r)?;
    if len > MAX_TENSOR_NAME_LEN {
        return Err(ForgeError::ParseError(format!(
            "GGUF tensor name length {len} exceeds {MAX_TENSOR_NAME_LEN} byte limit"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).map_err(ForgeError::Io)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Parses the tensor-info section that follows the metadata key/values.
///
/// `file_size_bytes` bounds all untrusted counts and offsets. On success the reader is
/// positioned at the end of the tensor-info section; the returned `data_offset` is that
/// position rounded up to `alignment`.
pub fn read_tensor_records<R: Read + Seek>(
    r: &mut R,
    tensor_count: u64,
    file_size_bytes: u64,
    alignment: u64,
) -> ForgeResult<(Vec<TensorRecord>, u64)> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(ForgeError::ParseError(format!(
            "general.alignment {alignment} must be a non-zero power of two"
        )));
    }
    if tensor_count.saturating_mul(MIN_TENSOR_INFO_BYTES) > file_size_bytes {
        return Err(ForgeError::ParseError(format!(
            "GGUF declares {tensor_count} tensors, which cannot fit in a {file_size_bytes}-byte file"
        )));
    }

    // `tensor_count` is bounded by file size / 32 above, so this allocation is bounded too.
    let mut records: Vec<TensorRecord> = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
    let mut seen: HashSet<String> = HashSet::new();
    for _ in 0..tensor_count {
        let name = read_tensor_name(r)?;
        let n_dims = read_u32(r)?;
        if n_dims == 0 || n_dims > MAX_TENSOR_DIMS {
            return Err(ForgeError::ParseError(format!(
                "tensor '{name}' declares {n_dims} dimensions (expected 1..={MAX_TENSOR_DIMS})"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }
        let ggml_type = read_u32(r)?;
        let offset = read_u64(r)?;
        if !seen.insert(name.clone()) {
            return Err(ForgeError::ParseError(format!(
                "duplicate tensor name '{name}'"
            )));
        }
        if offset % alignment != 0 {
            return Err(ForgeError::ParseError(format!(
                "tensor '{name}' offset {offset} is not a multiple of alignment {alignment}"
            )));
        }
        let size = tensor_size_bytes(&name, &dims, ggml_type)?;
        records.push(TensorRecord {
            role: classify_tensor(&name),
            name,
            ggml_type,
            offset,
            size_bytes: size.unwrap_or(0),
            size_is_exact: size.is_some(),
        });
    }

    let info_end = r.stream_position().map_err(ForgeError::Io)?;
    let data_offset = align_up(info_end, alignment)
        .ok_or_else(|| ForgeError::ParseError("GGUF data offset overflows u64".to_string()))?;
    let data_size = file_size_bytes.checked_sub(data_offset).ok_or_else(|| {
        ForgeError::ParseError(format!(
            "GGUF tensor data section starts at {data_offset}, beyond file size {file_size_bytes}"
        ))
    })?;

    // Unknown GGML types: derive an upper-bound size from the gap to the next tensor offset.
    let mut offsets: Vec<u64> = records.iter().map(|rec| rec.offset).collect();
    offsets.sort_unstable();
    for rec in records.iter_mut().filter(|rec| !rec.size_is_exact) {
        let next = offsets
            .iter()
            .copied()
            .find(|&o| o > rec.offset)
            .unwrap_or(data_size);
        rec.size_bytes = next.saturating_sub(rec.offset);
    }

    for rec in &records {
        let end = rec.offset.checked_add(rec.size_bytes).ok_or_else(|| {
            ForgeError::ParseError(format!("tensor '{}' end offset overflows", rec.name))
        })?;
        if end > data_size {
            return Err(ForgeError::ParseError(format!(
                "tensor '{}' ends at data offset {end} but the data section is only {data_size} bytes (truncated file?)",
                rec.name
            )));
        }
    }
    Ok((records, data_offset))
}

/// Highest `blk.N` index + 1 among the records, or `None` if there are no layer tensors.
pub fn inferred_block_count(records: &[TensorRecord]) -> Option<u32> {
    records
        .iter()
        .filter_map(|r| match r.role {
            TensorRole::Layer { index, .. } => index.checked_add(1),
            _ => None,
        })
        .max()
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Synthetic GGUF builder. Produces tiny, structurally valid files; no real weights.

    use super::*;

    pub enum Kv {
        Str(&'static str, String),
        U32(&'static str, u32),
    }

    pub struct Tensor {
        pub name: String,
        pub dims: Vec<u64>,
        pub ggml_type: u32,
        /// Overrides the computed size (needed for unknown GGML types).
        pub size_override: Option<u64>,
    }

    pub fn t(name: &str, dims: &[u64], ggml_type: u32) -> Tensor {
        Tensor {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            size_override: None,
        }
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    pub fn build(kvs: &[Kv], tensors: &[Tensor], alignment: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in kvs {
            match kv {
                Kv::Str(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut out, v);
                }
                Kv::U32(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let mut offset = 0u64;
        let mut sizes = Vec::new();
        for ten in tensors {
            let size = ten.size_override.unwrap_or_else(|| {
                tensor_size_bytes(&ten.name, &ten.dims, ten.ggml_type)
                    .unwrap()
                    .unwrap()
            });
            put_str(&mut out, &ten.name);
            out.extend_from_slice(&(ten.dims.len() as u32).to_le_bytes());
            for d in &ten.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ten.ggml_type.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            sizes.push(size);
            offset += align_up(size, alignment).unwrap();
        }
        let padded = align_up(out.len() as u64, alignment).unwrap() as usize;
        out.resize(padded, 0);
        out.resize(out.len() + offset as usize, 0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::llm_optimizer::inspect_gguf_reader;
    use std::io::Cursor;
    use std::path::Path;

    fn inspect(bytes: Vec<u8>) -> ForgeResult<crate::llm_optimizer::GgufModelMetadata> {
        let len = bytes.len() as u64;
        inspect_gguf_reader(Cursor::new(bytes), len, Path::new("synthetic-model.gguf"))
    }

    fn arch_kvs(arch: &'static str, blocks: u32) -> Vec<Kv> {
        // Leaked &'static keys keep the fixture builder simple in tests.
        let key = |s: &str| -> &'static str { Box::leak(format!("{arch}.{s}").into_boxed_str()) };
        vec![
            Kv::Str("general.architecture", arch.to_string()),
            Kv::U32(key("block_count"), blocks),
            Kv::U32(key("embedding_length"), 256),
            Kv::U32(key("attention.head_count"), 4),
            Kv::U32(key("attention.head_count_kv"), 2),
        ]
    }

    fn dense_layer(i: u32, ty: u32) -> Vec<Tensor> {
        vec![
            t(&format!("blk.{i}.attn_q.weight"), &[256, 256], ty),
            t(&format!("blk.{i}.ffn_down.weight"), &[512, 256], ty),
            t(&format!("blk.{i}.attn_norm.weight"), &[256], 0),
        ]
    }

    #[test]
    fn llama_family_layout_separates_embedding_output_and_layers() {
        let mut tensors = vec![t("token_embd.weight", &[256, 1024], 12)];
        for i in 0..4 {
            tensors.extend(dense_layer(i, 12));
        }
        tensors.push(t("output_norm.weight", &[256], 0));
        tensors.push(t("output.weight", &[256, 1024], 14));
        tensors.push(t("rope_freqs.weight", &[32], 0));
        let meta = inspect(build(&arch_kvs("llama", 4), &tensors, 32)).unwrap();
        let l = meta.tensor_layout.as_ref().expect("layout present");

        assert_eq!(l.block_count(), 4);
        assert!(!l.tied_output_embedding);
        assert_eq!(l.embedding_bytes, align_up(1024 * 144, 32).unwrap());
        assert_eq!(l.output_bytes, 1024 + (1024 * 256 / 256) * 210);
        assert!(l.shared_bytes > 0);
        assert!(l.layer_bytes.iter().all(|&b| b == l.layer_bytes[0]));
        assert_eq!(l.bytes_by_type.len(), 3); // F32, Q4_K, Q6_K
        assert_eq!(meta.tensor_layout_basis(), "tensor_metadata");

        // Old approximation divided the whole file by block count: it must differ materially
        // from the real per-layer size because embeddings/output/header are not layers.
        let naive = meta.file_size_bytes / 4;
        assert!(naive > l.mean_layer_bytes());
        assert_eq!(meta.bytes_per_layer, l.mean_layer_bytes());
    }

    #[test]
    fn tied_embeddings_are_duplicated_on_the_gpu_output_head() {
        let mut tensors = vec![t("token_embd.weight", &[256, 512], 8)];
        for i in 0..2 {
            tensors.extend(dense_layer(i, 8));
        }
        tensors.push(t("output_norm.weight", &[256], 0));
        let meta = inspect(build(&arch_kvs("qwen2", 2), &tensors, 32)).unwrap();
        let l = meta.tensor_layout.as_ref().unwrap();
        assert!(l.tied_output_embedding);
        let layers: u64 = l.layer_bytes.iter().sum();
        // ngl == n_layer keeps the head and embeddings off the GPU.
        assert_eq!(l.gpu_weight_bytes(2, 0), layers + l.shared_bytes);
        // ngl > n_layer adds the output norm and a duplicate of the tied embedding.
        assert_eq!(
            l.gpu_weight_bytes(3, 0),
            layers + l.shared_bytes + l.output_bytes + l.embedding_bytes
        );
    }

    #[test]
    fn mixed_quantization_and_uneven_layers_use_the_last_n_layers() {
        let mut tensors = vec![t("token_embd.weight", &[256, 256], 12)];
        // Layers 0..2 are Q4_K, layer 3 is Q8_0 (much larger): uneven layer sizes.
        for i in 0..3 {
            tensors.extend(dense_layer(i, 12));
        }
        tensors.extend(dense_layer(3, 8));
        tensors.push(t("output.weight", &[256, 256], 14));
        let meta = inspect(build(&arch_kvs("mistral", 4), &tensors, 32)).unwrap();
        let l = meta.tensor_layout.as_ref().unwrap();
        assert!(l.bytes_by_type.contains_key("Q4_K"));
        assert!(l.bytes_by_type.contains_key("Q8_0"));
        assert!(l.layer_bytes[3] > l.layer_bytes[0]);
        assert_eq!(l.gpu_weight_bytes(1, 0), l.layer_bytes[3] + l.shared_bytes);
        assert_eq!(
            l.gpu_weight_bytes(2, 0),
            l.layer_bytes[3] + l.layer_bytes[2] + l.shared_bytes
        );
        // Budget search follows the real sizes, not a uniform average.
        let budget = l.layer_bytes[3] + l.layer_bytes[2];
        assert_eq!(l.max_ngl_within(budget + l.shared_bytes), 2);
        assert_eq!(l.max_ngl_within(l.layer_bytes[3] - 1), 0);
        let uniform = l.repeating_layer_bytes() / 4;
        assert_ne!(uniform, l.layer_bytes[3]);
    }

    #[test]
    fn moe_expert_bytes_drive_cpu_moe_planning() {
        let mut tensors = vec![t("token_embd.weight", &[256, 256], 12)];
        for i in 0..4u32 {
            tensors.push(t(&format!("blk.{i}.attn_q.weight"), &[256, 256], 12));
            tensors.push(t(&format!("blk.{i}.ffn_gate_inp.weight"), &[256, 8], 0));
            tensors.push(t(
                &format!("blk.{i}.ffn_up_exps.weight"),
                &[256, 512, 8],
                12,
            ));
            tensors.push(t(
                &format!("blk.{i}.ffn_down_exps.weight"),
                &[512, 256, 8],
                12,
            ));
        }
        tensors.push(t("output.weight", &[256, 256], 12));
        let mut kvs = arch_kvs("qwen3moe", 4);
        kvs.push(Kv::U32("qwen3moe.expert_count", 8));
        kvs.push(Kv::U32("qwen3moe.expert_used_count", 2));
        let meta = inspect(build(&kvs, &tensors, 32)).unwrap();
        let l = meta.tensor_layout.as_ref().unwrap();
        assert!(meta.is_moe && l.has_experts());
        for i in 0..4 {
            assert!(l.layer_expert_bytes[i] > l.layer_bytes[i] / 2);
        }
        let full = l.full_offload_bytes();
        assert_eq!(l.min_cpu_moe_within(full), 0);
        // Moving experts of the first two layers to CPU frees exactly their expert bytes.
        let freed = l.layer_expert_bytes[0] + l.layer_expert_bytes[1];
        assert_eq!(l.gpu_weight_bytes(u32::MAX, 2), full - freed);
        assert_eq!(l.min_cpu_moe_within(full - freed), 2);
        // Nothing fits: report all layers rather than inventing a smaller number.
        assert_eq!(l.min_cpu_moe_within(0), 4);
    }

    #[test]
    fn missing_block_count_is_inferred_from_tensors_and_reported() {
        let mut tensors = vec![t("token_embd.weight", &[256, 256], 12)];
        for i in 0..3 {
            tensors.extend(dense_layer(i, 12));
        }
        let kvs = vec![Kv::Str("general.architecture", "llama".to_string())];
        let meta = inspect(build(&kvs, &tensors, 32)).unwrap();
        assert_eq!(meta.block_count, 3);
        assert!(meta
            .assumed_defaults
            .iter()
            .any(|d| d.contains("block_count") && d.contains("tensor")));
        assert!(meta
            .assumed_defaults
            .iter()
            .any(|d| d.contains("embedding_length")));
    }

    #[test]
    fn model_without_tensors_falls_back_with_explicit_basis() {
        let meta = inspect(build(&arch_kvs("llama", 8), &[], 32)).unwrap();
        assert!(meta.tensor_layout.is_none());
        assert_eq!(meta.tensor_layout_basis(), "file_size_approximation");
    }

    #[test]
    fn unknown_ggml_type_uses_offset_gap_and_is_flagged() {
        let tensors = vec![
            t("token_embd.weight", &[256, 256], 12),
            Tensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![64, 64],
                ggml_type: 9999,
                size_override: Some(4096),
            },
            t("blk.0.attn_norm.weight", &[256], 0),
        ];
        let meta = inspect(build(&arch_kvs("llama", 1), &tensors, 32)).unwrap();
        let l = meta.tensor_layout.as_ref().unwrap();
        assert_eq!(l.unknown_type_tensors, 1);
        assert!(l.layer_bytes[0] >= 4096);
        assert!(meta
            .tensor_layout_notes()
            .iter()
            .any(|n| n.contains("unknown GGML")));
    }

    #[test]
    fn split_gguf_is_marked_as_partial() {
        let mut kvs = arch_kvs("llama", 2);
        kvs.push(Kv::U32("split.count", 3));
        kvs.push(Kv::U32("split.no", 0));
        let mut tensors = vec![t("token_embd.weight", &[256, 256], 12)];
        tensors.extend(dense_layer(0, 12));
        let meta = inspect(build(&kvs, &tensors, 32)).unwrap();
        assert_eq!(meta.split_count, Some(3));
        assert!(meta
            .tensor_layout_notes()
            .iter()
            .any(|n| n.contains("split")));
    }

    #[test]
    fn invalid_tensor_metadata_is_rejected() {
        // ne[0] not a multiple of the Q4_K block size.
        let err = tensor_size_bytes("w", &[100, 4], 12).unwrap_err();
        assert!(err.to_string().contains("multiple of block size"));
        // Zero dimension, too many dimensions, empty dimensions.
        assert!(tensor_size_bytes("w", &[0, 4], 0).is_err());
        assert!(tensor_size_bytes("w", &[1, 1, 1, 1, 1], 0).is_err());
        assert!(tensor_size_bytes("w", &[], 0).is_err());

        // Truncated file: declared data larger than what is present.
        let tensors = vec![t("token_embd.weight", &[256, 256], 12)];
        let mut bytes = build(&arch_kvs("llama", 1), &tensors, 32);
        bytes.truncate(bytes.len() - 1000);
        let err = inspect(bytes).unwrap_err();
        assert!(err.to_string().contains("truncated"));

        // Duplicate tensor names.
        let dup = vec![t("a.weight", &[32], 0), t("a.weight", &[32], 0)];
        let err = inspect(build(&arch_kvs("llama", 1), &dup, 32)).unwrap_err();
        assert!(err.to_string().contains("duplicate"));

        // Declared tensor count that cannot fit in the file.
        let mut bytes = build(&arch_kvs("llama", 1), &[], 32);
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let err = inspect(bytes).unwrap_err();
        assert!(err.to_string().contains("cannot fit"));
    }

    #[test]
    fn size_arithmetic_is_checked_at_u64_boundaries() {
        // Element count overflow.
        let err = tensor_size_bytes("w", &[1 << 32, 1 << 32], 0).unwrap_err();
        assert!(err.to_string().contains("element count overflows"));
        // Byte size overflow: elements fit u64 but elements * 4 does not.
        let err = tensor_size_bytes("w", &[1 << 62, 2], 0).unwrap_err();
        assert!(err.to_string().contains("byte size overflows"));
        // Largest representable tensor is accepted exactly.
        let ok = tensor_size_bytes("w", &[u64::MAX / 8], 28).unwrap();
        assert_eq!(ok, Some((u64::MAX / 8) * 8));
        // Layout accounting overflow is an error, not a wrap.
        let big = TensorRecord {
            name: "blk.0.w".into(),
            ggml_type: 0,
            offset: 0,
            size_bytes: u64::MAX - 8,
            size_is_exact: true,
            role: TensorRole::Layer {
                index: 0,
                expert: false,
            },
        };
        let recs = vec![big.clone(), TensorRecord { offset: 64, ..big }];
        assert!(GgufTensorLayout::from_records(&recs, 1, 32, 0).is_err());
        // Absurd block counts are refused instead of allocating gigabytes.
        assert!(GgufTensorLayout::from_records(&[], u32::MAX, 32, 0).is_err());
        // Placement arithmetic saturates rather than wrapping.
        let layout = GgufTensorLayout {
            tensors_parsed: 2,
            alignment: 32,
            data_offset: 0,
            total_tensor_bytes: 0,
            total_tensor_bytes_aligned: 0,
            layer_bytes: vec![u64::MAX, u64::MAX],
            layer_expert_bytes: vec![0, 0],
            embedding_bytes: 0,
            output_bytes: u64::MAX,
            shared_bytes: u64::MAX,
            extra_layer_bytes: 0,
            tied_output_embedding: false,
            unknown_type_tensors: 0,
            bytes_by_type: BTreeMap::new(),
        };
        assert_eq!(layout.full_offload_bytes(), u64::MAX);
    }

    #[test]
    fn large_tensor_counts_parse_and_are_deterministic() {
        let mut tensors = vec![t("token_embd.weight", &[256, 256], 12)];
        for i in 0..8u32 {
            for k in 0..2500u32 {
                tensors.push(t(&format!("blk.{i}.tiny_{k}.weight"), &[1], 0));
            }
        }
        let bytes = build(&arch_kvs("llama", 8), &tensors, 32);
        let a = inspect(bytes.clone()).unwrap();
        let b = inspect(bytes).unwrap();
        assert_eq!(a.tensor_count, 20_001);
        assert_eq!(a.tensor_layout, b.tensor_layout);
        let l = a.tensor_layout.as_ref().unwrap();
        // Each 4-byte tensor is padded to the 32-byte GGUF alignment.
        assert_eq!(l.layer_bytes[0], 2500 * 32);
        assert_eq!(l.total_tensor_bytes % 4, 0);
    }

    #[test]
    fn planner_uses_tensor_metadata_and_labels_the_fallback() {
        use crate::llm_optimizer::{plan_llm_configuration_for_gpu, LlmTargetGpuContext};
        use forge_core::GpuArchitecture;

        // Large embedding + output, small repeating layers: the uniform file/blocks split
        // overstates per-layer cost and therefore understates how many layers fit.
        let mut tensors = vec![t("token_embd.weight", &[256, 32768], 12)];
        for i in 0..8 {
            tensors.extend(dense_layer(i, 12));
        }
        tensors.push(t("output.weight", &[256, 32768], 12));
        let meta = inspect(build(&arch_kvs("llama", 8), &tensors, 32)).unwrap();
        let layout = meta.tensor_layout.as_ref().unwrap();
        let ctx = LlmTargetGpuContext {
            gpu_index: 0,
            gpu_name: "Synthetic".into(),
            gfx_target: "gfx1030".into(),
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
            matched_profile_id: None,
            matched_profile_name: None,
            vk_visible_device: 0,
            hip_visible_device: 0,
            disabled_gpu_indices: vec![],
            secondary_igpu_name: None,
            physical_cpu_cores: 8,
            total_ram_mib: 32_768,
            total_vram_mib: 4_096,
            used_vram_mib: 0,
        };
        let plan = plan_llm_configuration_for_gpu(&meta, &ctx, 512, 0);
        let b = &plan.kv_budgets[0];
        assert_eq!(b.weights_estimate_basis, "tensor_metadata");
        assert_eq!(b.model_file_mib, meta.file_size_bytes.div_ceil(1024 * 1024));
        // GPU weight estimate excludes the CPU-resident token embedding.
        assert_eq!(
            b.model_weights_mib,
            layout.full_offload_bytes().div_ceil(1024 * 1024)
        );
        assert!(layout.full_offload_bytes() < meta.file_size_bytes);

        // Same model with its tensor table stripped falls back and says so.
        let mut stripped = meta.clone();
        stripped.tensor_layout = None;
        let fallback = plan_llm_configuration_for_gpu(&stripped, &ctx, 512, 0);
        assert_eq!(
            fallback.kv_budgets[0].weights_estimate_basis,
            "file_size_approximation"
        );
        assert_eq!(
            fallback.kv_budgets[0].model_weights_mib,
            stripped.file_size_bytes.div_ceil(1024 * 1024)
        );
        assert!(fallback
            .warnings
            .iter()
            .any(|w| w.contains("approximated as file size")));
    }

    #[test]
    fn classification_follows_llama_cpp_names() {
        assert_eq!(
            classify_tensor("blk.12.ffn_down_exps.weight"),
            TensorRole::Layer {
                index: 12,
                expert: true
            }
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_down_shexp.weight"),
            TensorRole::Layer {
                index: 3,
                expert: false
            }
        );
        assert_eq!(classify_tensor("token_embd.weight"), TensorRole::Embedding);
        assert_eq!(classify_tensor("output_norm.weight"), TensorRole::Output);
        assert_eq!(classify_tensor("rope_freqs.weight"), TensorRole::Shared);
        assert_eq!(classify_tensor("blk.x.weight"), TensorRole::Shared);
    }
}
