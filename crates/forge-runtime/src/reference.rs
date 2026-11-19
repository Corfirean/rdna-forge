use serde::{Deserialize, Serialize};

use forge_core::{ForgeError, ForgeResult};

/// Summary of numerical comparison between GPU output and a reference implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalAccuracy {
    pub element_count: usize,
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub rmse: f64,
    pub all_finite: bool,
    pub passed: bool,
}

/// CPU FP64-accumulated reference matrix multiplication: `C[m, n] = A[m, k] * B[k, n]`.
pub fn ref_gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> ForgeResult<Vec<f32>> {
    if m == 0 || k == 0 || n == 0 {
        return Err(ForgeError::InvalidDimension(
            "GEMM dimensions m, k, n must all be > 0".to_string(),
        ));
    }
    let expected_a = m
        .checked_mul(k)
        .ok_or_else(|| ForgeError::InvalidDimension("m * k overflow".to_string()))?;
    let expected_b = k
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("k * n overflow".to_string()))?;
    let expected_c = m
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("m * n overflow".to_string()))?;

    if a.len() != expected_a || b.len() != expected_b {
        return Err(ForgeError::InvalidDimension(format!(
            "GEMM buffer length mismatch: got a={}, b={}, expected a={expected_a}, b={expected_b}",
            a.len(),
            b.len()
        )));
    }

    let mut out = vec![0.0f32; expected_c];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let out_row = &mut out[row * n..(row + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let av64 = f64::from(av);
            let b_row = &b[kk * n..(kk + 1) * n];
            for col in 0..n {
                out_row[col] = (f64::from(out_row[col]) + av64 * f64::from(b_row[col])) as f32;
            }
        }
    }
    Ok(out)
}

/// Higher-precision CPU reference GEMM accumulating full row dot products in `f64`.
pub fn ref_gemm_f32_exact_f64(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> ForgeResult<Vec<f32>> {
    if m == 0 || k == 0 || n == 0 {
        return Err(ForgeError::InvalidDimension(
            "GEMM dimensions m, k, n must all be > 0".to_string(),
        ));
    }
    let expected_a = m
        .checked_mul(k)
        .ok_or_else(|| ForgeError::InvalidDimension("m * k overflow".to_string()))?;
    let expected_b = k
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("k * n overflow".to_string()))?;
    let expected_c = m
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("m * n overflow".to_string()))?;
    if a.len() != expected_a || b.len() != expected_b {
        return Err(ForgeError::InvalidDimension(format!(
            "GEMM buffer length mismatch: got a={}, b={}, expected a={expected_a}, b={expected_b}",
            a.len(),
            b.len()
        )));
    }

    let mut acc = vec![0.0f64; expected_c];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let acc_row = &mut acc[row * n..(row + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let av64 = f64::from(av);
            let b_row = &b[kk * n..(kk + 1) * n];
            for col in 0..n {
                acc_row[col] += av64 * f64::from(b_row[col]);
            }
        }
    }
    Ok(acc.into_iter().map(|v| v as f32).collect())
}

/// CPU FP64-accumulated reference RMSNorm over `x[rows, cols]` with `weight[cols]`.
pub fn ref_rmsnorm_f32(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> ForgeResult<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Err(ForgeError::InvalidDimension(
            "RMSNorm dimensions rows and cols must be > 0".to_string(),
        ));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| ForgeError::InvalidDimension("rows * cols overflow".to_string()))?;
    if x.len() != total || weight.len() != cols {
        return Err(ForgeError::InvalidDimension(format!(
            "RMSNorm buffer length mismatch: got x={}, w={}, expected x={total}, w={cols}",
            x.len(),
            weight.len()
        )));
    }

    let mut out = vec![0.0f32; total];
    for r in 0..rows {
        let row_slice = &x[r * cols..(r + 1) * cols];
        let sq_sum: f64 = row_slice
            .iter()
            .map(|&v| {
                let d = f64::from(v);
                d * d
            })
            .sum();
        let inv_rms = 1.0 / ((sq_sum / cols as f64) + f64::from(eps)).sqrt();
        let out_slice = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out_slice[c] = (f64::from(row_slice[c]) * inv_rms * f64::from(weight[c])) as f32;
        }
    }
    Ok(out)
}

/// CPU FP64-trig reference RoPE rotating interleaved pairs `(2i, 2i+1)` for `[seq_len, n_heads, head_dim]`.
pub fn ref_rope_f32(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    theta_base: f32,
) -> ForgeResult<Vec<f32>> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(ForgeError::InvalidDimension(format!(
            "RoPE requires non-zero dimensions and even head_dim, got ({seq_len}, {n_heads}, {head_dim})"
        )));
    }
    if theta_base <= 0.0 || !theta_base.is_finite() {
        return Err(ForgeError::InvalidDimension(format!(
            "RoPE theta_base must be positive and finite, got {theta_base}"
        )));
    }

    let total = seq_len
        .checked_mul(n_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| ForgeError::InvalidDimension("RoPE tensor size overflow".to_string()))?;
    if input.len() != total {
        return Err(ForgeError::InvalidDimension(format!(
            "RoPE input length mismatch: got {}, expected {total}",
            input.len()
        )));
    }

    let half_dim = head_dim / 2;
    let mut out = vec![0.0f32; total];
    for pos in 0..seq_len {
        for head in 0..n_heads {
            let head_offset = (pos * n_heads + head) * head_dim;
            for i in 0..half_dim {
                let exponent = -((2 * i) as f64) / (head_dim as f64);
                let freq = f64::from(theta_base).powf(exponent);
                let angle = (pos as f64) * freq;
                let c = angle.cos();
                let s = angle.sin();
                let idx = head_offset + 2 * i;
                let x0 = f64::from(input[idx]);
                let x1 = f64::from(input[idx + 1]);
                out[idx] = (x0 * c - x1 * s) as f32;
                out[idx + 1] = (x0 * s + x1 * c) as f32;
            }
        }
    }
    Ok(out)
}

/// Deterministic pseudo-random float generator in `[-1.0, 1.0]` for reproducible tests.
pub fn deterministic_f32_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((state >> 40) as u32) as f32 / (16_777_216.0_f32);
        out.push(bits * 2.0 - 1.0);
    }
    out
}

/// Compares actual GPU results against expected reference values using `abs_tol + rel_tol * |expected|`.
pub fn compare_slices(
    actual: &[f32],
    expected: &[f32],
    abs_tol: f64,
    rel_tol: f64,
) -> ForgeResult<NumericalAccuracy> {
    if actual.len() != expected.len() {
        return Err(ForgeError::InvalidDimension(format!(
            "comparison slice length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        )));
    }
    if actual.is_empty() {
        return Err(ForgeError::InvalidDimension(
            "cannot compare empty slices".to_string(),
        ));
    }

    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    let mut all_finite = true;
    let mut passed = true;

    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if !a.is_finite() || !e.is_finite() {
            all_finite = false;
            passed = false;
            continue;
        }
        let af = f64::from(a);
        let ef = f64::from(e);
        let abs_err = (af - ef).abs();
        let rel_err = abs_err / ef.abs().max(1e-6);
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
        sum_sq_err += abs_err * abs_err;
        if abs_err > abs_tol + rel_tol * ef.abs() {
            passed = false;
        }
    }

    let rmse = (sum_sq_err / (actual.len() as f64)).sqrt();

    Ok(NumericalAccuracy {
        element_count: actual.len(),
        max_abs_err,
        max_rel_err,
        rmse,
        all_finite,
        passed,
    })
}

/// Deterministic pseudo-random signed `i8` generator in `[-64, 63]` to prevent `i32` matmul overflow.
pub fn deterministic_i8_data(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed ^ 0xA076_1D64_78BD_642F;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((state >> 57) as i8) - 64;
        out.push(v);
    }
    out
}

/// Packs row-major `A[rows, k]` of signed `i8` along `k` (4 bytes per `u32`, little-endian).
pub fn pack_i8_row_major_k4(a: &[i8], rows: usize, k: usize) -> ForgeResult<Vec<u32>> {
    if rows == 0 || k == 0 || !k.is_multiple_of(4) {
        return Err(ForgeError::InvalidDimension(format!(
            "packed INT8 A requires non-zero rows and k divisible by 4, got ({rows}, {k})"
        )));
    }
    let expected = rows
        .checked_mul(k)
        .ok_or_else(|| ForgeError::InvalidDimension("rows * k overflow".to_string()))?;
    if a.len() != expected {
        return Err(ForgeError::InvalidDimension(format!(
            "packed INT8 A length mismatch: got {}, expected {expected}",
            a.len()
        )));
    }

    let k_packed = k / 4;
    let mut out = Vec::with_capacity(rows * k_packed);
    for r in 0..rows {
        let row = &a[r * k..(r + 1) * k];
        for kp in 0..k_packed {
            let base = kp * 4;
            let bytes = [
                row[base] as u8,
                row[base + 1] as u8,
                row[base + 2] as u8,
                row[base + 3] as u8,
            ];
            out.push(u32::from_le_bytes(bytes));
        }
    }
    Ok(out)
}

/// Packs column-major transposed `B[k, n]` into `B_t[n, k/4]` so each `u32` holds 4 consecutive `k` elements.
pub fn pack_i8_b_transposed_k4(b: &[i8], k: usize, n: usize) -> ForgeResult<Vec<u32>> {
    if k == 0 || n == 0 || !k.is_multiple_of(4) {
        return Err(ForgeError::InvalidDimension(format!(
            "packed INT8 B requires non-zero n and k divisible by 4, got ({k}, {n})"
        )));
    }
    let expected = k
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("k * n overflow".to_string()))?;
    if b.len() != expected {
        return Err(ForgeError::InvalidDimension(format!(
            "packed INT8 B length mismatch: got {}, expected {expected}",
            b.len()
        )));
    }

    let k_packed = k / 4;
    let mut out = Vec::with_capacity(n * k_packed);
    for col in 0..n {
        for kp in 0..k_packed {
            let k0 = kp * 4;
            let bytes = [
                b[k0 * n + col] as u8,
                b[(k0 + 1) * n + col] as u8,
                b[(k0 + 2) * n + col] as u8,
                b[(k0 + 3) * n + col] as u8,
            ];
            out.push(u32::from_le_bytes(bytes));
        }
    }
    Ok(out)
}

/// Exact CPU reference INT8 matrix multiplication `C[m, n] = A[m, k] * B[k, n]` accumulated in `i64`.
pub fn ref_gemm_i8_to_i32(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> ForgeResult<Vec<i32>> {
    if m == 0 || k == 0 || n == 0 {
        return Err(ForgeError::InvalidDimension(
            "INT8 GEMM dimensions m, k, n must all be > 0".to_string(),
        ));
    }
    let expected_a = m
        .checked_mul(k)
        .ok_or_else(|| ForgeError::InvalidDimension("m * k overflow".to_string()))?;
    let expected_b = k
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("k * n overflow".to_string()))?;
    let expected_c = m
        .checked_mul(n)
        .ok_or_else(|| ForgeError::InvalidDimension("m * n overflow".to_string()))?;
    if a.len() != expected_a || b.len() != expected_b {
        return Err(ForgeError::InvalidDimension(format!(
            "INT8 GEMM buffer mismatch: got a={}, b={}, expected a={expected_a}, b={expected_b}",
            a.len(),
            b.len()
        )));
    }

    let mut acc = vec![0i64; expected_c];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let acc_row = &mut acc[row * n..(row + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let av64 = i64::from(av);
            let b_row = &b[kk * n..(kk + 1) * n];
            for col in 0..n {
                acc_row[col] += av64 * i64::from(b_row[col]);
            }
        }
    }
    Ok(acc.into_iter().map(|v| v as i32).collect())
}

/// Compares exact integer GPU outputs against an integer CPU reference.
pub fn compare_i32_exact(actual: &[i32], expected: &[i32]) -> ForgeResult<NumericalAccuracy> {
    if actual.len() != expected.len() {
        return Err(ForgeError::InvalidDimension(format!(
            "integer comparison length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        )));
    }
    if actual.is_empty() {
        return Err(ForgeError::InvalidDimension(
            "cannot compare empty slices".to_string(),
        ));
    }

    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;
    let mut sum_sq_err = 0.0f64;

    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let diff = (i64::from(a) - i64::from(e)).abs() as f64;
        max_abs_err = max_abs_err.max(diff);
        if e != 0 {
            max_rel_err = max_rel_err.max(diff / (e.abs() as f64));
        }
        sum_sq_err += diff * diff;
    }

    let rmse = (sum_sq_err / (actual.len() as f64)).sqrt();
    Ok(NumericalAccuracy {
        element_count: actual.len(),
        max_abs_err,
        max_rel_err,
        rmse,
        all_finite: true,
        passed: max_abs_err == 0.0,
    })
}

/// Elements per quantized `Q4_0` block (32 4-bit weights + 1 scalar scale).
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Generates deterministic `Q4_0` packed 4-bit weights (`[0..15]` biased `-8..+7`) and positive scales.
pub fn deterministic_q4_0_weights(
    n: usize,
    k: usize,
    seed: u64,
) -> ForgeResult<(Vec<u32>, Vec<f32>)> {
    if n == 0 || k == 0 || !k.is_multiple_of(Q4_0_BLOCK_SIZE) {
        return Err(ForgeError::InvalidDimension(format!(
            "Q4_0 weights require n > 0 and k divisible by {Q4_0_BLOCK_SIZE}, got ({n}, {k})"
        )));
    }
    let n_blocks = k / Q4_0_BLOCK_SIZE;
    let total_blocks = n
        .checked_mul(n_blocks)
        .ok_or_else(|| ForgeError::InvalidDimension("Q4_0 block count overflow".to_string()))?;
    let total_words = total_blocks
        .checked_mul(4)
        .ok_or_else(|| ForgeError::InvalidDimension("Q4_0 word count overflow".to_string()))?;

    let mut state = seed ^ 0x517C_C1B7_2722_0A95;
    let mut next_u64 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let mut packed = Vec::with_capacity(total_words);
    for _ in 0..total_words {
        let mut word = 0u32;
        for nib in 0..8 {
            let q = ((next_u64() >> 60) as u32) & 0xF;
            word |= q << (nib * 4);
        }
        packed.push(word);
    }

    let mut scales = Vec::with_capacity(total_blocks);
    for _ in 0..total_blocks {
        let frac = ((next_u64() >> 40) as f32) / 16_777_216.0_f32;
        scales.push(0.01 + frac * 0.24);
    }

    Ok((packed, scales))
}

/// Exact FP64 CPU reference for row-scaled INT8 weight-only GEMV: `y[n] = scales[n] * dot(W[n, k], x[k])`.
pub fn ref_gemv_w8a32_f32(
    w_i8: &[i8],
    scales: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
) -> ForgeResult<Vec<f32>> {
    if n == 0 || k == 0 {
        return Err(ForgeError::InvalidDimension(
            "W8A32 GEMV dimensions n and k must be > 0".to_string(),
        ));
    }
    let expected_w = n
        .checked_mul(k)
        .ok_or_else(|| ForgeError::InvalidDimension("n * k overflow".to_string()))?;
    if w_i8.len() != expected_w || scales.len() != n || x.len() != k {
        return Err(ForgeError::InvalidDimension(format!(
            "W8A32 GEMV buffer mismatch: got w={}, scales={}, x={}, expected ({expected_w}, {n}, {k})",
            w_i8.len(),
            scales.len(),
            x.len()
        )));
    }

    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let w_row = &w_i8[row * k..(row + 1) * k];
        let mut acc = 0.0f64;
        for (&w, &xv) in w_row.iter().zip(x.iter()) {
            acc += f64::from(w) * f64::from(xv);
        }
        out.push((acc * f64::from(scales[row])) as f32);
    }
    Ok(out)
}

/// Exact FP64 CPU reference for block-32 `Q4_0` dequantized GEMV matching `gemv_q4_0_wave32.wgsl`.
pub fn ref_gemv_q4_0_f32(
    w_q4_packed: &[u32],
    scales: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
) -> ForgeResult<Vec<f32>> {
    if n == 0 || k == 0 || !k.is_multiple_of(Q4_0_BLOCK_SIZE) {
        return Err(ForgeError::InvalidDimension(format!(
            "Q4_0 GEMV requires n > 0 and k divisible by {Q4_0_BLOCK_SIZE}, got ({n}, {k})"
        )));
    }
    let n_blocks = k / Q4_0_BLOCK_SIZE;
    let total_blocks = n
        .checked_mul(n_blocks)
        .ok_or_else(|| ForgeError::InvalidDimension("Q4_0 block count overflow".to_string()))?;
    let expected_words = total_blocks
        .checked_mul(4)
        .ok_or_else(|| ForgeError::InvalidDimension("Q4_0 word count overflow".to_string()))?;
    if w_q4_packed.len() != expected_words || scales.len() != total_blocks || x.len() != k {
        return Err(ForgeError::InvalidDimension(format!(
            "Q4_0 GEMV buffer mismatch: got w_words={}, scales={}, x={}, expected ({expected_words}, {total_blocks}, {k})",
            w_q4_packed.len(),
            scales.len(),
            x.len()
        )));
    }

    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let mut row_acc = 0.0f64;
        for blk in 0..n_blocks {
            let blk_idx = row * n_blocks + blk;
            let q_base = blk_idx * 4;
            let x_base = blk * Q4_0_BLOCK_SIZE;
            let mut blk_acc = 0.0f64;
            for j in 0..4 {
                let word = w_q4_packed[q_base + j];
                for nib in 0..8 {
                    let qval = ((word >> (nib * 4)) & 0xF) as i32 - 8;
                    blk_acc += f64::from(qval) * f64::from(x[x_base + j * 8 + nib]);
                }
            }
            row_acc += blk_acc * f64::from(scales[blk_idx]);
        }
        out.push(row_acc as f32);
    }
    Ok(out)
}

/// Exact FP64 CPU reference for fused RMSNorm + RoPE with KV-cache position offset `pos_offset`.
#[allow(clippy::too_many_arguments)]
pub fn ref_rmsnorm_rope_f32(
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    eps: f32,
    theta_base: f32,
) -> ForgeResult<Vec<f32>> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(ForgeError::InvalidDimension(format!(
            "fused RMSNorm+RoPE requires non-zero dims and even head_dim, got ({seq_len}, {n_heads}, {head_dim})"
        )));
    }
    if theta_base <= 0.0 || !theta_base.is_finite() {
        return Err(ForgeError::InvalidDimension(format!(
            "theta_base must be positive and finite, got {theta_base}"
        )));
    }
    let cols = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ForgeError::InvalidDimension("n_heads * head_dim overflow".to_string()))?;
    let total = seq_len
        .checked_mul(cols)
        .ok_or_else(|| ForgeError::InvalidDimension("seq_len * cols overflow".to_string()))?;
    if x.len() != total || weight.len() != cols {
        return Err(ForgeError::InvalidDimension(format!(
            "fused RMSNorm+RoPE buffer mismatch: got x={}, w={}, expected ({total}, {cols})",
            x.len(),
            weight.len()
        )));
    }

    let half_dim = head_dim / 2;
    let mut out = vec![0.0f32; total];
    for row in 0..seq_len {
        let row_off = row * cols;
        let row_slice = &x[row_off..row_off + cols];
        let sq_sum: f64 = row_slice
            .iter()
            .map(|&v| {
                let d = f64::from(v);
                d * d
            })
            .sum();
        let inv_rms = 1.0 / ((sq_sum / cols as f64) + f64::from(eps)).sqrt();
        let pos = (row + pos_offset) as f64;

        for head in 0..n_heads {
            let head_off = head * head_dim;
            for i in 0..half_dim {
                let exponent = -((2 * i) as f64) / (head_dim as f64);
                let angle = pos * f64::from(theta_base).powf(exponent);
                let cs = angle.cos();
                let sn = angle.sin();
                let c0 = head_off + 2 * i;
                let c1 = c0 + 1;
                let n0 = f64::from(row_slice[c0]) * inv_rms * f64::from(weight[c0]);
                let n1 = f64::from(row_slice[c1]) * inv_rms * f64::from(weight[c1]);
                out[row_off + c0] = (n0 * cs - n1 * sn) as f32;
                out[row_off + c1] = (n0 * sn + n1 * cs) as f32;
            }
        }
    }

    Ok(out)
}
