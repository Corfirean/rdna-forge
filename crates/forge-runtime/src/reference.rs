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
