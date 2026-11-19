struct Params {
    n: u32,
    n_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> w_q4k: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn unpack_f16(word: u32, shift: u32) -> f32 {
    let bits = (word >> shift) & 0xFFFFu;
    let sign = (bits & 0x8000u) << 16u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x03FFu;
    if (exp == 0u) {
        if (mant == 0u) {
            return bitcast<f32>(sign);
        }
        let mag = f32(mant) * 5.9604645e-8;
        return select(mag, -mag, sign != 0u);
    }
    return bitcast<f32>(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

fn eval_q4k_chunk(
    base: u32,
    c: u32,
    lane: u32,
    xb: u32,
    d_sc_lo: f32,
    d_m_lo: f32,
    d_sc_hi: f32,
    d_m_hi: f32,
) -> f32 {
    let q_word = w_q4k[base + 4u + c * 8u + (lane >> 2u)];
    let q_byte = (q_word >> ((lane & 3u) * 8u)) & 0xFFu;
    let q0 = f32(q_byte & 0xFu);
    let q1 = f32(q_byte >> 4u);

    let x_off = xb + c * 64u + lane;
    let x0 = x[x_off];
    let x1 = x[x_off + 32u];

    let w0 = fma(d_sc_lo, q0, -d_m_lo);
    let w1 = fma(d_sc_hi, q1, -d_m_hi);
    return fma(w0, x0, w1 * x1);
}

// 4 rows per workgroup, 1 Wave32 (32 lanes) per row -> zero LDS, zero barriers.
// Each super-block is 36 u32 words (144 B):
//   word 0       = d (f16 lo) | dmin (f16 hi)
//   words 1..3   = packed 6-bit scales & mins for 8 sub-blocks of 32 weights
//   words 4..35  = 128 bytes of packed 4-bit nibbles (4 chunks x 64 weights)
@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lane = local_id.x;
    let row = wg_id.x * 4u + local_id.y;

    var acc: f32 = 0.0;
    if (row < params.n) {
        let row_block_off = row * params.n_blocks;
        var blk: u32 = 0u;
        loop {
            if (blk >= params.n_blocks) {
                break;
            }
            let base = (row_block_off + blk) * 36u;
            let dm_word = w_q4k[base];
            let d = unpack_f16(dm_word, 0u);
            let dmin = unpack_f16(dm_word, 16u);

            let s0 = w_q4k[base + 1u];
            let s1 = w_q4k[base + 2u];
            let s2 = w_q4k[base + 3u];
            let xb = blk * 256u;

            // Chunks 0 and 1 use sub-blocks (0, 1) and (2, 3) stored directly in s0/s1 bytes 0..3.
            for (var c: u32 = 0u; c < 2u; c = c + 1u) {
                let sh0 = c * 16u;
                let sh1 = sh0 + 8u;
                let sc_lo = (s0 >> sh0) & 63u;
                let m_lo = (s1 >> sh0) & 63u;
                let sc_hi = (s0 >> sh1) & 63u;
                let m_hi = (s1 >> sh1) & 63u;
                acc = acc + eval_q4k_chunk(
                    base,
                    c,
                    lane,
                    xb,
                    d * f32(sc_lo),
                    dmin * f32(m_lo),
                    d * f32(sc_hi),
                    dmin * f32(m_hi),
                );
            }

            // Chunks 2 and 3 use sub-blocks (4, 5) and (6, 7) split across s2 (lo 4 bits) and s0/s1 (hi 2 bits).
            for (var c: u32 = 2u; c < 4u; c = c + 1u) {
                let sh0 = (c - 2u) * 16u;
                let sh1 = sh0 + 8u;
                let sc_lo = ((s2 >> sh0) & 0xFu) | (((s0 >> (sh0 + 6u)) & 3u) << 4u);
                let m_lo = ((s2 >> (sh0 + 4u)) & 0xFu) | (((s1 >> (sh0 + 6u)) & 3u) << 4u);
                let sc_hi = ((s2 >> sh1) & 0xFu) | (((s0 >> (sh1 + 6u)) & 3u) << 4u);
                let m_hi = ((s2 >> (sh1 + 4u)) & 0xFu) | (((s1 >> (sh1 + 6u)) & 3u) << 4u);
                acc = acc + eval_q4k_chunk(
                    base,
                    c,
                    lane,
                    xb,
                    d * f32(sc_lo),
                    dmin * f32(m_lo),
                    d * f32(sc_hi),
                    dmin * f32(m_hi),
                );
            }

            blk = blk + 1u;
        }
    }

    let total = subgroupAdd(acc);
    if (lane == 0u && row < params.n) {
        y[row] = total;
    }
}
