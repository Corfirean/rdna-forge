struct Params {
    total_heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> q_buf: array<f32>;
@group(0) @binding(1) var<storage, read> k_buf: array<f32>;
@group(0) @binding(2) var<storage, read> v_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> o_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// 16 keys x 128 max head_dim x 4 B = 8 KiB each -> 16 KiB total LDS (< 32 KiB Vulkan cap).
var<workgroup> s_k: array<f32, 2048>;
var<workgroup> s_v: array<f32, 2048>;

// 4 query rows per workgroup (1 Wave32 subgroup per row), BC=16 K/V tile.
// Streaming online softmax keeps O(S) score memory in registers and skips upper-triangle tiles.
@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lane = local_id.x;
    let row_in_wg = local_id.y;
    let wg_row_base = wg_id.x * 4u;
    let q_row = wg_row_base + row_in_wg;
    let head = wg_id.y;
    let head_dim = params.head_dim;
    let row_active = (q_row < params.seq_len) && (head < params.total_heads);
    let head_off = head * params.seq_len * head_dim;
    let q_base = head_off + q_row * head_dim;

    var q0: f32 = 0.0;
    var q1: f32 = 0.0;
    var q2: f32 = 0.0;
    var q3: f32 = 0.0;
    if (row_active) {
        if (lane < head_dim) {
            q0 = q_buf[q_base + lane];
        }
        if (lane + 32u < head_dim) {
            q1 = q_buf[q_base + lane + 32u];
        }
        if (lane + 64u < head_dim) {
            q2 = q_buf[q_base + lane + 64u];
        }
        if (lane + 96u < head_dim) {
            q3 = q_buf[q_base + lane + 96u];
        }
    }

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var m_i: f32 = -1.0e30;
    var l_i: f32 = 0.0;

    // Workgroup-uniform upper bound on K/V positions.
    let max_wg_k = min(wg_row_base + 4u, params.seq_len);
    let k_limit = select(params.seq_len, max_wg_k, params.causal != 0u);
    let num_tiles = (k_limit + 15u) / 16u;

    var tile: u32 = 0u;
    loop {
        if (tile >= num_tiles) {
            break;
        }
        let tile_k_start = tile * 16u;
        let tile_keys = min(16u, k_limit - tile_k_start);

        // Cooperatively load up to 16 K/V rows using all 128 workgroup threads.
        for (var r: u32 = 0u; r < 4u; r = r + 1u) {
            let j = r * 4u + row_in_wg;
            if (j < tile_keys) {
                let kv_global = head_off + (tile_k_start + j) * head_dim;
                let lds_row = j * 128u;
                if (lane < head_dim) {
                    s_k[lds_row + lane] = k_buf[kv_global + lane];
                    s_v[lds_row + lane] = v_buf[kv_global + lane];
                }
                if (lane + 32u < head_dim) {
                    s_k[lds_row + lane + 32u] = k_buf[kv_global + lane + 32u];
                    s_v[lds_row + lane + 32u] = v_buf[kv_global + lane + 32u];
                }
                if (head_dim > 64u) {
                    if (lane + 64u < head_dim) {
                        s_k[lds_row + lane + 64u] = k_buf[kv_global + lane + 64u];
                        s_v[lds_row + lane + 64u] = v_buf[kv_global + lane + 64u];
                    }
                    if (lane + 96u < head_dim) {
                        s_k[lds_row + lane + 96u] = k_buf[kv_global + lane + 96u];
                        s_v[lds_row + lane + 96u] = v_buf[kv_global + lane + 96u];
                    }
                }
            }
        }
        workgroupBarrier();

        var j: u32 = 0u;
        loop {
            if (j >= tile_keys) {
                break;
            }
            let lds_row = j * 128u;
            var partial: f32 = 0.0;
            if (lane < head_dim) {
                partial = fma(q0, s_k[lds_row + lane], partial);
            }
            if (lane + 32u < head_dim) {
                partial = fma(q1, s_k[lds_row + lane + 32u], partial);
            }
            if (head_dim > 64u) {
                if (lane + 64u < head_dim) {
                    partial = fma(q2, s_k[lds_row + lane + 64u], partial);
                }
                if (lane + 96u < head_dim) {
                    partial = fma(q3, s_k[lds_row + lane + 96u], partial);
                }
            }

            let qk = subgroupAdd(partial);
            let k_pos = tile_k_start + j;
            let key_valid = (params.causal == 0u) || (k_pos <= q_row);

            if (row_active && key_valid) {
                let score = qk * params.scale;
                let m_new = max(m_i, score);
                let alpha = exp(m_i - m_new);
                let p = exp(score - m_new);
                l_i = fma(l_i, alpha, p);
                m_i = m_new;

                if (lane < head_dim) {
                    acc0 = fma(acc0, alpha, p * s_v[lds_row + lane]);
                }
                if (lane + 32u < head_dim) {
                    acc1 = fma(acc1, alpha, p * s_v[lds_row + lane + 32u]);
                }
                if (head_dim > 64u) {
                    if (lane + 64u < head_dim) {
                        acc2 = fma(acc2, alpha, p * s_v[lds_row + lane + 64u]);
                    }
                    if (lane + 96u < head_dim) {
                        acc3 = fma(acc3, alpha, p * s_v[lds_row + lane + 96u]);
                    }
                }
            }
            j = j + 1u;
        }

        workgroupBarrier();
        tile = tile + 1u;
    }

    if (row_active && l_i > 0.0) {
        let inv_l = 1.0 / l_i;
        if (lane < head_dim) {
            o_buf[q_base + lane] = acc0 * inv_l;
        }
        if (lane + 32u < head_dim) {
            o_buf[q_base + lane + 32u] = acc1 * inv_l;
        }
        if (head_dim > 64u) {
            if (lane + 64u < head_dim) {
                o_buf[q_base + lane + 64u] = acc2 * inv_l;
            }
            if (lane + 96u < head_dim) {
                o_buf[q_base + lane + 96u] = acc3 * inv_l;
            }
        }
    }
}
