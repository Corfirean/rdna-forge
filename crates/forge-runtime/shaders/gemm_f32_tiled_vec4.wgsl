struct Params {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Workgroup tile: 128 x 64 x 16 computed by 16 x 16 threads (8 x 4 register block per thread).
// Shared memory: tile_a[16][129] = 2064 f32 (transposed + bank padded) + tile_b[16][65] = 1040 f32
// = 12,416 bytes total (fits comfortably within 32 KiB Vulkan maxComputeSharedMemorySize).
const BM: u32 = 128u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const STRIDE_A: u32 = 129u;
const STRIDE_B: u32 = 65u;

var<workgroup> tile_a: array<f32, 2064>;
var<workgroup> tile_b: array<f32, 1040>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let tid = ly * 16u + lx;

    let wg_m = wg_id.y * BM;
    let wg_n = wg_id.x * BN;

    // 8 rows x 4 cols per thread accumulated in vec4 registers.
    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);
    var acc4: vec4<f32> = vec4<f32>(0.0);
    var acc5: vec4<f32> = vec4<f32>(0.0);
    var acc6: vec4<f32> = vec4<f32>(0.0);
    var acc7: vec4<f32> = vec4<f32>(0.0);

    // Load coordinates for A (128 x 16 = 2048 floats -> 2 chunks of 4 floats per thread)
    let a_m0 = tid / 4u;
    let a_m1 = a_m0 + 64u;
    let a_k_base = (tid % 4u) * 4u;
    let gm0 = wg_m + a_m0;
    let gm1 = wg_m + a_m1;

    // Load coordinates for B (16 x 64 = 1024 floats -> 1 chunk of 4 floats per thread)
    let b_k_loc = tid / 16u;
    let b_n_base = (tid % 16u) * 4u;
    let gn = wg_n + b_n_base;

    let num_tiles = (params.k + BK - 1u) / BK;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k0 = t * BK;
        let gk_a = k0 + a_k_base;

        if (gm0 < params.m) {
            let base_a0 = gm0 * params.k + gk_a;
            if (gk_a + 3u < params.k) {
                tile_a[(a_k_base + 0u) * STRIDE_A + a_m0] = a[base_a0 + 0u];
                tile_a[(a_k_base + 1u) * STRIDE_A + a_m0] = a[base_a0 + 1u];
                tile_a[(a_k_base + 2u) * STRIDE_A + a_m0] = a[base_a0 + 2u];
                tile_a[(a_k_base + 3u) * STRIDE_A + a_m0] = a[base_a0 + 3u];
            } else {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    var val: f32 = 0.0;
                    if (gk_a + j < params.k) {
                        val = a[base_a0 + j];
                    }
                    tile_a[(a_k_base + j) * STRIDE_A + a_m0] = val;
                }
            }
        } else {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                tile_a[(a_k_base + j) * STRIDE_A + a_m0] = 0.0;
            }
        }

        if (gm1 < params.m) {
            let base_a1 = gm1 * params.k + gk_a;
            if (gk_a + 3u < params.k) {
                tile_a[(a_k_base + 0u) * STRIDE_A + a_m1] = a[base_a1 + 0u];
                tile_a[(a_k_base + 1u) * STRIDE_A + a_m1] = a[base_a1 + 1u];
                tile_a[(a_k_base + 2u) * STRIDE_A + a_m1] = a[base_a1 + 2u];
                tile_a[(a_k_base + 3u) * STRIDE_A + a_m1] = a[base_a1 + 3u];
            } else {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    var val: f32 = 0.0;
                    if (gk_a + j < params.k) {
                        val = a[base_a1 + j];
                    }
                    tile_a[(a_k_base + j) * STRIDE_A + a_m1] = val;
                }
            }
        } else {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                tile_a[(a_k_base + j) * STRIDE_A + a_m1] = 0.0;
            }
        }

        let gk_b = k0 + b_k_loc;
        let b_tile_off = b_k_loc * STRIDE_B + b_n_base;
        if (gk_b < params.k) {
            let base_b = gk_b * params.n + gn;
            if (gn + 3u < params.n) {
                tile_b[b_tile_off + 0u] = b[base_b + 0u];
                tile_b[b_tile_off + 1u] = b[base_b + 1u];
                tile_b[b_tile_off + 2u] = b[base_b + 2u];
                tile_b[b_tile_off + 3u] = b[base_b + 3u];
            } else {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    var val: f32 = 0.0;
                    if (gn + j < params.n) {
                        val = b[base_b + j];
                    }
                    tile_b[b_tile_off + j] = val;
                }
            }
        } else {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                tile_b[b_tile_off + j] = 0.0;
            }
        }

        workgroupBarrier();

        let a_row_loc = ly * 8u;
        let b_col_loc = lx * 4u;

        for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {
            let b_idx = kk * STRIDE_B + b_col_loc;
            let bv = vec4<f32>(
                tile_b[b_idx + 0u],
                tile_b[b_idx + 1u],
                tile_b[b_idx + 2u],
                tile_b[b_idx + 3u],
            );

            let a_idx = kk * STRIDE_A + a_row_loc;
            let av0 = vec4<f32>(
                tile_a[a_idx + 0u],
                tile_a[a_idx + 1u],
                tile_a[a_idx + 2u],
                tile_a[a_idx + 3u],
            );
            let av1 = vec4<f32>(
                tile_a[a_idx + 4u],
                tile_a[a_idx + 5u],
                tile_a[a_idx + 6u],
                tile_a[a_idx + 7u],
            );

            acc0 = fma(vec4<f32>(av0.x), bv, acc0);
            acc1 = fma(vec4<f32>(av0.y), bv, acc1);
            acc2 = fma(vec4<f32>(av0.z), bv, acc2);
            acc3 = fma(vec4<f32>(av0.w), bv, acc3);
            acc4 = fma(vec4<f32>(av1.x), bv, acc4);
            acc5 = fma(vec4<f32>(av1.y), bv, acc5);
            acc6 = fma(vec4<f32>(av1.z), bv, acc6);
            acc7 = fma(vec4<f32>(av1.w), bv, acc7);
        }

        workgroupBarrier();
    }

    let out_m_base = wg_m + ly * 8u;
    let out_n_base = wg_n + lx * 4u;

    let accs = array<vec4<f32>, 8>(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    if (out_n_base + 3u < params.n) {
        for (var r: u32 = 0u; r < 8u; r = r + 1u) {
            let row = out_m_base + r;
            if (row < params.m) {
                let c_off = row * params.n + out_n_base;
                let v = accs[r];
                c[c_off + 0u] = v.x;
                c[c_off + 1u] = v.y;
                c[c_off + 2u] = v.z;
                c[c_off + 3u] = v.w;
            }
        }
    } else {
        for (var r: u32 = 0u; r < 8u; r = r + 1u) {
            let row = out_m_base + r;
            if (row < params.m) {
                let c_off = row * params.n + out_n_base;
                let v = accs[r];
                if (out_n_base + 0u < params.n) { c[c_off + 0u] = v.x; }
                if (out_n_base + 1u < params.n) { c[c_off + 1u] = v.y; }
                if (out_n_base + 2u < params.n) { c[c_off + 2u] = v.z; }
                if (out_n_base + 3u < params.n) { c[c_off + 3u] = v.w; }
            }
        }
    }
}
