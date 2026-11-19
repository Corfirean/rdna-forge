requires packed_4x8_integer_dot_product;

struct Params {
    m: u32,
    k_packed: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a_packed: array<u32>;
@group(0) @binding(1) var<storage, read> b_t_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

// Workgroup tile: BM=64, BN=64, BK_PACKED=16 (64 INT8 elements along K per tile).
// Total LDS: (1088 + 1088) * 4 = 8,704 bytes.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK_P: u32 = 16u;
const STRIDE_A: u32 = 17u;
const STRIDE_B: u32 = 65u;

var<workgroup> tile_a: array<u32, 1088>;
var<workgroup> tile_b: array<u32, 1088>;

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

    // 4 rows x 4 cols per thread (col stride 16 across N eliminates LDS bank conflicts and coalesces stores).
    var acc0: vec4<i32> = vec4<i32>(0);
    var acc1: vec4<i32> = vec4<i32>(0);
    var acc2: vec4<i32> = vec4<i32>(0);
    var acc3: vec4<i32> = vec4<i32>(0);

    // 64 rows * 16 k_packed = 1024 u32s -> 4 consecutive u32s per thread.
    let rc_loc = tid / 4u;
    let kp_base = (tid % 4u) * 4u;
    let gm = wg_m + rc_loc;
    let gn = wg_n + rc_loc;

    let num_tiles = (params.k_packed + BK_P - 1u) / BK_P;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let kp0 = t * BK_P + kp_base;

        let a_lds_off = rc_loc * STRIDE_A + kp_base;
        if (gm < params.m) {
            let a_glb_off = gm * params.k_packed + kp0;
            if (kp0 + 3u < params.k_packed) {
                tile_a[a_lds_off + 0u] = a_packed[a_glb_off + 0u];
                tile_a[a_lds_off + 1u] = a_packed[a_glb_off + 1u];
                tile_a[a_lds_off + 2u] = a_packed[a_glb_off + 2u];
                tile_a[a_lds_off + 3u] = a_packed[a_glb_off + 3u];
            } else {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    var val: u32 = 0u;
                    if (kp0 + j < params.k_packed) {
                        val = a_packed[a_glb_off + j];
                    }
                    tile_a[a_lds_off + j] = val;
                }
            }
        } else {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                tile_a[a_lds_off + j] = 0u;
            }
        }

        if (gn < params.n) {
            let b_glb_off = gn * params.k_packed + kp0;
            if (kp0 + 3u < params.k_packed) {
                tile_b[(kp_base + 0u) * STRIDE_B + rc_loc] = b_t_packed[b_glb_off + 0u];
                tile_b[(kp_base + 1u) * STRIDE_B + rc_loc] = b_t_packed[b_glb_off + 1u];
                tile_b[(kp_base + 2u) * STRIDE_B + rc_loc] = b_t_packed[b_glb_off + 2u];
                tile_b[(kp_base + 3u) * STRIDE_B + rc_loc] = b_t_packed[b_glb_off + 3u];
            } else {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    var val: u32 = 0u;
                    if (kp0 + j < params.k_packed) {
                        val = b_t_packed[b_glb_off + j];
                    }
                    tile_b[(kp_base + j) * STRIDE_B + rc_loc] = val;
                }
            }
        } else {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                tile_b[(kp_base + j) * STRIDE_B + rc_loc] = 0u;
            }
        }

        workgroupBarrier();

        let a_r0 = (ly * 4u + 0u) * STRIDE_A;
        let a_r1 = (ly * 4u + 1u) * STRIDE_A;
        let a_r2 = (ly * 4u + 2u) * STRIDE_A;
        let a_r3 = (ly * 4u + 3u) * STRIDE_A;

        for (var kk: u32 = 0u; kk < BK_P; kk = kk + 1u) {
            let a0 = tile_a[a_r0 + kk];
            let a1 = tile_a[a_r1 + kk];
            let a2 = tile_a[a_r2 + kk];
            let a3 = tile_a[a_r3 + kk];

            let b_base = kk * STRIDE_B + lx;
            let b0 = tile_b[b_base + 0u];
            let b1 = tile_b[b_base + 16u];
            let b2 = tile_b[b_base + 32u];
            let b3 = tile_b[b_base + 48u];

            acc0.x = acc0.x + dot4I8Packed(a0, b0);
            acc0.y = acc0.y + dot4I8Packed(a0, b1);
            acc0.z = acc0.z + dot4I8Packed(a0, b2);
            acc0.w = acc0.w + dot4I8Packed(a0, b3);

            acc1.x = acc1.x + dot4I8Packed(a1, b0);
            acc1.y = acc1.y + dot4I8Packed(a1, b1);
            acc1.z = acc1.z + dot4I8Packed(a1, b2);
            acc1.w = acc1.w + dot4I8Packed(a1, b3);

            acc2.x = acc2.x + dot4I8Packed(a2, b0);
            acc2.y = acc2.y + dot4I8Packed(a2, b1);
            acc2.z = acc2.z + dot4I8Packed(a2, b2);
            acc2.w = acc2.w + dot4I8Packed(a2, b3);

            acc3.x = acc3.x + dot4I8Packed(a3, b0);
            acc3.y = acc3.y + dot4I8Packed(a3, b1);
            acc3.z = acc3.z + dot4I8Packed(a3, b2);
            acc3.w = acc3.w + dot4I8Packed(a3, b3);
        }

        workgroupBarrier();
    }

    let out_m_base = wg_m + ly * 4u;
    let c0 = wg_n + lx;
    let c1 = c0 + 16u;
    let c2 = c0 + 32u;
    let c3 = c0 + 48u;

    let accs = array<vec4<i32>, 4>(acc0, acc1, acc2, acc3);
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let row = out_m_base + r;
        if (row < params.m) {
            let row_off = row * params.n;
            let v = accs[r];
            if (c0 < params.n) { c[row_off + c0] = v.x; }
            if (c1 < params.n) { c[row_off + c1] = v.y; }
            if (c2 < params.n) { c[row_off + c2] = v.z; }
            if (c3 < params.n) { c[row_off + c3] = v.w; }
        }
    }
}
