struct Params {
    n: u32,
    n_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> w_q4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn dot_q4_u32(word: u32, x0: vec4<f32>, x1: vec4<f32>) -> f32 {
    let shifts_lo = vec4<u32>(0u, 4u, 8u, 12u);
    let shifts_hi = vec4<u32>(16u, 20u, 24u, 28u);
    let nib_lo = (vec4<u32>(word) >> shifts_lo) & vec4<u32>(0xFu);
    let nib_hi = (vec4<u32>(word) >> shifts_hi) & vec4<u32>(0xFu);
    let w_lo = vec4<f32>(nib_lo) - vec4<f32>(8.0);
    let w_hi = vec4<f32>(nib_hi) - vec4<f32>(8.0);
    return dot(w_lo, x0) + dot(w_hi, x1);
}

// 4 rows per workgroup, 1 Wave32 (32 lanes) per row -> zero LDS, zero barriers.
@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lane = local_id.x;
    let row = wg_id.x * 4u + local_id.y;

    var acc: f32 = 0.0;
    if (row < params.n) {
        let row_off = row * params.n_blocks;
        var blk: u32 = lane;
        loop {
            if (blk >= params.n_blocks) {
                break;
            }
            let q = w_q4[row_off + blk];
            let s = scales[row_off + blk];
            let xb = blk * 8u;
            var blk_sum: f32 = dot_q4_u32(q.x, x[xb + 0u], x[xb + 1u]);
            blk_sum = blk_sum + dot_q4_u32(q.y, x[xb + 2u], x[xb + 3u]);
            blk_sum = blk_sum + dot_q4_u32(q.z, x[xb + 4u], x[xb + 5u]);
            blk_sum = blk_sum + dot_q4_u32(q.w, x[xb + 6u], x[xb + 7u]);
            acc = fma(blk_sum, s, acc);
            blk = blk + 32u;
        }
    }

    let total = subgroupAdd(acc);
    if (lane == 0u && row < params.n) {
        y[row] = total;
    }
}
