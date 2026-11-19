struct Params {
    n: u32,
    k_packed: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> w_packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

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
        let row_off = row * params.k_packed;
        var kp: u32 = lane;
        loop {
            if (kp >= params.k_packed) {
                break;
            }
            let w_bits = bitcast<i32>(w_packed[row_off + kp]);
            let w0 = f32(extractBits(w_bits, 0u, 8u));
            let w1 = f32(extractBits(w_bits, 8u, 8u));
            let w2 = f32(extractBits(w_bits, 16u, 8u));
            let w3 = f32(extractBits(w_bits, 24u, 8u));
            let wv = vec4<f32>(w0, w1, w2, w3);
            acc = acc + dot(wv, x[kp]);
            kp = kp + 32u;
        }
    }

    // Subgroup reduction must remain in uniform control flow across all 32 lanes.
    let total = subgroupAdd(acc);
    if (lane == 0u && row < params.n) {
        y[row] = total * scales[row];
    }
}
