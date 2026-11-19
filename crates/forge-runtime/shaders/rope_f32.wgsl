struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
}

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.seq_len * params.n_heads * half_dim;
    let pair_idx = global_id.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    let head_pairs = params.n_heads * half_dim;
    let pos = pair_idx / head_pairs;
    let rem = pair_idx % head_pairs;
    let head = rem / half_dim;
    let i = rem % half_dim;

    let exponent = -f32(2u * i) / f32(params.head_dim);
    let freq = pow(params.theta_base, exponent);
    let angle = f32(pos) * freq;
    let c = cos(angle);
    let s = sin(angle);

    let base = (pos * params.n_heads + head) * params.head_dim + 2u * i;
    let x0 = input_buf[base];
    let x1 = input_buf[base + 1u];

    output_buf[base] = x0 * c - x1 * s;
    output_buf[base + 1u] = x0 * s + x1 * c;
}
