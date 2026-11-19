struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    pos_offset: u32,
    eps: f32,
    theta_base: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> sg_sums: array<f32, 16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(num_subgroups) num_sg: u32,
) {
    let row = wg_id.x;
    let tid = local_id.x;
    let cols = params.n_heads * params.head_dim;
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.n_heads * half_dim;
    let row_off = row * cols;

    var partial: f32 = 0.0;
    var pair: u32 = tid;
    loop {
        if (pair >= total_pairs) {
            break;
        }
        if (row < params.seq_len) {
            let c = pair * 2u;
            let x0 = x[row_off + c];
            let x1 = x[row_off + c + 1u];
            partial = partial + x0 * x0 + x1 * x1;
        }
        pair = pair + WG_SIZE;
    }

    let sg_sum = subgroupAdd(partial);
    if (sg_lane == 0u) {
        sg_sums[sg_id] = sg_sum;
    }
    workgroupBarrier();

    var lane_val: f32 = 0.0;
    if (tid < num_sg) {
        lane_val = sg_sums[tid];
    }
    let total_sq = subgroupAdd(lane_val);
    if (tid == 0u) {
        sg_sums[0] = inverseSqrt(total_sq / f32(cols) + params.eps);
    }
    workgroupBarrier();

    if (row >= params.seq_len) {
        return;
    }

    let inv_rms = sg_sums[0];
    let pos = f32(row + params.pos_offset);

    pair = tid;
    loop {
        if (pair >= total_pairs) {
            break;
        }
        let c = pair * 2u;
        let pair_in_head = pair % half_dim;
        let exponent = -f32(2u * pair_in_head) / f32(params.head_dim);
        let freq = pow(params.theta_base, exponent);
        let angle = pos * freq;
        let cs = cos(angle);
        let sn = sin(angle);

        let n0 = x[row_off + c] * inv_rms * weight[c];
        let n1 = x[row_off + c + 1u] * inv_rms * weight[c + 1u];
        y[row_off + c] = n0 * cs - n1 * sn;
        y[row_off + c + 1u] = n0 * sn + n1 * cs;

        pair = pair + WG_SIZE;
    }
}
