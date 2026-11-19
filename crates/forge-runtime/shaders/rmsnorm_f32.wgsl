struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = wg_id.x;
    if (row >= params.rows) {
        return;
    }

    let tid = local_id.x;
    let row_offset = row * params.cols;

    var partial: f32 = 0.0;
    var col: u32 = tid;
    loop {
        if (col >= params.cols) {
            break;
        }
        let val = x[row_offset + col];
        partial = partial + val * val;
        col = col + WG_SIZE;
    }

    shared_sum[tid] = partial;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let inv_rms = inverseSqrt(shared_sum[0] / f32(params.cols) + params.eps);

    col = tid;
    loop {
        if (col >= params.cols) {
            break;
        }
        y[row_offset + col] = x[row_offset + col] * inv_rms * weight[col];
        col = col + WG_SIZE;
    }
}
