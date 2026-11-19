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

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;
    let tile_idx = local_id.y * TILE + local_id.x;
    let num_tiles = (params.k + TILE - 1u) / TILE;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + local_id.x;
        if (row < params.m && a_col < params.k) {
            tile_a[tile_idx] = a[row * params.k + a_col];
        } else {
            tile_a[tile_idx] = 0.0;
        }

        let b_row = t * TILE + local_id.y;
        if (b_row < params.k && col < params.n) {
            tile_b[tile_idx] = b[b_row * params.n + col];
        } else {
            tile_b[tile_idx] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[local_id.y * TILE + i] * tile_b[i * TILE + local_id.x];
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        c[row * params.n + col] = acc;
    }
}
