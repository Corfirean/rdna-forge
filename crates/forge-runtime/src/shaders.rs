use forge_core::{ForgeError, ForgeResult};

pub const GEMM_F32_WGSL: &str = include_str!("../shaders/gemm_f32.wgsl");
pub const RMSNORM_F32_WGSL: &str = include_str!("../shaders/rmsnorm_f32.wgsl");
pub const ROPE_F32_WGSL: &str = include_str!("../shaders/rope_f32.wgsl");

/// Compiles a WGSL compute shader into SPIR-V words in pure Rust via `naga`.
pub fn compile_wgsl_to_spirv(source: &str) -> ForgeResult<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|err| ForgeError::ExecutionFailed(err.emit_to_string(source)))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|err| ForgeError::ExecutionFailed(format!("WGSL validation failed: {err}")))?;

    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        fake_missing_bindings: true,
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        emit_int_div_checks: false,
        ..Default::default()
    };

    naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|err| ForgeError::ExecutionFailed(format!("SPIR-V codegen failed: {err}")))
}
