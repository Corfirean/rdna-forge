use forge_core::{ForgeError, ForgeResult};

pub const GEMM_F32_WGSL: &str = include_str!("../shaders/gemm_f32.wgsl");
pub const GEMM_F32_TILED_VEC4_WGSL: &str = include_str!("../shaders/gemm_f32_tiled_vec4.wgsl");
pub const GEMM_I8_DP4A_WGSL: &str = include_str!("../shaders/gemm_i8_dp4a.wgsl");
pub const GEMV_W8A32_WAVE32_WGSL: &str = include_str!("../shaders/gemv_w8a32_wave32.wgsl");
pub const GEMV_Q4_0_WAVE32_WGSL: &str = include_str!("../shaders/gemv_q4_0_wave32.wgsl");
pub const GEMV_Q4_K_WAVE32_WGSL: &str = include_str!("../shaders/gemv_q4_k_wave32.wgsl");
pub const RMSNORM_F32_WGSL: &str = include_str!("../shaders/rmsnorm_f32.wgsl");
pub const ROPE_F32_WGSL: &str = include_str!("../shaders/rope_f32.wgsl");
pub const RMSNORM_ROPE_F32_SUBGROUP_WGSL: &str =
    include_str!("../shaders/rmsnorm_rope_f32_subgroup.wgsl");
pub const FLASH_ATTENTION_F32_WAVE32_WGSL: &str =
    include_str!("../shaders/flash_attention_f32_wave32.wgsl");

/// SPIR-V opcode for `OpSDot` (`SPV_KHR_integer_dot_product` / SPIR-V 1.6).
pub const SPIRV_OP_SDOT: u16 = 4450;

/// SPIR-V opcode for `OpGroupNonUniformFAdd` (`GroupNonUniformArithmetic`).
pub const SPIRV_OP_GROUP_NON_UNIFORM_FADD: u16 = 350;

/// Compiles a WGSL compute shader into SPIR-V 1.3 words in pure Rust via `naga`.
pub fn compile_wgsl_to_spirv(source: &str) -> ForgeResult<Vec<u32>> {
    compile_wgsl_with_options(source, (1, 3), naga::valid::Capabilities::empty())
}

/// Compiles a WGSL compute shader using subgroup operations into SPIR-V 1.3 words via `naga`.
pub fn compile_wgsl_to_spirv_subgroup(source: &str) -> ForgeResult<Vec<u32>> {
    compile_wgsl_with_options(source, (1, 3), naga::valid::Capabilities::SUBGROUP)
}

/// Compiles a WGSL compute shader into SPIR-V 1.6 words enabling integer dot-product instructions (`OpSDot`).
pub fn compile_wgsl_to_spirv_dp4a(source: &str) -> ForgeResult<Vec<u32>> {
    compile_wgsl_with_options(source, (1, 6), naga::valid::Capabilities::empty())
}

fn compile_wgsl_with_options(
    source: &str,
    lang_version: (u8, u8),
    capabilities: naga::valid::Capabilities,
) -> ForgeResult<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|err| ForgeError::ExecutionFailed(err.emit_to_string(source)))?;

    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities);
    if capabilities.contains(naga::valid::Capabilities::SUBGROUP) {
        validator.subgroup_stages(naga::valid::ShaderStages::COMPUTE);
        validator.subgroup_operations(
            naga::valid::SubgroupOperationSet::BASIC
                | naga::valid::SubgroupOperationSet::ARITHMETIC,
        );
    }
    let info = validator.validate(&module).map_err(|err| {
        ForgeError::ExecutionFailed(format!(
            "WGSL validation failed:\n{}",
            err.emit_to_string(source)
        ))
    })?;

    let options = naga::back::spv::Options {
        lang_version,
        flags: naga::back::spv::WriterFlags::empty(),
        fake_missing_bindings: true,
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        emit_int_div_checks: false,
        ..Default::default()
    };

    naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|err| ForgeError::ExecutionFailed(format!("SPIR-V codegen failed: {err}")))
}

/// Scans a SPIR-V word stream and reports whether `target_opcode` appears in any instruction.
pub fn spirv_contains_opcode(words: &[u32], target_opcode: u16) -> bool {
    // SPIR-V header is 5 words: magic, version, generator, bound, schema.
    if words.len() <= 5 || words[0] != 0x0723_0203 {
        return false;
    }
    let mut idx = 5usize;
    while idx < words.len() {
        let word = words[idx];
        let wc = (word >> 16) as usize;
        let op = (word & 0xffff) as u16;
        if op == target_opcode {
            return true;
        }
        if wc == 0 {
            return false;
        }
        idx += wc;
    }
    false
}
