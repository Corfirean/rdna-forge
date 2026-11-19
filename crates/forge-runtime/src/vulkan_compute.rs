use std::ffi::CString;
use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use forge_core::{
    validate_kernel_execution, ForgeError, ForgeResult, GpuDeviceInfo, KernelRequirements,
};
use forge_device::discover_gpus;
use serde::{Deserialize, Serialize};

use crate::shaders::{
    compile_wgsl_to_spirv, compile_wgsl_to_spirv_dp4a, GEMM_F32_TILED_VEC4_WGSL, GEMM_F32_WGSL,
    GEMM_I8_DP4A_WGSL, RMSNORM_F32_WGSL, ROPE_F32_WGSL,
};

/// Timing measurements from a single Vulkan compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpuTiming {
    pub gpu_ms: Option<f64>,
    pub wall_ms: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
}

struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    query_pool: Option<vk::QueryPool>,
    timestamp_period_ns: f32,
    int8_dot_product_enabled: bool,
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(qp) = self.query_pool {
                self.device.destroy_query_pool(qp, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct GpuBuffer {
    ctx: Arc<VulkanContext>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.buffer, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

struct ComputePipeline {
    ctx: Arc<VulkanContext>,
    shader_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.ctx
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

/// Safe RAII Vulkan compute executor with hardware capability validation and timestamp profiling.
pub struct VulkanComputeEngine {
    ctx: Arc<VulkanContext>,
    device_info: GpuDeviceInfo,
}

impl VulkanComputeEngine {
    /// Creates a Vulkan compute engine bound to a discovered device index (default `0`, sorted preferred first).
    pub fn new(device_index: usize) -> ForgeResult<Self> {
        let gpus = discover_gpus();
        let target_info = gpus.get(device_index).cloned().ok_or_else(|| {
            ForgeError::DeviceNotFound(format!("GPU device index {device_index} not found"))
        })?;

        let entry = unsafe { ash::Entry::load() }.map_err(|e| ForgeError::RuntimeUnavailable {
            runtime: "Vulkan".to_string(),
            reason: format!("failed to load vulkan-1.dll: {e}"),
        })?;

        let raw_ver = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        let api_version = if raw_ver >= vk::API_VERSION_1_3 {
            vk::API_VERSION_1_3
        } else if raw_ver >= vk::API_VERSION_1_2 {
            vk::API_VERSION_1_2
        } else {
            vk::API_VERSION_1_1
        };

        let app_name = CString::new("RDNA Forge").expect("static string has no NUL");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(api_version);
        let inst_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe { entry.create_instance(&inst_info, None) }.map_err(|e| {
            ForgeError::RuntimeUnavailable {
                runtime: "Vulkan".to_string(),
                reason: format!("vkCreateInstance failed: {e}"),
            }
        })?;

        let pds = match unsafe { instance.enumerate_physical_devices() } {
            Ok(list) => list,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(ForgeError::RuntimeUnavailable {
                    runtime: "Vulkan".to_string(),
                    reason: format!("vkEnumeratePhysicalDevices failed: {e}"),
                });
            }
        };

        let pd = pds
            .into_iter()
            .find(|&candidate| {
                let p = unsafe { instance.get_physical_device_properties(candidate) };
                let name = unsafe { std::ffi::CStr::from_ptr(p.device_name.as_ptr()) }
                    .to_string_lossy()
                    .trim()
                    .to_string();
                p.vendor_id == target_info.vendor_id
                    && p.device_id == target_info.device_id
                    && name == target_info.name
            })
            .ok_or_else(|| {
                unsafe { instance.destroy_instance(None) };
                ForgeError::DeviceNotFound(format!(
                    "Vulkan physical device '{}' (0x{:04x}:0x{:04x}) not found",
                    target_info.name, target_info.vendor_id, target_info.device_id
                ))
            })?;

        let props = unsafe { instance.get_physical_device_properties(pd) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };

        let (qf_index, qf_props) = queue_families
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| {
                unsafe { instance.destroy_instance(None) };
                ForgeError::UnsupportedCapability(format!(
                    "device '{}' exposes no Vulkan compute queue family",
                    target_info.name
                ))
            })?;

        let has_vk13 = props.api_version >= vk::API_VERSION_1_3;
        let dp4a_supported = if has_vk13 {
            let mut vk13_props = vk::PhysicalDeviceVulkan13Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut vk13_props);
            unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

            let mut vk13_feat = vk::PhysicalDeviceVulkan13Features::default();
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut vk13_feat);
            unsafe { instance.get_physical_device_features2(pd, &mut feat2) };

            vk13_feat.shader_integer_dot_product == vk::TRUE
                && vk13_props.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE
        } else {
            false
        };

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_index as u32)
            .queue_priorities(&queue_priorities);
        let mut enabled_vk13_feat = vk::PhysicalDeviceVulkan13Features::default()
            .shader_integer_dot_product(dp4a_supported);
        let mut device_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
        if dp4a_supported {
            device_info = device_info.push_next(&mut enabled_vk13_feat);
        }

        let device = match unsafe { instance.create_device(pd, &device_info, None) } {
            Ok(dev) => dev,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(ForgeError::ExecutionFailed(format!(
                    "vkCreateDevice failed: {e}"
                )));
            }
        };

        let queue = unsafe { device.get_device_queue(qf_index as u32, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_index as u32)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(e) => {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(ForgeError::ExecutionFailed(format!(
                    "vkCreateCommandPool failed: {e}"
                )));
            }
        };

        let query_pool = if qf_props.timestamp_valid_bits > 0 && props.limits.timestamp_period > 0.0
        {
            let qp_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            unsafe { device.create_query_pool(&qp_info, None) }.ok()
        } else {
            None
        };

        let ctx = Arc::new(VulkanContext {
            _entry: entry,
            instance,
            device,
            mem_props,
            queue,
            command_pool,
            query_pool,
            timestamp_period_ns: props.limits.timestamp_period,
            int8_dot_product_enabled: dp4a_supported,
        });

        Ok(Self {
            ctx,
            device_info: target_info,
        })
    }

    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    pub fn supports_int8_dp4a(&self) -> bool {
        self.ctx.int8_dot_product_enabled
    }

    /// Executes baseline `16x16` FP32 `C[m, n] = A[m, k] * B[k, n]` once and returns output plus timing.
    pub fn run_gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> ForgeResult<(Vec<f32>, GpuTiming)> {
        let (mut out, mut timings) = self.bench_gemm_f32(a, b, m, k, n, 0, 1)?;
        let timing = timings.pop().ok_or_else(|| {
            ForgeError::ExecutionFailed("missing GEMM timing measurement".to_string())
        })?;
        Ok((std::mem::take(&mut out), timing))
    }

    /// Runs `warmup_iters` + `measured_iters` of baseline `16x16` tiled FP32 GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> ForgeResult<(Vec<f32>, Vec<GpuTiming>)> {
        self.bench_gemm_f32_with_shader(
            a,
            b,
            m,
            k,
            n,
            warmup_iters,
            measured_iters,
            GEMM_F32_WGSL,
            16,
            16,
        )
    }

    /// Executes register-blocked `128x64` FP32 `C[m, n] = A[m, k] * B[k, n]` once and returns output plus timing.
    pub fn run_gemm_f32_tiled_vec4(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> ForgeResult<(Vec<f32>, GpuTiming)> {
        let (mut out, mut timings) = self.bench_gemm_f32_tiled_vec4(a, b, m, k, n, 0, 1)?;
        let timing = timings.pop().ok_or_else(|| {
            ForgeError::ExecutionFailed("missing tiled GEMM timing measurement".to_string())
        })?;
        Ok((std::mem::take(&mut out), timing))
    }

    /// Runs `warmup_iters` + `measured_iters` of register-blocked `128x64` FP32 GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_gemm_f32_tiled_vec4(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> ForgeResult<(Vec<f32>, Vec<GpuTiming>)> {
        self.bench_gemm_f32_with_shader(
            a,
            b,
            m,
            k,
            n,
            warmup_iters,
            measured_iters,
            GEMM_F32_TILED_VEC4_WGSL,
            64,
            128,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bench_gemm_f32_with_shader(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
        warmup_iters: usize,
        measured_iters: usize,
        shader_src: &str,
        tile_n: u32,
        tile_m: u32,
    ) -> ForgeResult<(Vec<f32>, Vec<GpuTiming>)> {
        if m == 0 || k == 0 || n == 0 {
            return Err(ForgeError::InvalidDimension(
                "GEMM dimensions m, k, n must all be > 0".to_string(),
            ));
        }
        if measured_iters == 0 {
            return Err(ForgeError::InvalidDimension(
                "measured_iters must be > 0".to_string(),
            ));
        }

        let elems_a = (m as usize)
            .checked_mul(k as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("m * k overflow".to_string()))?;
        let elems_b = (k as usize)
            .checked_mul(n as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("k * n overflow".to_string()))?;
        let elems_c = (m as usize)
            .checked_mul(n as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("m * n overflow".to_string()))?;

        if a.len() != elems_a || b.len() != elems_b {
            return Err(ForgeError::InvalidDimension(format!(
                "GEMM input mismatch: got a={}, b={}, expected a={elems_a}, b={elems_b}",
                a.len(),
                b.len()
            )));
        }

        let bytes_a = (elems_a as u64) * 4;
        let bytes_b = (elems_b as u64) * 4;
        let bytes_c = (elems_c as u64) * 4;
        let max_buf = bytes_a.max(bytes_b).max(bytes_c);
        let total_vram = bytes_a.saturating_add(bytes_b).saturating_add(bytes_c);

        validate_kernel_execution(
            &self.device_info,
            &KernelRequirements {
                required_vram_bytes: total_vram,
                max_single_buffer_bytes: max_buf,
                ..Default::default()
            },
        )?;

        let spirv = compile_wgsl_to_spirv(shader_src)?;
        let pipeline = self.create_compute_pipeline(&spirv, 3, true)?;

        let buf_a = self.upload_device_local_slice(a)?;
        let buf_b = self.upload_device_local_slice(b)?;
        let buf_c = self.create_device_local_buffer(bytes_c)?;
        let params = GemmParams { m, k, n, _pad: 0 };
        let ubo = self.create_uniform_buffer(&params)?;

        self.bind_buffers(&pipeline, &[&buf_a, &buf_b, &buf_c], Some(&ubo));

        let gx = n.div_ceil(tile_n);
        let gy = m.div_ceil(tile_m);
        let timings =
            self.execute_timed_loop(&pipeline, gx, gy, 1, warmup_iters, measured_iters)?;
        let out = self.download_device_local_vec::<f32>(&buf_c, elems_c)?;

        Ok((out, timings))
    }

    /// Executes DP4A INT8 GEMM (`OpSDot`) once and returns exact `i32` accumulators plus timing.
    pub fn run_gemm_i8_dp4a(
        &self,
        a_packed: &[u32],
        b_t_packed: &[u32],
        m: u32,
        k_packed: u32,
        n: u32,
    ) -> ForgeResult<(Vec<i32>, GpuTiming)> {
        let (mut out, mut timings) =
            self.bench_gemm_i8_dp4a(a_packed, b_t_packed, m, k_packed, n, 0, 1)?;
        let timing = timings.pop().ok_or_else(|| {
            ForgeError::ExecutionFailed("missing INT8 DP4A GEMM timing measurement".to_string())
        })?;
        Ok((std::mem::take(&mut out), timing))
    }

    /// Benchmarks DP4A INT8 GEMM (`OpSDot` / `V_DOT4_I32_I8`) on device-local buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_gemm_i8_dp4a(
        &self,
        a_packed: &[u32],
        b_t_packed: &[u32],
        m: u32,
        k_packed: u32,
        n: u32,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> ForgeResult<(Vec<i32>, Vec<GpuTiming>)> {
        if !self.ctx.int8_dot_product_enabled {
            return Err(ForgeError::UnsupportedCapability(format!(
                "Vulkan 1.3 shader_integer_dot_product (packed 4x8 signed) is not supported on '{}'",
                self.device_info.name
            )));
        }
        if m == 0 || k_packed == 0 || n == 0 {
            return Err(ForgeError::InvalidDimension(
                "INT8 GEMM dimensions m, k_packed, n must all be > 0".to_string(),
            ));
        }
        if measured_iters == 0 {
            return Err(ForgeError::InvalidDimension(
                "measured_iters must be > 0".to_string(),
            ));
        }

        let elems_a = (m as usize)
            .checked_mul(k_packed as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("m * k_packed overflow".to_string()))?;
        let elems_b = (n as usize)
            .checked_mul(k_packed as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("n * k_packed overflow".to_string()))?;
        let elems_c = (m as usize)
            .checked_mul(n as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("m * n overflow".to_string()))?;

        if a_packed.len() != elems_a || b_t_packed.len() != elems_b {
            return Err(ForgeError::InvalidDimension(format!(
                "INT8 DP4A input mismatch: got a_packed={}, b_t_packed={}, expected ({elems_a}, {elems_b})",
                a_packed.len(),
                b_t_packed.len()
            )));
        }

        let bytes_a = (elems_a as u64) * 4;
        let bytes_b = (elems_b as u64) * 4;
        let bytes_c = (elems_c as u64) * 4;
        let max_buf = bytes_a.max(bytes_b).max(bytes_c);
        let total_vram = bytes_a.saturating_add(bytes_b).saturating_add(bytes_c);

        validate_kernel_execution(
            &self.device_info,
            &KernelRequirements {
                required_vram_bytes: total_vram,
                max_single_buffer_bytes: max_buf,
                require_int8_dp4a: true,
                ..Default::default()
            },
        )?;

        let spirv = compile_wgsl_to_spirv_dp4a(GEMM_I8_DP4A_WGSL)?;
        let pipeline = self.create_compute_pipeline(&spirv, 3, true)?;

        let buf_a = self.upload_device_local_slice(a_packed)?;
        let buf_b = self.upload_device_local_slice(b_t_packed)?;
        let buf_c = self.create_device_local_buffer(bytes_c)?;
        let params = GemmParams {
            m,
            k: k_packed,
            n,
            _pad: 0,
        };
        let ubo = self.create_uniform_buffer(&params)?;

        self.bind_buffers(&pipeline, &[&buf_a, &buf_b, &buf_c], Some(&ubo));

        let gx = n.div_ceil(64);
        let gy = m.div_ceil(64);
        let timings =
            self.execute_timed_loop(&pipeline, gx, gy, 1, warmup_iters, measured_iters)?;
        let out = self.download_device_local_vec::<i32>(&buf_c, elems_c)?;

        Ok((out, timings))
    }

    /// Executes row-wise RMSNorm once and returns output plus timing.
    pub fn run_rmsnorm_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> ForgeResult<(Vec<f32>, GpuTiming)> {
        let (out, mut timings) = self.bench_rmsnorm_f32(x, weight, rows, cols, eps, 0, 1)?;
        let timing = timings.pop().ok_or_else(|| {
            ForgeError::ExecutionFailed("missing RMSNorm timing measurement".to_string())
        })?;
        Ok((out, timing))
    }

    /// Benchmarks row-wise RMSNorm on device-local buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_rmsnorm_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        rows: u32,
        cols: u32,
        eps: f32,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> ForgeResult<(Vec<f32>, Vec<GpuTiming>)> {
        if rows == 0 || cols == 0 {
            return Err(ForgeError::InvalidDimension(
                "RMSNorm rows and cols must be > 0".to_string(),
            ));
        }
        if measured_iters == 0 {
            return Err(ForgeError::InvalidDimension(
                "measured_iters must be > 0".to_string(),
            ));
        }
        let elems = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or_else(|| ForgeError::InvalidDimension("rows * cols overflow".to_string()))?;
        if x.len() != elems || weight.len() != cols as usize {
            return Err(ForgeError::InvalidDimension(format!(
                "RMSNorm input mismatch: got x={}, w={}, expected x={elems}, w={cols}",
                x.len(),
                weight.len()
            )));
        }

        let bytes_x = (elems as u64) * 4;
        let bytes_w = (cols as u64) * 4;
        validate_kernel_execution(
            &self.device_info,
            &KernelRequirements {
                required_vram_bytes: bytes_x.saturating_mul(2).saturating_add(bytes_w),
                max_single_buffer_bytes: bytes_x.max(bytes_w),
                ..Default::default()
            },
        )?;

        let spirv = compile_wgsl_to_spirv(RMSNORM_F32_WGSL)?;
        let pipeline = self.create_compute_pipeline(&spirv, 3, true)?;

        let buf_x = self.upload_device_local_slice(x)?;
        let buf_w = self.upload_device_local_slice(weight)?;
        let buf_y = self.create_device_local_buffer(bytes_x)?;
        let params = RmsNormParams {
            rows,
            cols,
            eps,
            _pad: 0,
        };
        let ubo = self.create_uniform_buffer(&params)?;

        self.bind_buffers(&pipeline, &[&buf_x, &buf_w, &buf_y], Some(&ubo));

        let timings =
            self.execute_timed_loop(&pipeline, rows, 1, 1, warmup_iters, measured_iters)?;
        let out = self.download_device_local_vec::<f32>(&buf_y, elems)?;

        Ok((out, timings))
    }

    /// Executes RoPE once and returns output plus timing.
    pub fn run_rope_f32(
        &self,
        input: &[f32],
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta_base: f32,
    ) -> ForgeResult<(Vec<f32>, GpuTiming)> {
        let (out, mut timings) =
            self.bench_rope_f32(input, seq_len, n_heads, head_dim, theta_base, 0, 1)?;
        let timing = timings.pop().ok_or_else(|| {
            ForgeError::ExecutionFailed("missing RoPE timing measurement".to_string())
        })?;
        Ok((out, timing))
    }

    /// Benchmarks RoPE on device-local buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_rope_f32(
        &self,
        input: &[f32],
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta_base: f32,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> ForgeResult<(Vec<f32>, Vec<GpuTiming>)> {
        if seq_len == 0 || n_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(ForgeError::InvalidDimension(format!(
                "RoPE requires non-zero dimensions and even head_dim, got ({seq_len}, {n_heads}, {head_dim})"
            )));
        }
        if theta_base <= 0.0 || !theta_base.is_finite() {
            return Err(ForgeError::InvalidDimension(format!(
                "RoPE theta_base must be positive and finite, got {theta_base}"
            )));
        }
        if measured_iters == 0 {
            return Err(ForgeError::InvalidDimension(
                "measured_iters must be > 0".to_string(),
            ));
        }

        let elems = (seq_len as usize)
            .checked_mul(n_heads as usize)
            .and_then(|v| v.checked_mul(head_dim as usize))
            .ok_or_else(|| ForgeError::InvalidDimension("RoPE tensor size overflow".to_string()))?;
        if input.len() != elems {
            return Err(ForgeError::InvalidDimension(format!(
                "RoPE input length mismatch: got {}, expected {elems}",
                input.len()
            )));
        }

        let bytes = (elems as u64) * 4;
        validate_kernel_execution(
            &self.device_info,
            &KernelRequirements {
                required_vram_bytes: bytes.saturating_mul(2),
                max_single_buffer_bytes: bytes,
                ..Default::default()
            },
        )?;

        let spirv = compile_wgsl_to_spirv(ROPE_F32_WGSL)?;
        let pipeline = self.create_compute_pipeline(&spirv, 2, true)?;

        let buf_in = self.upload_device_local_slice(input)?;
        let buf_out = self.create_device_local_buffer(bytes)?;
        let params = RopeParams {
            seq_len,
            n_heads,
            head_dim,
            theta_base,
        };
        let ubo = self.create_uniform_buffer(&params)?;

        self.bind_buffers(&pipeline, &[&buf_in, &buf_out], Some(&ubo));

        let total_pairs = seq_len
            .checked_mul(n_heads)
            .and_then(|v| v.checked_mul(head_dim / 2))
            .ok_or_else(|| ForgeError::InvalidDimension("RoPE pair count overflow".to_string()))?;
        let gx = total_pairs.div_ceil(256);

        let timings = self.execute_timed_loop(&pipeline, gx, 1, 1, warmup_iters, measured_iters)?;
        let out = self.download_device_local_vec::<f32>(&buf_out, elems)?;

        Ok((out, timings))
    }

    fn create_compute_pipeline(
        &self,
        spirv: &[u32],
        storage_count: u32,
        has_uniform: bool,
    ) -> ForgeResult<ComputePipeline> {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = unsafe { self.ctx.device.create_shader_module(&shader_info, None) }
            .map_err(|e| ForgeError::ExecutionFailed(format!("create_shader_module: {e}")))?;

        let mut bindings = Vec::new();
        for idx in 0..storage_count {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(idx)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        if has_uniform {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(storage_count)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.ctx
                .device
                .create_descriptor_set_layout(&dsl_info, None)
        }
        .map_err(|e| {
            unsafe { self.ctx.device.destroy_shader_module(shader_module, None) };
            ForgeError::ExecutionFailed(format!("create_descriptor_set_layout: {e}"))
        })?;

        let set_layouts = [descriptor_set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { self.ctx.device.create_pipeline_layout(&pl_info, None) }
            .map_err(|e| {
                unsafe {
                    self.ctx
                        .device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    self.ctx.device.destroy_shader_module(shader_module, None);
                }
                ForgeError::ExecutionFailed(format!("create_pipeline_layout: {e}"))
            })?;

        let entry_name = CString::new("main").expect("static string has no NUL");
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);

        let cp_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.ctx.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&cp_info),
                None,
            )
        }
        .map_err(|(_, e)| {
            unsafe {
                self.ctx
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
                self.ctx
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                self.ctx.device.destroy_shader_module(shader_module, None);
            }
            ForgeError::ExecutionFailed(format!("create_compute_pipelines: {e}"))
        })?[0];

        let mut pool_sizes = vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: storage_count,
        }];
        if has_uniform {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            });
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { self.ctx.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| {
            unsafe {
                self.ctx.device.destroy_pipeline(pipeline, None);
                self.ctx
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
                self.ctx
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                self.ctx.device.destroy_shader_module(shader_module, None);
            }
            ForgeError::ExecutionFailed(format!("create_descriptor_pool: {e}"))
        })?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { self.ctx.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| {
                unsafe {
                    self.ctx
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None);
                    self.ctx.device.destroy_pipeline(pipeline, None);
                    self.ctx
                        .device
                        .destroy_pipeline_layout(pipeline_layout, None);
                    self.ctx
                        .device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    self.ctx.device.destroy_shader_module(shader_module, None);
                }
                ForgeError::ExecutionFailed(format!("allocate_descriptor_sets: {e}"))
            })?[0];

        Ok(ComputePipeline {
            ctx: Arc::clone(&self.ctx),
            shader_module,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_set,
        })
    }

    fn bind_buffers(
        &self,
        pipeline: &ComputePipeline,
        storage_bufs: &[&GpuBuffer],
        uniform_buf: Option<&GpuBuffer>,
    ) {
        let storage_infos: Vec<vk::DescriptorBufferInfo> = storage_bufs
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(b.size)
            })
            .collect();

        let mut writes = Vec::new();
        for (i, info) in storage_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(pipeline.descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }

        let ubo_info = uniform_buf.map(|u| {
            vk::DescriptorBufferInfo::default()
                .buffer(u.buffer)
                .offset(0)
                .range(u.size)
        });
        if let Some(ref info) = ubo_info {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(pipeline.descriptor_set)
                    .dst_binding(storage_bufs.len() as u32)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }

        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
    }

    fn execute_timed_loop(
        &self,
        pipeline: &ComputePipeline,
        gx: u32,
        gy: u32,
        gz: u32,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> ForgeResult<Vec<GpuTiming>> {
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.ctx.device.allocate_command_buffers(&cmd_alloc) }
            .map_err(|e| ForgeError::ExecutionFailed(format!("allocate_command_buffers: {e}")))?[0];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.ctx.device.create_fence(&fence_info, None) }
            .map_err(|e| ForgeError::ExecutionFailed(format!("create_fence: {e}")))?;

        let total = warmup_iters + measured_iters;
        let mut timings = Vec::with_capacity(measured_iters);

        let result = (|| -> ForgeResult<()> {
            for iter in 0..total {
                unsafe {
                    self.ctx
                        .device
                        .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                        .map_err(|e| ForgeError::ExecutionFailed(format!("reset_cmd: {e}")))?;

                    let begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    self.ctx
                        .device
                        .begin_command_buffer(cmd, &begin_info)
                        .map_err(|e| ForgeError::ExecutionFailed(format!("begin_cmd: {e}")))?;

                    if let Some(qp) = self.ctx.query_pool {
                        self.ctx.device.cmd_reset_query_pool(cmd, qp, 0, 2);
                        self.ctx.device.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            qp,
                            0,
                        );
                    }

                    self.ctx.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.pipeline,
                    );
                    self.ctx.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.pipeline_layout,
                        0,
                        std::slice::from_ref(&pipeline.descriptor_set),
                        &[],
                    );
                    self.ctx.device.cmd_dispatch(cmd, gx, gy, gz);

                    if let Some(qp) = self.ctx.query_pool {
                        self.ctx.device.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            qp,
                            1,
                        );
                    }

                    self.ctx
                        .device
                        .end_command_buffer(cmd)
                        .map_err(|e| ForgeError::ExecutionFailed(format!("end_cmd: {e}")))?;
                }

                let wall_start = Instant::now();
                let submit_info =
                    vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
                unsafe {
                    self.ctx
                        .device
                        .reset_fences(std::slice::from_ref(&fence))
                        .map_err(|e| ForgeError::ExecutionFailed(format!("reset_fences: {e}")))?;
                    self.ctx
                        .device
                        .queue_submit(self.ctx.queue, std::slice::from_ref(&submit_info), fence)
                        .map_err(|e| ForgeError::ExecutionFailed(format!("queue_submit: {e}")))?;
                    self.ctx
                        .device
                        .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                        .map_err(|e| {
                            ForgeError::ExecutionFailed(format!("wait_for_fences: {e}"))
                        })?;
                }
                let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

                if iter >= warmup_iters {
                    let gpu_ms = self.ctx.query_pool.and_then(|qp| {
                        let mut timestamps = [0u64; 2];
                        let rc = unsafe {
                            self.ctx.device.get_query_pool_results(
                                qp,
                                0,
                                &mut timestamps,
                                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                            )
                        };
                        if rc.is_ok() && timestamps[1] >= timestamps[0] {
                            let ticks = timestamps[1] - timestamps[0];
                            let ns = (ticks as f64) * f64::from(self.ctx.timestamp_period_ns);
                            Some(ns / 1_000_000.0)
                        } else {
                            None
                        }
                    });
                    timings.push(GpuTiming { gpu_ms, wall_ms });
                }
            }
            Ok(())
        })();

        unsafe {
            self.ctx.device.destroy_fence(fence, None);
            self.ctx
                .device
                .free_command_buffers(self.ctx.command_pool, std::slice::from_ref(&cmd));
        }

        result?;
        Ok(timings)
    }

    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        mem_flags: vk::MemoryPropertyFlags,
    ) -> ForgeResult<GpuBuffer> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.ctx.device.create_buffer(&info, None) }
            .map_err(|e| ForgeError::ExecutionFailed(format!("create_buffer: {e}")))?;

        let req = unsafe { self.ctx.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self
            .find_memory_type(req.memory_type_bits, mem_flags)
            .ok_or_else(|| {
                unsafe { self.ctx.device.destroy_buffer(buffer, None) };
                ForgeError::ExecutionFailed(format!(
                    "no suitable Vulkan memory type for flags {mem_flags:?}"
                ))
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);

        let memory =
            unsafe { self.ctx.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
                unsafe { self.ctx.device.destroy_buffer(buffer, None) };
                ForgeError::ExecutionFailed(format!("allocate_memory: {e}"))
            })?;

        if let Err(e) = unsafe { self.ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.ctx.device.free_memory(memory, None);
                self.ctx.device.destroy_buffer(buffer, None);
            }
            return Err(ForgeError::ExecutionFailed(format!(
                "bind_buffer_memory: {e}"
            )));
        }

        Ok(GpuBuffer {
            ctx: Arc::clone(&self.ctx),
            buffer,
            memory,
            size,
        })
    }

    fn find_memory_type(&self, type_filter: u32, required: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.ctx.mem_props.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && self.ctx.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(required)
        })
    }

    fn create_device_local_buffer(&self, size: u64) -> ForgeResult<GpuBuffer> {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    fn create_uniform_buffer<T: Copy>(&self, value: &T) -> ForgeResult<GpuBuffer> {
        let size = std::mem::size_of::<T>() as u64;
        let buf = self.create_buffer(
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(buf.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| ForgeError::ExecutionFailed(format!("map_memory(ubo): {e}")))?;
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(value).cast::<u8>(),
                ptr.cast::<u8>(),
                size as usize,
            );
            self.ctx.device.unmap_memory(buf.memory);
        }
        Ok(buf)
    }

    fn upload_device_local_slice<T: Copy>(&self, data: &[T]) -> ForgeResult<GpuBuffer> {
        let size = (std::mem::size_of_val(data)) as u64;
        let staging = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| ForgeError::ExecutionFailed(format!("map_memory(upload): {e}")))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<T>(), data.len());
            self.ctx.device.unmap_memory(staging.memory);
        }

        let dev_buf = self.create_device_local_buffer(size)?;
        self.copy_buffer(&staging, &dev_buf, size)?;
        Ok(dev_buf)
    }

    fn download_device_local_vec<T: Copy + Default>(
        &self,
        dev_buf: &GpuBuffer,
        len: usize,
    ) -> ForgeResult<Vec<T>> {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let staging = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.copy_buffer(dev_buf, &staging, size)?;

        let mut out = vec![T::default(); len];
        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| ForgeError::ExecutionFailed(format!("map_memory(download): {e}")))?;
            std::ptr::copy_nonoverlapping(ptr.cast::<T>(), out.as_mut_ptr(), len);
            self.ctx.device.unmap_memory(staging.memory);
        }
        Ok(out)
    }

    fn copy_buffer(&self, src: &GpuBuffer, dst: &GpuBuffer, size: u64) -> ForgeResult<()> {
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.ctx.device.allocate_command_buffers(&cmd_alloc) }
            .map_err(|e| ForgeError::ExecutionFailed(format!("allocate_command_buffers: {e}")))?[0];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.ctx.device.create_fence(&fence_info, None) }
            .map_err(|e| ForgeError::ExecutionFailed(format!("create_fence: {e}")))?;

        let result = (|| -> ForgeResult<()> {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let region = vk::BufferCopy::default().size(size);
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

            unsafe {
                self.ctx
                    .device
                    .begin_command_buffer(cmd, &begin_info)
                    .map_err(|e| ForgeError::ExecutionFailed(format!("begin_copy: {e}")))?;
                self.ctx.device.cmd_copy_buffer(
                    cmd,
                    src.buffer,
                    dst.buffer,
                    std::slice::from_ref(&region),
                );
                self.ctx
                    .device
                    .end_command_buffer(cmd)
                    .map_err(|e| ForgeError::ExecutionFailed(format!("end_copy: {e}")))?;
                self.ctx
                    .device
                    .queue_submit(self.ctx.queue, std::slice::from_ref(&submit_info), fence)
                    .map_err(|e| ForgeError::ExecutionFailed(format!("submit_copy: {e}")))?;
                self.ctx
                    .device
                    .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                    .map_err(|e| ForgeError::ExecutionFailed(format!("wait_copy: {e}")))?;
            }
            Ok(())
        })();

        unsafe {
            self.ctx.device.destroy_fence(fence, None);
            self.ctx
                .device
                .free_command_buffers(self.ctx.command_pool, std::slice::from_ref(&cmd));
        }

        result
    }
}
