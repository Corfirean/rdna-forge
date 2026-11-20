use std::path::Path;
use std::process::Command;

use forge_core::{BackendKind, BackendStatus, GpuArchitecture, SupportStatus};
use forge_device::DiscoverySnapshot;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PythonProbePayload {
    python_exe: String,
    python_version: String,
    torch_version: Option<String>,
    torch_hip_version: Option<String>,
    torch_cuda_available: bool,
    device_name: Option<String>,
    triton_version: Option<String>,
    triton_windows_version: Option<String>,
    triton_amd_backend: bool,
    error: Option<String>,
}

/// Probes all relevant Windows compute, telemetry, media, and inference backends.
pub fn probe_all_backends(
    snapshot: &DiscoverySnapshot,
    python_override: Option<&Path>,
) -> (Vec<BackendStatus>, Vec<String>) {
    let mut backends = Vec::new();
    let mut missing = Vec::new();

    // 1. Vulkan
    if snapshot.vulkan.available && !snapshot.vulkan.devices.is_empty() {
        let compute_ready = snapshot
            .vulkan
            .devices
            .iter()
            .any(|d| d.compute_queue_supported);
        let version = snapshot.vulkan.instance_version.clone();
        let mut details = Vec::new();
        if let Some(d0) = snapshot.gpus.first() {
            if let Some(ref drv) = d0.vulkan_driver_info {
                details.push(format!("Driver: {drv}"));
            }
            details.push(format!(
                "FP16={}, INT8={}, CoopMatrix={}, ReBAR={}",
                d0.vulkan_fp16,
                d0.vulkan_int8,
                d0.vulkan_cooperative_matrix,
                d0.rebar_enabled
                    .map(|b| if b {
                        "enabled"
                    } else {
                        "disabled (256 MiB BAR)"
                    })
                    .unwrap_or("unknown")
            ));
        }
        backends.push(BackendStatus {
            kind: BackendKind::Vulkan,
            status: if compute_ready {
                SupportStatus::Verified
            } else {
                SupportStatus::PartiallyVerified
            },
            version,
            summary: if compute_ready {
                format!(
                    "Detected (Vulkan {}, compute queue ready)",
                    snapshot
                        .vulkan
                        .instance_version
                        .as_deref()
                        .unwrap_or("unknown")
                )
            } else {
                "Vulkan loader available, no compute queue found".to_string()
            },
            details,
        });
    } else {
        let err = snapshot
            .vulkan
            .error
            .clone()
            .unwrap_or_else(|| "vulkan-1.dll not available".to_string());
        missing.push(format!("Vulkan compute runtime: {err}"));
        backends.push(BackendStatus {
            kind: BackendKind::Vulkan,
            status: SupportStatus::Unsupported,
            version: None,
            summary: format!("Unavailable ({err})"),
            details: Vec::new(),
        });
    }

    // 2. HIP / ROCm
    let has_hipcc = command_exists("hipcc");
    let has_rocm_sdk = command_exists("rocm-sdk");
    if snapshot.hip.available && !snapshot.hip.devices.is_empty() {
        let rt_ver = snapshot
            .hip
            .runtime_version
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let lib_name = snapshot
            .hip
            .loaded_library
            .clone()
            .unwrap_or_else(|| "amdhip64".to_string());
        let mut details = vec![format!("Library: {lib_name}")];
        if has_rocm_sdk {
            details.push("TheRock rocm-sdk CLI found in PATH".to_string());
        } else {
            details.push(
                "Driver runtime DLL present; full ROCm SDK (rocm-sdk / hipcc) not in PATH"
                    .to_string(),
            );
            missing.push(
                "ROCm/TheRock SDK dev tools (rocm-sdk / hipcc) not in PATH; only Adrenalin HIP runtime DLL is active"
                    .to_string(),
            );
        }
        backends.push(BackendStatus {
            kind: BackendKind::Hip,
            status: if has_rocm_sdk || has_hipcc {
                SupportStatus::Verified
            } else {
                SupportStatus::PartiallyVerified
            },
            version: Some(rt_ver.clone()),
            summary: if has_rocm_sdk || has_hipcc {
                format!("Detected (HIP {rt_ver}, SDK toolchain present)")
            } else {
                format!("Driver runtime detected (HIP {rt_ver}; SDK toolchain absent)")
            },
            details,
        });
    } else {
        let err = snapshot
            .hip
            .error
            .clone()
            .unwrap_or_else(|| "amdhip64 DLL not found".to_string());
        missing.push(format!("AMD HIP runtime: {err}"));
        backends.push(BackendStatus {
            kind: BackendKind::Hip,
            status: SupportStatus::Unsupported,
            version: None,
            summary: format!("Unavailable ({err})"),
            details: Vec::new(),
        });
    }

    // 3. Triton (Windows)
    let primary_arch = snapshot.gpus.first().map(|g| g.architecture);
    let triton_status = probe_triton_python(python_override, primary_arch);
    if triton_status.status == SupportStatus::Unsupported {
        missing.push(format!("Triton: {}", triton_status.summary));
    }
    backends.push(triton_status);

    // 4. ADLX
    if snapshot.adlx.available {
        let ver = snapshot
            .adlx
            .full_version
            .clone()
            .or_else(|| snapshot.adlx.version.clone());
        backends.push(BackendStatus {
            kind: BackendKind::Adlx,
            status: SupportStatus::Verified,
            version: ver.clone(),
            summary: format!("Detected (ADLX {})", ver.as_deref().unwrap_or("unknown")),
            details: vec!["amdadlx64.dll dynamically loaded".to_string()],
        });
    } else {
        backends.push(BackendStatus {
            kind: BackendKind::Adlx,
            status: SupportStatus::Unsupported,
            version: None,
            summary: "Unavailable (amdadlx64.dll not loaded)".to_string(),
            details: Vec::new(),
        });
    }

    // 5. DirectML (Windows maintenance mode)
    let dml_path = system32_dll_path("DirectML.dll");
    backends.push(if dml_path.is_some() {
        BackendStatus {
            kind: BackendKind::DirectMl,
            status: SupportStatus::PartiallyVerified,
            version: None,
            summary: "Present (DirectML.dll; upstream in maintenance mode)".to_string(),
            details: vec![
                "Microsoft recommends Windows ML on Windows 11 24H2+; DirectML receives security fixes only"
                    .to_string(),
            ],
        }
    } else {
        BackendStatus {
            kind: BackendKind::DirectMl,
            status: SupportStatus::Unsupported,
            version: None,
            summary: "Unavailable (DirectML.dll not found)".to_string(),
            details: Vec::new(),
        }
    });

    // 6. AMF (Hardware video pipeline)
    backends.push(probe_amf_backend());

    // 7. llama.cpp CLI
    if let Some(path) = command_path("llama-bench") {
        backends.push(BackendStatus {
            kind: BackendKind::LlamaCpp,
            status: SupportStatus::Verified,
            version: None,
            summary: format!("llama-bench found ({})", path),
            details: Vec::new(),
        });
    } else {
        missing.push("llama-bench binary not found in PATH (pass --vulkan-bin / --hip-bin to `forge bench llama`)".to_string());
        backends.push(BackendStatus {
            kind: BackendKind::LlamaCpp,
            status: SupportStatus::NotTested,
            version: None,
            summary: "Not in PATH (external binary)".to_string(),
            details: Vec::new(),
        });
    }

    (backends, missing)
}

fn probe_amf_backend() -> BackendStatus {
    match crate::amf::probe_amf_capabilities() {
        Ok(report) => {
            let mut hw_encoders = Vec::new();
            let mut hw_decoders = Vec::new();
            let mut unsupported = Vec::new();

            for c in &report.codecs {
                match (c.direction, c.acceleration) {
                    (
                        crate::amf::AmfCodecDirection::Encode,
                        crate::amf::AmfAcceleration::Hardware,
                    ) => {
                        hw_encoders.push(c.codec.as_str());
                    }
                    (
                        crate::amf::AmfCodecDirection::Decode,
                        crate::amf::AmfAcceleration::Hardware,
                    ) => {
                        hw_decoders.push(c.codec.as_str());
                    }
                    (_, crate::amf::AmfAcceleration::NotSupported) => {
                        unsupported.push(format!("{} {}", c.codec, c.direction));
                    }
                    _ => {}
                }
            }

            let mut details = vec![format!(
                "amfrt64.dll DX11 probe OK; HW Encoders: [{}]",
                hw_encoders.join(", ")
            )];
            details.push(format!("HW Decoders: [{}]", hw_decoders.join(", ")));
            if !unsupported.is_empty() {
                details.push(format!("Unsupported: [{}]", unsupported.join(", ")));
            }

            BackendStatus {
                kind: BackendKind::Amf,
                status: SupportStatus::Verified,
                version: Some(report.runtime_version.clone()),
                summary: format!(
                    "Detected (AMF {}, {} HW encoders / {} HW decoders)",
                    report.runtime_version,
                    hw_encoders.len(),
                    hw_decoders.len()
                ),
                details,
            }
        }
        Err(err) => {
            if system32_dll_path("amfrt64.dll").is_some() {
                BackendStatus {
                    kind: BackendKind::Amf,
                    status: SupportStatus::PartiallyVerified,
                    version: None,
                    summary: "Detected (amfrt64.dll present; live capability probe failed)"
                        .to_string(),
                    details: vec![err.to_string()],
                }
            } else {
                BackendStatus {
                    kind: BackendKind::Amf,
                    status: SupportStatus::Unsupported,
                    version: None,
                    summary: "Unavailable (amfrt64.dll not found)".to_string(),
                    details: Vec::new(),
                }
            }
        }
    }
}

fn probe_triton_python(
    python_override: Option<&Path>,
    primary_arch: Option<GpuArchitecture>,
) -> BackendStatus {
    // Resolve through the shared validator: explicit paths must be files, bare names come from PATH
    // only. The path is kept as a `Path` (not stringified) so non-Unicode paths survive.
    let requested = python_override.unwrap_or_else(|| Path::new("python"));
    let python_cmd = match forge_core::resolve_python_executable(requested) {
        Ok(p) => p,
        Err(e) => {
            return BackendStatus {
                kind: BackendKind::Triton,
                status: SupportStatus::Unsupported,
                version: None,
                summary: format!("Python interpreter unavailable: {e}"),
                details: Vec::new(),
            };
        }
    };

    const PROBE_SCRIPT: &str = r#"
import importlib.metadata, json, sys
out = {
    "python_exe": sys.executable,
    "python_version": sys.version.split()[0],
    "torch_version": None,
    "torch_hip_version": None,
    "torch_cuda_available": False,
    "device_name": None,
    "triton_version": None,
    "triton_windows_version": None,
    "triton_amd_backend": False,
    "error": None,
}
try:
    import torch
    out["torch_version"] = getattr(torch, "__version__", None)
    out["torch_hip_version"] = getattr(torch.version, "hip", None)
    out["torch_cuda_available"] = bool(torch.cuda.is_available())
    if out["torch_cuda_available"]:
        out["device_name"] = torch.cuda.get_device_name(0)
except Exception as e:
    out["error"] = f"torch import error: {e}"
try:
    import triton
    out["triton_version"] = getattr(triton, "__version__", None)
    try:
        from triton.backends.amd import driver as _amd_driver
        out["triton_amd_backend"] = True
    except Exception:
        out["triton_amd_backend"] = False
except Exception as e:
    err = f"triton import error: {e}"
    out["error"] = f"{out['error']}; {err}" if out["error"] else err
try:
    out["triton_windows_version"] = importlib.metadata.version("triton-windows")
except Exception:
    pass
print(json.dumps(out))
"#;

    let output = Command::new(&python_cmd)
        .arg("-c")
        .arg(PROBE_SCRIPT)
        .output();

    let Ok(out) = output else {
        return BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: None,
            summary: format!(
                "Python executable '{}' could not be launched",
                python_cmd.display()
            ),
            details: Vec::new(),
        };
    };

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: None,
            summary: format!("Python probe failed: {stderr}"),
            details: Vec::new(),
        };
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let Ok(parsed) = serde_json::from_str::<PythonProbePayload>(stdout.trim()) else {
        return BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: None,
            summary: "Failed to parse Python/Triton probe JSON".to_string(),
            details: Vec::new(),
        };
    };

    let mut details = vec![format!(
        "Python {} ({})",
        parsed.python_version, parsed.python_exe
    )];
    if let Some(ref tv) = parsed.torch_version {
        details.push(format!(
            "PyTorch {tv}, HIP={}",
            parsed.torch_hip_version.as_deref().unwrap_or("none")
        ));
    }
    if let Some(ref dev) = parsed.device_name {
        details.push(format!("Torch GPU[0]: {dev}"));
    }
    if let Some(ref err) = parsed.error {
        details.push(err.clone());
    }

    match (
        parsed.triton_version.as_ref(),
        parsed.torch_hip_version.as_ref(),
        parsed.torch_cuda_available,
    ) {
        (Some(triton_ver), Some(hip_ver), true) => {
            let pkg_ver = parsed
                .triton_windows_version
                .clone()
                .unwrap_or_else(|| triton_ver.clone());
            let arch_note = match primary_arch {
                Some(arch @ (GpuArchitecture::Rdna1 | GpuArchitecture::Rdna2)) => {
                    format!("; community {arch} support")
                }
                Some(
                    arch @ (GpuArchitecture::Rdna3
                    | GpuArchitecture::Rdna35
                    | GpuArchitecture::Rdna4),
                ) => format!("; native {arch} support"),
                _ => String::new(),
            };
            BackendStatus {
                kind: BackendKind::Triton,
                status: if parsed.triton_amd_backend {
                    SupportStatus::PartiallyVerified
                } else {
                    SupportStatus::Experimental
                },
                version: Some(pkg_ver.clone()),
                summary: format!(
                    "Detected (triton-windows {pkg_ver}, PyTorch HIP {hip_ver}{arch_note})"
                ),
                details,
            }
        }
        (Some(triton_ver), None, true) => BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: Some(triton_ver.clone()),
            summary: "Triton installed, but active PyTorch build is not ROCm/HIP".to_string(),
            details,
        },
        (Some(triton_ver), _, false) => BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: Some(triton_ver.clone()),
            summary: "Triton installed, but PyTorch HIP/CUDA device is unavailable".to_string(),
            details,
        },
        (None, Some(hip_ver), true) => BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: None,
            summary: format!("PyTorch ROCm (HIP {hip_ver}) active, triton-windows not installed"),
            details,
        },
        (None, _, _) => BackendStatus {
            kind: BackendKind::Triton,
            status: SupportStatus::Unsupported,
            version: None,
            summary: "Not installed in selected Python environment".to_string(),
            details,
        },
    }
}

fn system32_dll_path(name: &str) -> Option<std::path::PathBuf> {
    #[cfg(windows)]
    {
        let sys_root = std::env::var_os("SystemRoot")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Windows"));
        let p = sys_root.join("System32").join(name);
        p.exists().then_some(p)
    }
    #[cfg(not(windows))]
    {
        let _ = name;
        None
    }
}

fn command_exists(name: &str) -> bool {
    command_path(name).is_some()
}

fn command_path(name: &str) -> Option<String> {
    let exe_name = if cfg!(windows) && !name.ends_with(".exe") {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(&exe_name);
        if candidate.is_file() {
            return Some(candidate.display().to_string());
        }
    }
    None
}
