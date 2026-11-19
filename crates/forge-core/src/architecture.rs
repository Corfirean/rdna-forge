use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{ForgeError, ForgeResult};

/// AMD PCI Vendor ID.
pub const AMD_PCI_VENDOR_ID: u32 = 0x1002;

/// AMD GPU microarchitecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuArchitecture {
    Rdna1,
    Rdna2,
    Rdna3,
    Rdna35,
    Rdna4,
    Cdna1,
    Cdna2,
    Cdna3,
    Cdna4,
    Unknown,
}

impl fmt::Display for GpuArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rdna1 => write!(f, "RDNA1"),
            Self::Rdna2 => write!(f, "RDNA2"),
            Self::Rdna3 => write!(f, "RDNA3"),
            Self::Rdna35 => write!(f, "RDNA3.5"),
            Self::Rdna4 => write!(f, "RDNA4"),
            Self::Cdna1 => write!(f, "CDNA1"),
            Self::Cdna2 => write!(f, "CDNA2"),
            Self::Cdna3 => write!(f, "CDNA3"),
            Self::Cdna4 => write!(f, "CDNA4"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Hardware ISA and matrix/vector math capabilities for an AMD GPU family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArchitectureCapabilities {
    pub has_wmma: bool,
    pub has_mfma: bool,
    pub has_dp4a_i8: bool,
    pub has_dot8_i4: bool,
    pub has_native_fp16: bool,
    pub has_native_bf16: bool,
    pub has_native_fp8: bool,
    pub supported_wave_sizes: &'static [u32],
}

impl GpuArchitecture {
    /// Returns conservative ISA-level math capabilities for this architecture.
    pub const fn arch_capabilities(self) -> ArchitectureCapabilities {
        match self {
            Self::Rdna1 => ArchitectureCapabilities {
                has_wmma: false,
                has_mfma: false,
                has_dp4a_i8: false,
                has_dot8_i4: false,
                has_native_fp16: true,
                has_native_bf16: false,
                has_native_fp8: false,
                supported_wave_sizes: &[32, 64],
            },
            Self::Rdna2 => ArchitectureCapabilities {
                has_wmma: false,
                has_mfma: false,
                has_dp4a_i8: true,
                has_dot8_i4: true,
                has_native_fp16: true,
                has_native_bf16: false,
                has_native_fp8: false,
                supported_wave_sizes: &[32, 64],
            },
            Self::Rdna3 | Self::Rdna35 => ArchitectureCapabilities {
                has_wmma: true,
                has_mfma: false,
                has_dp4a_i8: true,
                has_dot8_i4: true,
                has_native_fp16: true,
                has_native_bf16: true,
                has_native_fp8: false,
                supported_wave_sizes: &[32, 64],
            },
            Self::Rdna4 => ArchitectureCapabilities {
                has_wmma: true,
                has_mfma: false,
                has_dp4a_i8: true,
                has_dot8_i4: true,
                has_native_fp16: true,
                has_native_bf16: true,
                has_native_fp8: true,
                supported_wave_sizes: &[32, 64],
            },
            Self::Cdna1 => ArchitectureCapabilities {
                has_wmma: false,
                has_mfma: true,
                has_dp4a_i8: true,
                has_dot8_i4: false,
                has_native_fp16: true,
                has_native_bf16: true,
                has_native_fp8: false,
                supported_wave_sizes: &[64],
            },
            Self::Cdna2 => ArchitectureCapabilities {
                has_wmma: false,
                has_mfma: true,
                has_dp4a_i8: true,
                has_dot8_i4: false,
                has_native_fp16: true,
                has_native_bf16: true,
                has_native_fp8: false,
                supported_wave_sizes: &[64],
            },
            Self::Cdna3 | Self::Cdna4 => ArchitectureCapabilities {
                has_wmma: false,
                has_mfma: true,
                has_dp4a_i8: true,
                has_dot8_i4: false,
                has_native_fp16: true,
                has_native_bf16: true,
                has_native_fp8: true,
                supported_wave_sizes: &[64],
            },
            Self::Unknown => ArchitectureCapabilities {
                has_wmma: false,
                has_mfma: false,
                has_dp4a_i8: false,
                has_dot8_i4: false,
                has_native_fp16: false,
                has_native_bf16: false,
                has_native_fp8: false,
                supported_wave_sizes: &[],
            },
        }
    }
}

/// Validated AMD LLVM/ROCm target identifier (for example `gfx1030` or `gfx1030:xnack-`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GfxTarget {
    pub raw: String,
    pub base: String,
    pub major: u16,
    pub minor: u8,
    pub stepping: u8,
    pub qualifiers: Vec<String>,
    pub architecture: GpuArchitecture,
}

impl GfxTarget {
    /// Parses a target string such as `gfx1030`, `gfx1030:xnack-`, or `gfx90a`.
    pub fn parse(input: &str) -> ForgeResult<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(ForgeError::InvalidGfxTarget(input.to_string()));
        }

        let mut parts = trimmed.split(':');
        let base_raw = parts
            .next()
            .ok_or_else(|| ForgeError::InvalidGfxTarget(input.to_string()))?;
        let base = base_raw.to_ascii_lowercase();

        if !base.starts_with("gfx") {
            return Err(ForgeError::InvalidGfxTarget(input.to_string()));
        }

        let suffix = &base[3..];
        // Valid AMD gfx targets have 3 digits/hex chars (gfx90a, gfx942) or 4 digits (gfx1030, gfx1100, gfx1201).
        if suffix.len() < 3 || suffix.len() > 4 {
            return Err(ForgeError::InvalidGfxTarget(input.to_string()));
        }
        if !suffix.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ForgeError::InvalidGfxTarget(input.to_string()));
        }

        let (major_str, minor_str, stepping_str) = if suffix.len() == 3 {
            (&suffix[0..1], &suffix[1..2], &suffix[2..3])
        } else {
            (&suffix[0..2], &suffix[2..3], &suffix[3..4])
        };

        let major = major_str
            .parse::<u16>()
            .map_err(|_| ForgeError::InvalidGfxTarget(input.to_string()))?;
        let minor = u8::from_str_radix(minor_str, 16)
            .map_err(|_| ForgeError::InvalidGfxTarget(input.to_string()))?;
        let stepping = u8::from_str_radix(stepping_str, 16)
            .map_err(|_| ForgeError::InvalidGfxTarget(input.to_string()))?;

        let mut qualifiers = Vec::new();
        for q in parts {
            let q = q.trim();
            if q.is_empty() {
                return Err(ForgeError::InvalidGfxTarget(input.to_string()));
            }
            qualifiers.push(q.to_ascii_lowercase());
        }

        let architecture = classify_architecture(major, minor, stepping);

        Ok(Self {
            raw: trimmed.to_string(),
            base,
            major,
            minor,
            stepping,
            qualifiers,
            architecture,
        })
    }
}

impl fmt::Display for GfxTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.base)
    }
}

fn classify_architecture(major: u16, minor: u8, stepping: u8) -> GpuArchitecture {
    match (major, minor, stepping) {
        (9, 0, 0x08) => GpuArchitecture::Cdna1,
        (9, 0, 0x0a) => GpuArchitecture::Cdna2,
        (9, 4, _) => GpuArchitecture::Cdna3,
        (9, 5, _) => GpuArchitecture::Cdna4,
        (10, 1, _) => GpuArchitecture::Rdna1,
        (10, 3, _) => GpuArchitecture::Rdna2,
        (11, 0, _) => GpuArchitecture::Rdna3,
        (11, 5, _) => GpuArchitecture::Rdna35,
        (11, 7, _) | (12, _, _) => GpuArchitecture::Rdna4,
        _ => GpuArchitecture::Unknown,
    }
}

/// Static metadata from AMD PCI device ID tables used only for known IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownAmdGpu {
    pub device_id: u32,
    pub canonical_name: &'static str,
    pub gfx_target: &'static str,
    pub architecture: GpuArchitecture,
    pub discrete: bool,
}

/// Looks up well-known AMD Radeon PCI device IDs.
///
/// Returns `None` for unrecognized IDs rather than guessing a fallback device.
pub const fn lookup_amd_pci_device(device_id: u32) -> Option<KnownAmdGpu> {
    match device_id {
        0x731F => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 5700 / 5700 XT",
            gfx_target: "gfx1010",
            architecture: GpuArchitecture::Rdna1,
            discrete: true,
        }),
        0x7340 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 5500 / 5500 XT",
            gfx_target: "gfx1012",
            architecture: GpuArchitecture::Rdna1,
            discrete: true,
        }),
        0x73AF => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 6900 XT",
            gfx_target: "gfx1030",
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
        }),
        0x73BF => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 6800 / 6800 XT / 6900 XT",
            gfx_target: "gfx1030",
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
        }),
        0x73DF => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 6700 XT / 6750 XT",
            gfx_target: "gfx1031",
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
        }),
        0x73E3 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon PRO W6600 / RX 6700",
            gfx_target: "gfx1032",
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
        }),
        0x73FF => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 6600 / 6600 XT / 6650 XT",
            gfx_target: "gfx1032",
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
        }),
        0x743F => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 6500 XT / 6400",
            gfx_target: "gfx1034",
            architecture: GpuArchitecture::Rdna2,
            discrete: true,
        }),
        0x1681 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon 680M / 660M Graphics",
            gfx_target: "gfx1035",
            architecture: GpuArchitecture::Rdna2,
            discrete: false,
        }),
        0x164E => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon(TM) Graphics (Raphael)",
            gfx_target: "gfx1036",
            architecture: GpuArchitecture::Rdna2,
            discrete: false,
        }),
        0x744C => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 7900 XT / XTX / GRE",
            gfx_target: "gfx1100",
            architecture: GpuArchitecture::Rdna3,
            discrete: true,
        }),
        0x747E => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 7800 XT / 7700 XT",
            gfx_target: "gfx1101",
            architecture: GpuArchitecture::Rdna3,
            discrete: true,
        }),
        0x7480 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 7600 / 7600 XT",
            gfx_target: "gfx1102",
            architecture: GpuArchitecture::Rdna3,
            discrete: true,
        }),
        0x15BF | 0x1900 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon 780M / 760M Graphics",
            gfx_target: "gfx1103",
            architecture: GpuArchitecture::Rdna3,
            discrete: false,
        }),
        0x150E | 0x1586 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon 890M / 8060S Graphics",
            gfx_target: "gfx1151",
            architecture: GpuArchitecture::Rdna35,
            discrete: false,
        }),
        0x7590 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 9060 / 9060 XT",
            gfx_target: "gfx1200",
            architecture: GpuArchitecture::Rdna4,
            discrete: true,
        }),
        0x7550 => Some(KnownAmdGpu {
            device_id,
            canonical_name: "AMD Radeon RX 9070 / 9070 XT",
            gfx_target: "gfx1201",
            architecture: GpuArchitecture::Rdna4,
            discrete: true,
        }),
        _ => None,
    }
}
