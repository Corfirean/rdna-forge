use forge_core::{SystemInfo, WindowsSystemAudit};

/// Builds a `WindowsSystemAudit` and actionable advisories from raw Windows registry settings
/// and GPU Resizable BAR status.
pub fn build_windows_system_audit(
    hags_mode: Option<u32>,
    tdr_delay_secs: Option<u32>,
    long_paths_enabled: Option<bool>,
    rebar_enabled: Option<bool>,
) -> WindowsSystemAudit {
    let hags_enabled = hags_mode.map(|m| m == 2);
    let effective_tdr_delay_secs = tdr_delay_secs.unwrap_or(2);
    let mut advisories = Vec::new();

    if rebar_enabled == Some(false) {
        advisories.push(
            "Resizable BAR is DISABLED (256 MiB PCIe BAR). Enable Above 4G Decoding and Resizable BAR in UEFI/BIOS to avoid host-device staging bottlenecks."
                .to_string(),
        );
    }
    if effective_tdr_delay_secs <= 2 {
        advisories.push(format!(
            "WDDM TdrDelay is at the Windows default ({effective_tdr_delay_secs}s). Long GPU compute dispatches or heavy LLM prompt prefill exceeding {effective_tdr_delay_secs}s can trigger a driver TDR timeout; consider raising HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\TdrDelay (REG_DWORD) to 10-30s as Administrator."
        ));
    }
    if hags_enabled == Some(true) {
        advisories.push(
            "Hardware-Accelerated GPU Scheduling (HAGS, HwSchMode=2) is ENABLED. If you observe WDDM VRAM preemption stutters or slower multi-app LLM inference, test disabling HAGS."
                .to_string(),
        );
    }
    if long_paths_enabled != Some(true) {
        advisories.push(
            "Windows LongPathsEnabled is DISABLED (260-char MAX_PATH limit). Enable HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled=1 to prevent ROCm/TheRock wheel or HuggingFace cache path errors."
                .to_string(),
        );
    }

    WindowsSystemAudit {
        hags_mode,
        hags_enabled,
        tdr_delay_secs,
        effective_tdr_delay_secs,
        long_paths_enabled,
        advisories,
    }
}

#[cfg(windows)]
fn read_hklm_dword(subkey_path: &str, value_name: &str) -> Option<u32> {
    use windows_sys::Win32::System::Registry::{
        RegCloseKey, RegOpenKeyExW, RegQueryValueExW, HKEY, HKEY_LOCAL_MACHINE, KEY_READ,
        REG_DWORD, REG_QWORD,
    };

    let subkey: Vec<u16> = subkey_path
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let val_w: Vec<u16> = value_name
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let mut hkey: HKEY = std::ptr::null_mut();

    unsafe {
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, KEY_READ, &mut hkey) != 0 {
            return None;
        }
        let mut reg_type: u32 = 0;
        let mut buf = [0u8; 8];
        let mut byte_len = buf.len() as u32;
        let rc = RegQueryValueExW(
            hkey,
            val_w.as_ptr(),
            std::ptr::null(),
            &mut reg_type,
            buf.as_mut_ptr(),
            &mut byte_len,
        );
        RegCloseKey(hkey);
        if rc != 0 {
            return None;
        }
        if reg_type == REG_DWORD && byte_len == 4 {
            Some(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]))
        } else if reg_type == REG_QWORD && byte_len == 8 {
            u32::try_from(u64::from_le_bytes(buf)).ok()
        } else {
            None
        }
    }
}

/// Reads Windows HAGS (`HwSchMode`), WDDM timeout (`TdrDelay`), and `LongPathsEnabled`
/// from HKLM without modifying system state or requiring administrator privileges.
#[cfg(windows)]
pub fn detect_windows_system_audit(rebar_enabled: Option<bool>) -> WindowsSystemAudit {
    const GRAPHICS_DRIVERS: &str = "SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers";
    const FILE_SYSTEM: &str = "SYSTEM\\CurrentControlSet\\Control\\FileSystem";

    let hags_mode = read_hklm_dword(GRAPHICS_DRIVERS, "HwSchMode");
    let tdr_delay_secs = read_hklm_dword(GRAPHICS_DRIVERS, "TdrDelay");
    let long_paths_enabled = read_hklm_dword(FILE_SYSTEM, "LongPathsEnabled").map(|v| v == 1);

    build_windows_system_audit(hags_mode, tdr_delay_secs, long_paths_enabled, rebar_enabled)
}

#[cfg(not(windows))]
pub fn detect_windows_system_audit(rebar_enabled: Option<bool>) -> WindowsSystemAudit {
    build_windows_system_audit(None, None, None, rebar_enabled)
}

#[cfg(windows)]
pub fn detect_system_info() -> SystemInfo {
    use std::ffi::OsString;
    use std::mem::{size_of, zeroed};
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::Registry::{
        RegCloseKey, RegOpenKeyExW, RegQueryValueExW, HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_SZ,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    #[repr(C)]
    #[allow(non_snake_case)]
    struct OsVersionInfoExW {
        dwOSVersionInfoSize: u32,
        dwMajorVersion: u32,
        dwMinorVersion: u32,
        dwBuildNumber: u32,
        dwPlatformId: u32,
        szCSDVersion: [u16; 128],
        wServicePackMajor: u16,
        wServicePackMinor: u16,
        wSuiteMask: u16,
        wProductType: u8,
        wReserved: u8,
    }

    let (major, minor, build) = unsafe {
        let mut info: OsVersionInfoExW = zeroed();
        info.dwOSVersionInfoSize = size_of::<OsVersionInfoExW>() as u32;
        if let Ok(ntdll) = libloading::Library::new("ntdll.dll") {
            if let Ok(rtl_get_version) = ntdll
                .get::<unsafe extern "system" fn(*mut OsVersionInfoExW) -> i32>(b"RtlGetVersion\0")
            {
                if rtl_get_version(&mut info) == 0 {
                    (info.dwMajorVersion, info.dwMinorVersion, info.dwBuildNumber)
                } else {
                    (10, 0, 0)
                }
            } else {
                (10, 0, 0)
            }
        } else {
            (10, 0, 0)
        }
    };

    let os_name = if major == 10 && build >= 22000 {
        "Windows 11".to_string()
    } else if major == 10 {
        "Windows 10".to_string()
    } else if major > 10 {
        format!("Windows {major}")
    } else {
        format!("Windows {major}.{minor}")
    };

    let os_version = format!("{major}.{minor}.{build}");

    let cpu_name = unsafe {
        let subkey: Vec<u16> = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\0"
            .encode_utf16()
            .collect();
        let value_name: Vec<u16> = "ProcessorNameString\0".encode_utf16().collect();
        let mut hkey: HKEY = std::ptr::null_mut();
        let mut name = None;
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, KEY_READ, &mut hkey) == 0 {
            let mut reg_type: u32 = 0;
            let mut buf = [0u16; 256];
            let mut byte_len = (buf.len() * 2) as u32;
            if RegQueryValueExW(
                hkey,
                value_name.as_ptr(),
                std::ptr::null(),
                &mut reg_type,
                buf.as_mut_ptr().cast::<u8>(),
                &mut byte_len,
            ) == 0
                && reg_type == REG_SZ
            {
                let u16_len = (byte_len as usize) / 2;
                let end = buf[..u16_len]
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(u16_len);
                let decoded = OsString::from_wide(&buf[..end])
                    .to_string_lossy()
                    .trim()
                    .to_string();
                if !decoded.is_empty() {
                    name = Some(decoded);
                }
            }
            RegCloseKey(hkey);
        }
        name.unwrap_or_else(|| "Unknown CPU".to_string())
    };

    let total_ram_bytes = unsafe {
        let mut mem: MEMORYSTATUSEX = zeroed();
        mem.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut mem) != 0 {
            mem.ullTotalPhys
        } else {
            0
        }
    };

    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);

    SystemInfo {
        os_name,
        os_version,
        os_build: build,
        architecture: std::env::consts::ARCH.to_string(),
        cpu_name,
        logical_cores,
        total_ram_bytes,
    }
}

#[cfg(not(windows))]
pub fn detect_system_info() -> SystemInfo {
    SystemInfo {
        os_name: std::env::consts::OS.to_string(),
        os_version: "unknown".to_string(),
        os_build: 0,
        architecture: std::env::consts::ARCH.to_string(),
        cpu_name: "Unknown CPU".to_string(),
        logical_cores: std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1),
        total_ram_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_system_audit_flags_default_traps_and_passes_clean_config() {
        let dirty = build_windows_system_audit(Some(2), None, Some(false), Some(false));
        assert_eq!(dirty.hags_enabled, Some(true));
        assert_eq!(dirty.effective_tdr_delay_secs, 2);
        assert_eq!(dirty.advisories.len(), 4);
        assert!(dirty.advisories.iter().any(|a| a.contains("Resizable BAR")));
        assert!(dirty.advisories.iter().any(|a| a.contains("TdrDelay")));
        assert!(dirty.advisories.iter().any(|a| a.contains("HAGS")));
        assert!(dirty
            .advisories
            .iter()
            .any(|a| a.contains("LongPathsEnabled")));

        let clean = build_windows_system_audit(Some(1), Some(20), Some(true), Some(true));
        assert_eq!(clean.hags_enabled, Some(false));
        assert_eq!(clean.effective_tdr_delay_secs, 20);
        assert!(clean.advisories.is_empty());
    }
}
