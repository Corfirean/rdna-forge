use forge_core::SystemInfo;

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
