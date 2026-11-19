use serde::{Deserialize, Serialize};

/// Display adapter metadata extracted from the Windows Registry display class key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryAdapterInfo {
    pub description: String,
    pub driver_version: Option<String>,
    pub adrenalin_version: Option<String>,
    pub release_version: Option<String>,
    pub matching_device_id: Option<String>,
    pub vendor_id: Option<u32>,
    pub device_id: Option<u32>,
    pub vram_bytes: Option<u64>,
}

/// Extracts `(vendor_id, device_id)` from a Windows PnP `MatchingDeviceId` string
/// such as `PCI\VEN_1002&DEV_73AF&REV_C0`.
pub fn parse_pci_ids_from_matching_id(matching_id: &str) -> (Option<u32>, Option<u32>) {
    let upper = matching_id.to_ascii_uppercase();
    let vendor = extract_hex_tag(&upper, "VEN_");
    let device = extract_hex_tag(&upper, "DEV_");
    (vendor, device)
}

fn extract_hex_tag(haystack: &str, prefix: &str) -> Option<u32> {
    let start = haystack.find(prefix)? + prefix.len();
    let tail = &haystack[start..];
    let hex: String = tail
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .take(4)
        .collect();
    if hex.len() == 4 {
        u32::from_str_radix(&hex, 16).ok()
    } else {
        None
    }
}

#[cfg(windows)]
pub fn query_amd_registry_adapters() -> Vec<RegistryAdapterInfo> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::Registry::{
        RegCloseKey, RegEnumKeyExW, RegOpenKeyExW, RegQueryValueExW, HKEY, HKEY_LOCAL_MACHINE,
        KEY_READ, REG_BINARY, REG_DWORD, REG_EXPAND_SZ, REG_QWORD, REG_SZ,
    };

    const DISPLAY_CLASS_PATH: &str =
        "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";

    fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    unsafe fn read_reg_string(hkey: HKEY, name: &str) -> Option<String> {
        let wname = to_wide(name);
        let mut reg_type = 0u32;
        let mut buf = [0u16; 512];
        let mut bytes = (buf.len() * 2) as u32;
        if RegQueryValueExW(
            hkey,
            wname.as_ptr(),
            std::ptr::null(),
            &mut reg_type,
            buf.as_mut_ptr().cast::<u8>(),
            &mut bytes,
        ) != 0
        {
            return None;
        }
        if reg_type != REG_SZ && reg_type != REG_EXPAND_SZ {
            return None;
        }
        let len = (bytes as usize) / 2;
        let end = buf[..len].iter().position(|&c| c == 0).unwrap_or(len);
        let s = OsString::from_wide(&buf[..end])
            .to_string_lossy()
            .trim()
            .to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    unsafe fn read_reg_u64(hkey: HKEY, name: &str) -> Option<u64> {
        let wname = to_wide(name);
        let mut reg_type = 0u32;
        let mut buf = [0u8; 8];
        let mut bytes = buf.len() as u32;
        if RegQueryValueExW(
            hkey,
            wname.as_ptr(),
            std::ptr::null(),
            &mut reg_type,
            buf.as_mut_ptr(),
            &mut bytes,
        ) != 0
        {
            return None;
        }
        if (reg_type == REG_QWORD || reg_type == REG_BINARY) && bytes == 8 {
            Some(u64::from_le_bytes(buf))
        } else if reg_type == REG_DWORD && bytes == 4 {
            Some(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as u64)
        } else {
            None
        }
    }

    let mut adapters = Vec::new();
    let root_wide = to_wide(DISPLAY_CLASS_PATH);
    let mut root_key: HKEY = std::ptr::null_mut();

    unsafe {
        if RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            root_wide.as_ptr(),
            0,
            KEY_READ,
            &mut root_key,
        ) != 0
        {
            return adapters;
        }

        let mut index = 0u32;
        loop {
            let mut sub_buf = [0u16; 128];
            let mut sub_len = sub_buf.len() as u32;
            let rc = RegEnumKeyExW(
                root_key,
                index,
                sub_buf.as_mut_ptr(),
                &mut sub_len,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            index += 1;
            if rc != 0 {
                break;
            }
            let sub_name = OsString::from_wide(&sub_buf[..sub_len as usize])
                .to_string_lossy()
                .to_string();
            if !sub_name.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            let mut child_key: HKEY = std::ptr::null_mut();
            let sub_w = to_wide(&sub_name);
            if RegOpenKeyExW(root_key, sub_w.as_ptr(), 0, KEY_READ, &mut child_key) != 0 {
                continue;
            }

            let desc = read_reg_string(child_key, "DriverDesc");
            let matching = read_reg_string(child_key, "MatchingDeviceId");
            let (vendor_id, device_id) = matching
                .as_deref()
                .map(parse_pci_ids_from_matching_id)
                .unwrap_or((None, None));

            let is_amd = vendor_id == Some(forge_core::AMD_PCI_VENDOR_ID)
                || desc
                    .as_deref()
                    .map(|d| d.contains("AMD") || d.contains("Radeon"))
                    .unwrap_or(false);

            if is_amd {
                adapters.push(RegistryAdapterInfo {
                    description: desc.unwrap_or_else(|| "AMD Display Adapter".to_string()),
                    driver_version: read_reg_string(child_key, "DriverVersion"),
                    adrenalin_version: read_reg_string(child_key, "RadeonSoftwareVersion"),
                    release_version: read_reg_string(child_key, "ReleaseVersion"),
                    matching_device_id: matching,
                    vendor_id,
                    device_id,
                    vram_bytes: read_reg_u64(child_key, "HardwareInformation.qwMemorySize"),
                });
            }

            RegCloseKey(child_key);
        }

        RegCloseKey(root_key);
    }

    adapters
}

#[cfg(not(windows))]
pub fn query_amd_registry_adapters() -> Vec<RegistryAdapterInfo> {
    Vec::new()
}
