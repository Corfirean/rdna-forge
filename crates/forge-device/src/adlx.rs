use std::ffi::{c_char, CStr};

use serde::{Deserialize, Serialize};

/// Probe status for AMD Device Library eXtra (`amdadlx64.dll`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdlxProbeReport {
    pub available: bool,
    pub version: Option<String>,
    pub full_version: Option<String>,
    pub error: Option<String>,
}

/// Decodes a packed 64-bit ADLX full version (`major.minor.release.build`).
pub fn decode_adlx_full_version(raw: u64) -> String {
    let major = (raw >> 48) & 0xffff;
    let minor = (raw >> 32) & 0xffff;
    let release = (raw >> 16) & 0xffff;
    let build = raw & 0xffff;
    format!("{major}.{minor}.{release}.{build}")
}

/// Probes `amdadlx64.dll` dynamically without vendoring proprietary ADLX headers.
#[cfg(windows)]
pub fn probe_adlx() -> AdlxProbeReport {
    let lib = match unsafe { libloading::Library::new("amdadlx64.dll") } {
        Ok(lib) => lib,
        Err(err) => {
            return AdlxProbeReport {
                available: false,
                version: None,
                full_version: None,
                error: Some(format!("amdadlx64.dll unavailable: {err}")),
            };
        }
    };

    unsafe {
        type QueryVersionFn = extern "C" fn(*mut *const c_char) -> i32;
        type QueryFullVersionFn = extern "C" fn(*mut u64) -> i32;

        let version = lib
            .get::<QueryVersionFn>(b"ADLXQueryVersion\0")
            .ok()
            .and_then(|f| {
                let mut ptr: *const c_char = std::ptr::null();
                if f(&mut ptr) == 0 && !ptr.is_null() {
                    let s = CStr::from_ptr(ptr).to_string_lossy().trim().to_string();
                    (!s.is_empty()).then_some(s)
                } else {
                    None
                }
            });

        let full_version = lib
            .get::<QueryFullVersionFn>(b"ADLXQueryFullVersion\0")
            .ok()
            .and_then(|f| {
                let mut raw: u64 = 0;
                (f(&mut raw) == 0 && raw != 0).then(|| decode_adlx_full_version(raw))
            });

        AdlxProbeReport {
            available: true,
            version,
            full_version,
            error: None,
        }
    }
}

#[cfg(not(windows))]
pub fn probe_adlx() -> AdlxProbeReport {
    AdlxProbeReport {
        available: false,
        version: None,
        full_version: None,
        error: Some("ADLX is Windows-only".to_string()),
    }
}
