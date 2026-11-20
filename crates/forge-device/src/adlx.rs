#[cfg(windows)]
use std::ffi::{c_char, CStr};

use serde::{Deserialize, Serialize};

/// Probe status for AMD Device Library eXtra (`amdadlx64.dll`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdlxProbeReport {
    pub available: bool,
    pub version: Option<String>,
    pub full_version: Option<String>,
    /// Verified absolute path of the loaded `amdadlx64.dll` (System32 only; see `forge_core::loader`).
    #[serde(default)]
    pub loaded_library: Option<String>,
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
    let loaded = match forge_core::load_system_library("amdadlx64.dll") {
        Ok(loaded) => loaded,
        Err(err) => {
            return AdlxProbeReport {
                available: false,
                version: None,
                full_version: None,
                loaded_library: None,
                error: Some(err.to_string()),
            };
        }
    };
    let loaded_library = Some(loaded.path.display().to_string());
    let lib = loaded.library;

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

        // The DLL loaded, but a version query that fails is surfaced instead of being ignored.
        let error = (version.is_none() && full_version.is_none())
            .then(|| "amdadlx64.dll loaded but exposes no usable version query".to_string());
        AdlxProbeReport {
            available: true,
            version,
            full_version,
            loaded_library,
            error,
        }
    }
}

#[cfg(not(windows))]
pub fn probe_adlx() -> AdlxProbeReport {
    AdlxProbeReport {
        available: false,
        version: None,
        full_version: None,
        loaded_library: None,
        error: Some("ADLX is Windows-only".to_string()),
    }
}
