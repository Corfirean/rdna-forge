//! Secure dynamic-library loading policy.
//!
//! `LoadLibrary("name.dll")` with a bare name uses the Windows default search order, which can
//! pick up a planted DLL from the application directory, the current working directory or any
//! writable `PATH` entry. RDNA Forge is often run from a user-writable download folder, so every
//! native library is loaded through this module instead.
//!
//! # Policy (Windows)
//!
//! * **System libraries** (`ntdll.dll`, `pdh.dll`, `vulkan-1.dll`, and the AMD driver components
//!   `amdadlx64.dll` / `amfrt64.dll`, which the AMD driver installs into System32) are resolved
//!   *only* inside the directory returned by `GetSystemDirectoryW`, loaded by absolute path with
//!   `LOAD_LIBRARY_SEARCH_SYSTEM32`. Nothing is searched in the application directory, the
//!   current directory or `PATH`.
//! * **Vendor SDK libraries** (the HIP runtime) may come from an SDK install directory. A candidate
//!   path is accepted only if its canonical location (symlinks, junctions and `..` resolved) lies
//!   under a trusted root: the system directory, `%ProgramFiles%` or `%ProgramFiles(x86)%`, as
//!   returned by `SHGetKnownFolderPath` (not environment variables). These directories are
//!   writable only by administrators by default. Candidates outside the roots are rejected and
//!   reported; there is no silent fallback to a bare-name search.
//! * Dependencies of a vendor library are resolved from its own directory and System32 only
//!   (`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32`).
//! * After loading, the module's actual path is read back with `GetModuleFileNameW` and compared
//!   to the validated path; a mismatch unloads the library and fails.
//!
//! Errors distinguish a missing runtime ([`LibraryLoadError::NotFound`]), an untrusted location
//! ([`LibraryLoadError::Untrusted`]) and an OS-level load failure such as a bad image, wrong
//! architecture or missing dependency ([`LibraryLoadError::LoadFailed`]). Initialization failures
//! of an already-loaded library are the caller's concern.
//!
//! On non-Windows targets the platform's `dlopen` rules apply unchanged; this project's
//! validated target is Windows.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::ForgeError;

/// Why a library could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryLoadError {
    /// The runtime is not installed in any trusted location that was searched.
    NotFound {
        name: String,
        searched: Vec<PathBuf>,
    },
    /// A matching file exists but the loading policy refuses it.
    Untrusted { path: PathBuf, reason: String },
    /// The file passed the policy but the OS loader rejected it (bad image, architecture
    /// mismatch, missing dependency, DllMain failure).
    LoadFailed { path: PathBuf, os_error: String },
}

impl fmt::Display for LibraryLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { name, searched } => {
                write!(f, "{name} not found")?;
                if searched.is_empty() {
                    write!(f, " (no trusted location available)")
                } else {
                    let list: Vec<String> =
                        searched.iter().map(|p| p.display().to_string()).collect();
                    write!(f, " in trusted locations: {}", list.join("; "))
                }
            }
            Self::Untrusted { path, reason } => {
                write!(f, "refused to load '{}': {reason}", path.display())
            }
            Self::LoadFailed { path, os_error } => {
                write!(f, "failed to load '{}': {os_error}", path.display())
            }
        }
    }
}

impl std::error::Error for LibraryLoadError {}

impl LibraryLoadError {
    /// Converts to the workspace error type for the named runtime.
    pub fn into_forge_error(self, runtime: &str) -> ForgeError {
        ForgeError::RuntimeUnavailable {
            runtime: runtime.to_string(),
            reason: self.to_string(),
        }
    }
}

/// A loaded library together with the verified path it was loaded from.
#[derive(Debug)]
pub struct LoadedLibrary {
    pub library: libloading::Library,
    /// Absolute path of the loaded module (read back from the OS on Windows).
    pub path: PathBuf,
}

/// Accepts only a plain file name such as `pdh.dll`: no separators, no drive, no `..`.
pub fn validate_library_file_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 128 {
        return Err("library name must be 1..=128 characters".to_string());
    }
    if name == "." || name == ".." || name.contains("..") {
        return Err("library name must not contain '..'".to_string());
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(
            "library name must be a bare file name (ASCII letters, digits, '.', '_', '-')"
                .to_string(),
        );
    }
    Ok(())
}

/// Result of checking a candidate path against trusted roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathCheck {
    /// The file does not exist.
    Missing,
    /// Canonical, validated path inside a trusted root.
    Trusted(PathBuf),
    /// Exists but must not be loaded; carries the reason.
    Rejected(String),
}

/// Strips the `\\?\` verbatim prefix produced by `canonicalize` on Windows (but keeps UNC paths
/// verbatim so they stay distinguishable and are then rejected by the root check).
fn simplify_verbatim(path: PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        if !rest.starts_with("UNC\\") {
            return PathBuf::from(rest.to_string());
        }
    }
    path
}

/// Checks that `candidate` is an absolute path to an existing regular file whose canonical
/// location (symlinks, junctions and `..` resolved) lies under one of `roots`.
pub fn check_trusted_path(candidate: &Path, roots: &[PathBuf]) -> PathCheck {
    if !candidate.is_absolute() {
        return PathCheck::Rejected("path is not absolute".to_string());
    }
    let canonical = match std::fs::canonicalize(candidate) {
        Ok(p) => simplify_verbatim(p),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return PathCheck::Missing,
        Err(e) => return PathCheck::Rejected(format!("cannot resolve path: {e}")),
    };
    match std::fs::metadata(&canonical) {
        Ok(m) if m.is_file() => {}
        Ok(_) => return PathCheck::Rejected("not a regular file".to_string()),
        Err(e) => return PathCheck::Rejected(format!("cannot stat file: {e}")),
    }
    let in_root = roots.iter().any(|root| {
        std::fs::canonicalize(root)
            .map(simplify_verbatim)
            .map(|r| canonical.starts_with(r))
            .unwrap_or(false)
    });
    if in_root {
        PathCheck::Trusted(canonical)
    } else {
        PathCheck::Rejected(format!(
            "resolved location '{}' is outside the trusted install roots",
            canonical.display()
        ))
    }
}

/// Directories from which vendor SDK libraries may be loaded.
///
/// Windows: System32, `%ProgramFiles%` and `%ProgramFiles(x86)%` from the OS (not from the
/// environment). Other platforms: empty (see module docs).
pub fn trusted_runtime_roots() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        win::trusted_roots()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// The Windows system directory (`GetSystemDirectoryW`); `None` elsewhere.
pub fn system_directory() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        win::system_directory()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Loads a Windows system or AMD-driver library by bare file name from System32 only.
///
/// On non-Windows targets this falls back to the platform `dlopen` search for `file_name`.
pub fn load_system_library(file_name: &str) -> Result<LoadedLibrary, LibraryLoadError> {
    if let Err(reason) = validate_library_file_name(file_name) {
        return Err(LibraryLoadError::Untrusted {
            path: PathBuf::from(file_name),
            reason,
        });
    }
    #[cfg(windows)]
    {
        let Some(sys) = win::system_directory() else {
            return Err(LibraryLoadError::NotFound {
                name: file_name.to_string(),
                searched: Vec::new(),
            });
        };
        let candidate = sys.join(file_name);
        match check_trusted_path(&candidate, std::slice::from_ref(&sys)) {
            PathCheck::Missing => Err(LibraryLoadError::NotFound {
                name: file_name.to_string(),
                searched: vec![sys],
            }),
            PathCheck::Rejected(reason) => Err(LibraryLoadError::Untrusted {
                path: candidate,
                reason,
            }),
            PathCheck::Trusted(path) => win::load_verified(path, /* vendor */ false),
        }
    }
    #[cfg(not(windows))]
    {
        // SAFETY: loading a library runs its initializers; the caller names a system library.
        unsafe { libloading::Library::new(file_name) }
            .map(|library| LoadedLibrary {
                library,
                path: PathBuf::from(file_name),
            })
            .map_err(|e| LibraryLoadError::LoadFailed {
                path: PathBuf::from(file_name),
                os_error: e.to_string(),
            })
    }
}

/// Loads the first loadable candidate from an explicit list of absolute paths, applying the
/// trusted-root policy of [`trusted_runtime_roots`].
pub fn load_vendor_library(
    name: &str,
    candidates: &[PathBuf],
) -> Result<LoadedLibrary, LibraryLoadError> {
    load_vendor_library_with_roots(name, candidates, &trusted_runtime_roots())
}

/// Same as [`load_vendor_library`] with explicit trusted roots (used by tests).
pub fn load_vendor_library_with_roots(
    name: &str,
    candidates: &[PathBuf],
    roots: &[PathBuf],
) -> Result<LoadedLibrary, LibraryLoadError> {
    let mut first_untrusted: Option<LibraryLoadError> = None;
    let mut first_failed: Option<LibraryLoadError> = None;
    let mut searched = Vec::new();
    for candidate in candidates {
        match check_trusted_path(candidate, roots) {
            PathCheck::Missing => searched.push(candidate.clone()),
            PathCheck::Rejected(reason) => {
                first_untrusted.get_or_insert(LibraryLoadError::Untrusted {
                    path: candidate.clone(),
                    reason,
                });
            }
            PathCheck::Trusted(path) => match load_trusted(path.clone()) {
                Ok(lib) => return Ok(lib),
                Err(e) => {
                    first_failed.get_or_insert(e);
                }
            },
        }
    }
    Err(first_failed
        .or(first_untrusted)
        .unwrap_or_else(|| LibraryLoadError::NotFound {
            name: name.to_string(),
            searched,
        }))
}

fn load_trusted(path: PathBuf) -> Result<LoadedLibrary, LibraryLoadError> {
    #[cfg(windows)]
    {
        win::load_verified(path, true)
    }
    #[cfg(not(windows))]
    {
        // SAFETY: path was validated by `check_trusted_path`; loading runs initializers.
        unsafe { libloading::Library::new(&path) }
            .map(|library| LoadedLibrary {
                library,
                path: path.clone(),
            })
            .map_err(|e| LibraryLoadError::LoadFailed {
                path,
                os_error: e.to_string(),
            })
    }
}

#[cfg(windows)]
mod win {
    use super::*;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    use libloading::os::windows::{
        Library as WinLibrary, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
    };
    use windows_sys::core::GUID;
    use windows_sys::Win32::System::Com::CoTaskMemFree;
    use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;
    use windows_sys::Win32::UI::Shell::{
        FOLDERID_ProgramFiles, FOLDERID_ProgramFilesX86, SHGetKnownFolderPath,
    };

    pub fn system_directory() -> Option<PathBuf> {
        let mut buf = [0u16; 512];
        // SAFETY: `buf` is a valid writable buffer of the stated length.
        let len = unsafe { GetSystemDirectoryW(buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if len == 0 || len >= buf.len() {
            return None;
        }
        Some(PathBuf::from(OsString::from_wide(&buf[..len])))
    }

    fn known_folder(id: &GUID) -> Option<PathBuf> {
        let mut raw: *mut u16 = std::ptr::null_mut();
        // SAFETY: `id` points to a valid GUID, flags 0 and a null token request the current
        // user's view of a machine-wide folder, and `raw` receives a CoTaskMem-allocated string.
        let hr = unsafe { SHGetKnownFolderPath(id, 0, std::ptr::null_mut(), &mut raw) };
        if hr < 0 || raw.is_null() {
            return None;
        }
        // SAFETY: on success `raw` is a NUL-terminated UTF-16 string that we own and must free.
        let path = unsafe {
            let mut len = 0usize;
            while *raw.add(len) != 0 {
                len += 1;
            }
            let p = PathBuf::from(OsString::from_wide(std::slice::from_raw_parts(raw, len)));
            CoTaskMemFree(raw.cast());
            p
        };
        Some(path)
    }

    pub fn trusted_roots() -> Vec<PathBuf> {
        let mut roots = Vec::new();
        if let Some(p) = system_directory() {
            roots.push(p);
        }
        for id in [&FOLDERID_ProgramFiles, &FOLDERID_ProgramFilesX86] {
            if let Some(p) = known_folder(id) {
                if !roots.contains(&p) {
                    roots.push(p);
                }
            }
        }
        roots
    }

    fn module_path(handle: isize) -> Option<PathBuf> {
        let mut cap = 512usize;
        while cap <= 32_768 {
            let mut buf = vec![0u16; cap];
            // SAFETY: `handle` is a live module handle owned by the caller and `buf` is writable
            // for `cap` UTF-16 units.
            let len =
                unsafe { GetModuleFileNameW(handle as _, buf.as_mut_ptr(), cap as u32) } as usize;
            if len == 0 {
                return None;
            }
            if len < cap {
                return Some(PathBuf::from(OsString::from_wide(&buf[..len])));
            }
            cap *= 2; // truncated: retry with a larger buffer
        }
        None
    }

    /// Loads `path` (already validated and canonical) with a restricted dependency search and
    /// verifies the module that actually got mapped.
    pub fn load_verified(path: PathBuf, vendor: bool) -> Result<LoadedLibrary, LibraryLoadError> {
        let flags = if vendor {
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32
        } else {
            LOAD_LIBRARY_SEARCH_SYSTEM32
        };
        // SAFETY: `path` is an absolute, validated file; the restricted flags prevent dependency
        // resolution from the CWD, application directory or PATH. Library initializers run here.
        let lib = unsafe { WinLibrary::load_with_flags(&path, flags) }.map_err(|e| {
            LibraryLoadError::LoadFailed {
                path: path.clone(),
                os_error: e.to_string(),
            }
        })?;

        let raw = lib.into_raw();
        let actual = module_path(raw);
        // SAFETY: `raw` came from `into_raw` just above and has not been freed.
        let lib = unsafe { WinLibrary::from_raw(raw) };

        let actual = actual.map(|p| simplify_verbatim(std::fs::canonicalize(&p).unwrap_or(p)));
        match actual {
            Some(actual) if actual == path => Ok(LoadedLibrary {
                library: lib.into(),
                path: actual,
            }),
            Some(actual) => Err(LibraryLoadError::Untrusted {
                path: actual,
                reason: format!(
                    "loaded module path differs from the validated path '{}'",
                    path.display()
                ),
            }),
            None => Err(LibraryLoadError::Untrusted {
                path,
                reason: "could not read back the loaded module path".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rdna_forge_loader_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn library_names_must_be_bare_file_names() {
        assert!(validate_library_file_name("amdadlx64.dll").is_ok());
        assert!(validate_library_file_name("vulkan-1.dll").is_ok());
        for bad in [
            "",
            "..\\evil.dll",
            "..",
            "a/b.dll",
            "a\\b.dll",
            "C:\\x.dll",
            "C:x.dll",
            "evil..dll",
            "name with space.dll",
            "\\\\server\\share\\x.dll",
        ] {
            assert!(validate_library_file_name(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn candidate_outside_trusted_root_is_rejected_even_if_it_exists() {
        let trusted = scratch("trusted");
        let other = scratch("other");
        let planted = other.join("amdhip64.dll");
        fs::write(&planted, b"not really a dll").unwrap();

        match check_trusted_path(&planted, std::slice::from_ref(&trusted)) {
            PathCheck::Rejected(reason) => assert!(reason.contains("outside")),
            other => panic!("expected rejection, got {other:?}"),
        }
        let err =
            load_vendor_library_with_roots("amdhip64.dll", &[planted], &[trusted]).unwrap_err();
        assert!(matches!(err, LibraryLoadError::Untrusted { .. }), "{err}");
    }

    #[test]
    fn dot_dot_traversal_out_of_the_root_is_rejected() {
        let root = scratch("root");
        let outside = scratch("outside");
        let planted = outside.join("lib.dll");
        fs::write(&planted, b"x").unwrap();
        let sneaky = root
            .join("..")
            .join(outside.file_name().unwrap())
            .join("lib.dll");
        assert!(matches!(
            check_trusted_path(&sneaky, &[root]),
            PathCheck::Rejected(_)
        ));
    }

    #[test]
    fn relative_directory_and_missing_candidates_are_classified() {
        let root = scratch("classify");
        assert!(matches!(
            check_trusted_path(Path::new("amdhip64.dll"), std::slice::from_ref(&root)),
            PathCheck::Rejected(r) if r.contains("not absolute")
        ));
        assert_eq!(
            check_trusted_path(&root.join("missing.dll"), std::slice::from_ref(&root)),
            PathCheck::Missing
        );
        let sub = root.join("dir.dll");
        fs::create_dir_all(&sub).unwrap();
        assert!(matches!(
            check_trusted_path(&sub, std::slice::from_ref(&root)),
            PathCheck::Rejected(r) if r.contains("regular file")
        ));
        let err = load_vendor_library_with_roots(
            "x.dll",
            &[root.join("missing.dll")],
            std::slice::from_ref(&root),
        )
        .unwrap_err();
        assert!(
            matches!(err, LibraryLoadError::NotFound { ref searched, .. } if searched.len() == 1)
        );
        assert!(err.to_string().contains("x.dll not found"));
    }

    #[test]
    fn trusted_but_unloadable_file_reports_load_failure_not_missing() {
        let root = scratch("badimage");
        let bad = root.join("broken.dll");
        fs::write(&bad, b"this is not a PE/ELF image").unwrap();
        let err = load_vendor_library_with_roots("broken.dll", &[bad], &[root]).unwrap_err();
        assert!(matches!(err, LibraryLoadError::LoadFailed { .. }), "{err}");
    }

    #[test]
    fn failed_load_takes_priority_over_untrusted_candidate() {
        let root = scratch("prio");
        let other = scratch("prio_other");
        let untrusted = other.join("a.dll");
        let broken = root.join("b.dll");
        fs::write(&untrusted, b"x").unwrap();
        fs::write(&broken, b"x").unwrap();
        let err =
            load_vendor_library_with_roots("a.dll", &[untrusted, broken], &[root]).unwrap_err();
        assert!(matches!(err, LibraryLoadError::LoadFailed { .. }), "{err}");
    }

    #[cfg(windows)]
    #[test]
    fn system_libraries_resolve_only_inside_system32() {
        let sys = system_directory().expect("system directory");
        let sys = simplify_verbatim(fs::canonicalize(sys).unwrap());
        let lib = load_system_library("kernel32.dll").expect("kernel32 loads");
        assert!(
            lib.path.starts_with(&sys),
            "{:?} not under {:?}",
            lib.path,
            sys
        );
        assert!(lib
            .path
            .file_name()
            .unwrap()
            .eq_ignore_ascii_case("kernel32.dll"));

        // A name that exists nowhere reports NotFound with the searched directory.
        match load_system_library("rdna_forge_no_such_runtime.dll") {
            Err(LibraryLoadError::NotFound { searched, .. }) => assert_eq!(searched.len(), 1),
            other => panic!("expected NotFound, got {other:?}"),
        }
        // Path-like names are refused before touching the loader.
        assert!(matches!(
            load_system_library("..\\kernel32.dll"),
            Err(LibraryLoadError::Untrusted { .. })
        ));
    }

    #[cfg(windows)]
    #[test]
    fn trusted_roots_come_from_the_os_and_include_system32() {
        let roots = trusted_runtime_roots();
        let sys = system_directory().unwrap();
        assert!(roots.contains(&sys));
        assert!(roots.len() >= 2, "expected ProgramFiles in {roots:?}");
    }
}
