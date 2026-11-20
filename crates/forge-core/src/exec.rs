//! Validation and resolution of user-supplied interpreter paths.
//!
//! The Python interpreter is a *user choice* (a venv, conda environment, `py` launcher shim, or a
//! system install can all be legitimate), so this module deliberately keeps the rules small:
//!
//! * An explicit path (anything with a directory component) must exist and be a regular file.
//!   It is otherwise accepted as given; no particular install location is required.
//! * A bare name such as `python` is resolved by searching `PATH` only. The Windows process
//!   creation search order would also try the application directory first, which could run a
//!   `python.exe` placed next to `forge.exe`.
//! * On Windows, batch files (`.bat`, `.cmd`) are refused: they are interpreted by `cmd.exe`, whose
//!   argument parsing differs from the normal command line and cannot safely carry the `-c <code>`
//!   argument used to probe the environment.
//!
//! The result is always passed to [`std::process::Command`] as a program plus separate arguments;
//! no command line is ever assembled from strings and no shell is involved, so spaces, quotes and
//! other special characters in paths are handled by the OS API rather than by quoting.

use std::path::{Path, PathBuf};

/// Why an interpreter path was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterError {
    Empty,
    InvalidCharacters,
    NotFound(PathBuf),
    NotAFile(PathBuf),
    /// A bare name that is not on `PATH`.
    NotOnPath(String),
    /// Windows batch file.
    BatchFile(PathBuf),
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "the Python executable path is empty"),
            Self::InvalidCharacters => {
                write!(f, "the Python executable path contains a NUL character")
            }
            Self::NotFound(p) => write!(
                f,
                "Python executable '{}' does not exist; pass the path to python.exe with --python",
                p.display()
            ),
            Self::NotAFile(p) => write!(
                f,
                "'{}' is not a file (is it a directory? point --python at python.exe itself)",
                p.display()
            ),
            Self::NotOnPath(n) => write!(
                f,
                "'{n}' was not found on PATH; install Python or pass the full path with --python"
            ),
            Self::BatchFile(p) => write!(
                f,
                "'{}' is a batch file; pass the real python.exe instead (batch files cannot safely receive the probe arguments)",
                p.display()
            ),
        }
    }
}

impl std::error::Error for InterpreterError {}

fn has_directory_component(p: &Path) -> bool {
    p.is_absolute() || p.components().count() > 1
}

#[cfg(windows)]
fn reject_batch(p: &Path) -> Result<(), InterpreterError> {
    match p.extension() {
        Some(e) if e.eq_ignore_ascii_case("bat") || e.eq_ignore_ascii_case("cmd") => {
            Err(InterpreterError::BatchFile(p.to_path_buf()))
        }
        _ => Ok(()),
    }
}

#[cfg(not(windows))]
fn reject_batch(_p: &Path) -> Result<(), InterpreterError> {
    Ok(())
}

/// Resolves `spec` to the executable to launch, searching `PATH` for bare names.
pub fn resolve_python_executable(spec: &Path) -> Result<PathBuf, InterpreterError> {
    let dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    resolve_python_executable_in(spec, &dirs)
}

/// Same as [`resolve_python_executable`] with an explicit list of search directories (tests).
pub fn resolve_python_executable_in(
    spec: &Path,
    search_dirs: &[PathBuf],
) -> Result<PathBuf, InterpreterError> {
    let text = spec.as_os_str().to_string_lossy();
    if text.trim().is_empty() {
        return Err(InterpreterError::Empty);
    }
    if text.contains('\0') {
        return Err(InterpreterError::InvalidCharacters);
    }

    if has_directory_component(spec) {
        reject_batch(spec)?;
        return match std::fs::metadata(spec) {
            Ok(m) if m.is_file() => Ok(spec.to_path_buf()),
            Ok(_) => Err(InterpreterError::NotAFile(spec.to_path_buf())),
            Err(_) => Err(InterpreterError::NotFound(spec.to_path_buf())),
        };
    }

    // Bare name: PATH only. On Windows a name without an extension gets `.exe` appended.
    let name = spec.as_os_str();
    #[cfg(windows)]
    let candidates = {
        let mut c = vec![name.to_os_string()];
        if spec.extension().is_none() {
            let mut with_exe = name.to_os_string();
            with_exe.push(".exe");
            c.insert(0, with_exe);
        }
        c
    };
    #[cfg(not(windows))]
    let candidates = vec![name.to_os_string()];
    for dir in search_dirs.iter().filter(|d| d.is_absolute()) {
        for cand in &candidates {
            let full = dir.join(cand);
            if full.is_file() {
                reject_batch(&full)?;
                return Ok(full);
            }
        }
    }
    Err(InterpreterError::NotOnPath(text.into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rdna_forge_exec_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn exe_name() -> &'static str {
        if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        }
    }

    #[test]
    fn explicit_existing_file_is_accepted_as_given() {
        let dir = scratch("ok");
        let py = dir.join(exe_name());
        fs::write(&py, b"x").unwrap();
        assert_eq!(resolve_python_executable_in(&py, &[]).unwrap(), py);
    }

    #[test]
    fn spaces_and_shell_metacharacters_in_paths_are_just_characters() {
        // Windows forbids some characters in names; these are all legal there and on Unix.
        let dir = scratch("special").join("my env (x86) & co ^ 'q' ; %PATH% $HOME");
        fs::create_dir_all(&dir).unwrap();
        let py = dir.join(exe_name());
        fs::write(&py, b"x").unwrap();
        assert_eq!(resolve_python_executable_in(&py, &[]).unwrap(), py);
    }

    #[test]
    fn missing_directory_empty_and_nul_are_clear_errors() {
        let dir = scratch("bad");
        assert!(matches!(
            resolve_python_executable_in(&dir.join("nope").join(exe_name()), &[]),
            Err(InterpreterError::NotFound(_))
        ));
        assert!(matches!(
            resolve_python_executable_in(&dir, &[]),
            Err(InterpreterError::NotAFile(_))
        ));
        assert_eq!(
            resolve_python_executable_in(Path::new(""), &[]),
            Err(InterpreterError::Empty)
        );
        assert_eq!(
            resolve_python_executable_in(Path::new("   "), &[]),
            Err(InterpreterError::Empty)
        );
        assert_eq!(
            resolve_python_executable_in(Path::new("py\0thon"), &[]),
            Err(InterpreterError::InvalidCharacters)
        );
        // The messages tell the user what to do.
        let msg = InterpreterError::NotFound(dir.clone()).to_string();
        assert!(msg.contains("--python"));
        assert!(InterpreterError::NotOnPath("python".into())
            .to_string()
            .contains("PATH"));
    }

    #[test]
    fn bare_names_resolve_from_path_dirs_only_and_never_from_the_cwd() {
        let on_path = scratch("path");
        let py = on_path.join(exe_name());
        fs::write(&py, b"x").unwrap();
        assert_eq!(
            resolve_python_executable_in(Path::new("python"), std::slice::from_ref(&on_path))
                .unwrap(),
            py
        );
        // Not on any search dir: an executable in the CWD or elsewhere is NOT picked up.
        let elsewhere = scratch("elsewhere");
        fs::write(elsewhere.join(exe_name()), b"x").unwrap();
        assert!(matches!(
            resolve_python_executable_in(Path::new("python"), &[]),
            Err(InterpreterError::NotOnPath(_))
        ));
        // Relative search entries (such as "." in PATH) are ignored.
        assert!(matches!(
            resolve_python_executable_in(Path::new("python"), &[PathBuf::from(".")]),
            Err(InterpreterError::NotOnPath(_))
        ));
    }

    #[test]
    fn first_matching_path_entry_wins() {
        let a = scratch("a");
        let b = scratch("b");
        fs::write(a.join(exe_name()), b"a").unwrap();
        fs::write(b.join(exe_name()), b"b").unwrap();
        let got = resolve_python_executable_in(Path::new("python"), &[a.clone(), b]).unwrap();
        assert_eq!(got, a.join(exe_name()));
    }

    #[cfg(windows)]
    #[test]
    fn windows_batch_files_are_refused_but_other_wrappers_are_not() {
        let dir = scratch("bat");
        for name in ["python.bat", "python.CMD"] {
            let p = dir.join(name);
            fs::write(&p, b"@echo off").unwrap();
            assert!(matches!(
                resolve_python_executable_in(&p, &[]),
                Err(InterpreterError::BatchFile(_))
            ));
        }
        // A bare name that only exists as a batch file on PATH is refused too, not silently run.
        let only_bat = scratch("onlybat");
        fs::write(only_bat.join("python.bat"), b"x").unwrap();
        assert!(resolve_python_executable_in(Path::new("python.bat"), &[only_bat]).is_err());
        // `py.exe`-style launchers and venv shims (.exe) are legitimate.
        let launcher = dir.join("py.exe");
        fs::write(&launcher, b"x").unwrap();
        assert!(resolve_python_executable_in(&launcher, &[]).is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn extensionless_bare_name_finds_the_exe() {
        let dir = scratch("exe");
        fs::write(dir.join("python.exe"), b"x").unwrap();
        assert_eq!(
            resolve_python_executable_in(Path::new("python"), std::slice::from_ref(&dir)).unwrap(),
            dir.join("python.exe")
        );
    }
}
