use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Per-process Windows GPU WDDM memory usage reported via PDH `\GPU Process Memory(*)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessVramUsage {
    pub pid: u32,
    pub process_name: String,
    pub dedicated_vram_bytes: u64,
    pub shared_vram_bytes: u64,
    pub dedicated_vram_mib: u64,
    pub shared_vram_mib: u64,
    #[serde(default)]
    pub adapter_luid: Option<String>,
}

/// Parses a Windows PDH `\GPU Process Memory(*)` instance string such as
/// `pid_1444_luid_0x00000000_0x00010cb7_phys_0` into `(pid, Some(luid))`.
pub fn parse_gpu_process_instance_name(instance: &str) -> Option<(u32, Option<String>)> {
    let trimmed = instance.trim();
    let lower = trimmed.to_ascii_lowercase();
    let pid_start = lower.find("pid_")? + 4;
    let pid_tail = &lower[pid_start..];
    let pid_digits: String = pid_tail
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if pid_digits.is_empty() {
        return None;
    }
    let pid = pid_digits.parse::<u32>().ok()?;

    let adapter_luid = lower.find("_luid_").and_then(|pos| {
        let rest = &trimmed[pos + 6..];
        let rest_lower = &lower[pos + 6..];
        let end = rest_lower.find("_phys_").unwrap_or(rest.len());
        let luid = rest[..end].trim();
        if luid.is_empty() {
            None
        } else {
            Some(luid.to_string())
        }
    });

    Some((pid, adapter_luid))
}

/// Aggregates dedicated and shared GPU memory counter samples by PID, attaches process names,
/// filters zero-usage rows, sorts descending by dedicated VRAM, and truncates to `limit`.
pub fn aggregate_process_vram_samples(
    dedicated: &[(String, u64)],
    shared: &[(String, u64)],
    pid_names: &HashMap<u32, String>,
    limit: usize,
) -> Vec<ProcessVramUsage> {
    #[derive(Default)]
    struct Accum {
        dedicated_bytes: u64,
        shared_bytes: u64,
        best_luid: Option<String>,
        best_luid_bytes: u64,
    }

    let mut by_pid: HashMap<u32, Accum> = HashMap::new();

    for (instance, bytes) in dedicated {
        let Some((pid, luid)) = parse_gpu_process_instance_name(instance) else {
            continue;
        };
        let entry = by_pid.entry(pid).or_default();
        entry.dedicated_bytes = entry.dedicated_bytes.saturating_add(*bytes);
        if let Some(luid_val) = luid {
            if *bytes >= entry.best_luid_bytes || entry.best_luid.is_none() {
                entry.best_luid_bytes = *bytes;
                entry.best_luid = Some(luid_val);
            }
        }
    }

    for (instance, bytes) in shared {
        let Some((pid, luid)) = parse_gpu_process_instance_name(instance) else {
            continue;
        };
        let entry = by_pid.entry(pid).or_default();
        entry.shared_bytes = entry.shared_bytes.saturating_add(*bytes);
        if entry.best_luid.is_none() {
            entry.best_luid = luid;
        }
    }

    let mut rows: Vec<ProcessVramUsage> = by_pid
        .into_iter()
        .filter(|(_, acc)| acc.dedicated_bytes > 0 || acc.shared_bytes > 0)
        .map(|(pid, acc)| {
            let process_name = pid_names.get(&pid).cloned().unwrap_or_else(|| {
                if pid == 0 || pid == 4 {
                    "System".to_string()
                } else {
                    format!("pid-{pid}")
                }
            });
            ProcessVramUsage {
                pid,
                process_name,
                dedicated_vram_bytes: acc.dedicated_bytes,
                shared_vram_bytes: acc.shared_bytes,
                dedicated_vram_mib: acc.dedicated_bytes / (1024 * 1024),
                shared_vram_mib: acc.shared_bytes / (1024 * 1024),
                adapter_luid: acc.best_luid,
            }
        })
        .collect();

    rows.sort_by_key(|r| {
        (
            std::cmp::Reverse(r.dedicated_vram_bytes),
            std::cmp::Reverse(r.shared_vram_bytes),
            r.pid,
        )
    });
    if limit > 0 && rows.len() > limit {
        rows.truncate(limit);
    }
    rows
}

#[cfg(windows)]
mod imp {
    use std::collections::HashMap;
    use std::ptr;

    use forge_core::{ForgeError, ForgeResult};

    use super::{aggregate_process_vram_samples, ProcessVramUsage};

    const ERROR_SUCCESS: u32 = 0;
    const PDH_MORE_DATA: u32 = 0x8000_07D2;
    const PDH_NO_DATA: u32 = 0x8000_07D5;
    const PDH_FMT_LARGE: u32 = 0x0000_0400;
    const PDH_CSTATUS_VALID_DATA: u32 = 0;
    const PDH_CSTATUS_NEW_DATA: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PdhFmtCounterValue {
        c_status: u32,
        _pad: u32,
        large_value: i64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PdhFmtCounterValueItemW {
        sz_name: *mut u16,
        fmt_value: PdhFmtCounterValue,
    }

    type PdhOpenQueryWFn = unsafe extern "system" fn(*const u16, usize, *mut isize) -> u32;
    type PdhAddEnglishCounterWFn =
        unsafe extern "system" fn(isize, *const u16, usize, *mut isize) -> u32;
    type PdhCollectQueryDataFn = unsafe extern "system" fn(isize) -> u32;
    type PdhGetFormattedCounterArrayWFn =
        unsafe extern "system" fn(isize, u32, *mut u32, *mut u32, *mut u8) -> u32;
    type PdhCloseQueryFn = unsafe extern "system" fn(isize) -> u32;

    struct OwnedPdhQuery {
        handle: isize,
        close_fn: PdhCloseQueryFn,
    }

    impl Drop for OwnedPdhQuery {
        fn drop(&mut self) {
            if self.handle != 0 {
                unsafe {
                    (self.close_fn)(self.handle);
                }
                self.handle = 0;
            }
        }
    }

    fn utf16_null(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    unsafe fn read_wide_z(ptr: *const u16) -> String {
        if ptr.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        while len < 512 && *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        String::from_utf16_lossy(slice)
    }

    fn collect_pid_process_names() -> HashMap<u32, String> {
        const TH32CS_SNAPPROCESS: u32 = 0x0000_0002;
        const INVALID_HANDLE_VALUE: isize = -1;

        #[repr(C)]
        struct ProcessEntry32W {
            dw_size: u32,
            cnt_usage: u32,
            th32_process_id: u32,
            th32_default_heap_id: usize,
            th32_module_id: u32,
            cnt_threads: u32,
            th32_parent_process_id: u32,
            pc_pri_class_base: i32,
            dw_flags: u32,
            sz_exe_file: [u16; 260],
        }

        #[link(name = "kernel32")]
        extern "system" {
            fn CreateToolhelp32Snapshot(dw_flags: u32, th32_process_id: u32) -> isize;
            fn Process32FirstW(h_snapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
            fn Process32NextW(h_snapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
            fn CloseHandle(h_object: isize) -> i32;
        }

        let mut map = HashMap::new();
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snap == INVALID_HANDLE_VALUE || snap == 0 {
                return map;
            }
            let mut entry: ProcessEntry32W = std::mem::zeroed();
            entry.dw_size = std::mem::size_of::<ProcessEntry32W>() as u32;
            if Process32FirstW(snap, &mut entry) != 0 {
                loop {
                    let len = entry
                        .sz_exe_file
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(entry.sz_exe_file.len());
                    let name = String::from_utf16_lossy(&entry.sz_exe_file[..len]);
                    if !name.is_empty() {
                        map.insert(entry.th32_process_id, name);
                    }
                    if Process32NextW(snap, &mut entry) == 0 {
                        break;
                    }
                }
            }
            let _ = CloseHandle(snap);
        }
        map
    }

    fn read_counter_array(
        counter: isize,
        get_array: PdhGetFormattedCounterArrayWFn,
    ) -> ForgeResult<Vec<(String, u64)>> {
        let mut buf_size = 0u32;
        let mut item_count = 0u32;
        let rc = unsafe {
            get_array(
                counter,
                PDH_FMT_LARGE,
                &mut buf_size,
                &mut item_count,
                ptr::null_mut(),
            )
        };
        if rc == PDH_NO_DATA {
            return Ok(Vec::new());
        }
        if rc != PDH_MORE_DATA && rc != ERROR_SUCCESS {
            return Err(ForgeError::ExecutionFailed(format!(
                "PdhGetFormattedCounterArrayW size query failed with 0x{rc:08X}"
            )));
        }
        if buf_size == 0 || item_count == 0 {
            return Ok(Vec::new());
        }

        let words = (buf_size as usize).div_ceil(std::mem::size_of::<u64>());
        let mut backing = vec![0u64; words];
        let rc2 = unsafe {
            get_array(
                counter,
                PDH_FMT_LARGE,
                &mut buf_size,
                &mut item_count,
                backing.as_mut_ptr().cast::<u8>(),
            )
        };
        if rc2 == PDH_NO_DATA {
            return Ok(Vec::new());
        }
        if rc2 != ERROR_SUCCESS {
            return Err(ForgeError::ExecutionFailed(format!(
                "PdhGetFormattedCounterArrayW failed with 0x{rc2:08X}"
            )));
        }

        let item_size = std::mem::size_of::<PdhFmtCounterValueItemW>();
        let max_items = (buf_size as usize) / item_size;
        let count = (item_count as usize).min(max_items);
        let items = unsafe {
            std::slice::from_raw_parts(backing.as_ptr().cast::<PdhFmtCounterValueItemW>(), count)
        };

        let mut out = Vec::with_capacity(count);
        for item in items {
            if item.fmt_value.c_status == PDH_CSTATUS_VALID_DATA
                || item.fmt_value.c_status == PDH_CSTATUS_NEW_DATA
            {
                let name = unsafe { read_wide_z(item.sz_name) };
                let val = item.fmt_value.large_value.max(0) as u64;
                if !name.is_empty() && val > 0 {
                    out.push((name, val));
                }
            }
        }
        Ok(out)
    }

    /// Queries Windows PDH `\GPU Process Memory(*)` counters (no administrator privileges required)
    /// and returns the top processes sorted by dedicated VRAM consumption.
    pub fn query_top_vram_processes(limit: usize) -> ForgeResult<Vec<ProcessVramUsage>> {
        let lib = unsafe { libloading::Library::new("pdh.dll") }.map_err(|e| {
            ForgeError::RuntimeUnavailable {
                runtime: "PDH".to_string(),
                reason: format!("pdh.dll unavailable: {e}"),
            }
        })?;

        let (open_query, add_counter, collect_data, get_array, close_query) = unsafe {
            let open_query = *lib
                .get::<PdhOpenQueryWFn>(b"PdhOpenQueryW\0")
                .map_err(|e| ForgeError::RuntimeUnavailable {
                    runtime: "PDH".to_string(),
                    reason: format!("PdhOpenQueryW missing: {e}"),
                })?;
            let add_counter = *lib
                .get::<PdhAddEnglishCounterWFn>(b"PdhAddEnglishCounterW\0")
                .map_err(|e| ForgeError::RuntimeUnavailable {
                    runtime: "PDH".to_string(),
                    reason: format!("PdhAddEnglishCounterW missing: {e}"),
                })?;
            let collect_data = *lib
                .get::<PdhCollectQueryDataFn>(b"PdhCollectQueryData\0")
                .map_err(|e| ForgeError::RuntimeUnavailable {
                    runtime: "PDH".to_string(),
                    reason: format!("PdhCollectQueryData missing: {e}"),
                })?;
            let get_array = *lib
                .get::<PdhGetFormattedCounterArrayWFn>(b"PdhGetFormattedCounterArrayW\0")
                .map_err(|e| ForgeError::RuntimeUnavailable {
                    runtime: "PDH".to_string(),
                    reason: format!("PdhGetFormattedCounterArrayW missing: {e}"),
                })?;
            let close_query = *lib
                .get::<PdhCloseQueryFn>(b"PdhCloseQuery\0")
                .map_err(|e| ForgeError::RuntimeUnavailable {
                    runtime: "PDH".to_string(),
                    reason: format!("PdhCloseQuery missing: {e}"),
                })?;
            (
                open_query,
                add_counter,
                collect_data,
                get_array,
                close_query,
            )
        };

        let mut q_handle: isize = 0;
        let open_rc = unsafe { open_query(ptr::null(), 0, &mut q_handle) };
        if open_rc != ERROR_SUCCESS || q_handle == 0 {
            return Err(ForgeError::ExecutionFailed(format!(
                "PdhOpenQueryW failed with 0x{open_rc:08X}"
            )));
        }
        let query = OwnedPdhQuery {
            handle: q_handle,
            close_fn: close_query,
        };

        let ded_path = utf16_null(r"\GPU Process Memory(*)\Dedicated Usage");
        let sh_path = utf16_null(r"\GPU Process Memory(*)\Shared Usage");

        let mut ded_counter: isize = 0;
        let ded_rc = unsafe { add_counter(query.handle, ded_path.as_ptr(), 0, &mut ded_counter) };
        if ded_rc != ERROR_SUCCESS {
            drop(query);
            drop(lib);
            return Err(ForgeError::ExecutionFailed(format!(
                "PdhAddEnglishCounterW(\\GPU Process Memory(*)\\Dedicated Usage) failed with 0x{ded_rc:08X}"
            )));
        }

        let mut sh_counter: isize = 0;
        let sh_rc = unsafe { add_counter(query.handle, sh_path.as_ptr(), 0, &mut sh_counter) };

        let collect_rc = unsafe { collect_data(query.handle) };
        if collect_rc == PDH_NO_DATA {
            drop(query);
            drop(lib);
            return Ok(Vec::new());
        }
        if collect_rc != ERROR_SUCCESS {
            drop(query);
            drop(lib);
            return Err(ForgeError::ExecutionFailed(format!(
                "PdhCollectQueryData failed with 0x{collect_rc:08X}"
            )));
        }

        let ded_samples = read_counter_array(ded_counter, get_array)?;
        let sh_samples = if sh_rc == ERROR_SUCCESS && sh_counter != 0 {
            read_counter_array(sh_counter, get_array).unwrap_or_default()
        } else {
            Vec::new()
        };

        drop(query);
        drop(lib);

        let pid_names = collect_pid_process_names();
        Ok(aggregate_process_vram_samples(
            &ded_samples,
            &sh_samples,
            &pid_names,
            limit,
        ))
    }
}

#[cfg(not(windows))]
mod stub {
    use forge_core::ForgeResult;

    use super::ProcessVramUsage;

    pub fn query_top_vram_processes(_limit: usize) -> ForgeResult<Vec<ProcessVramUsage>> {
        Ok(Vec::new())
    }
}

#[cfg(windows)]
pub use imp::query_top_vram_processes;

#[cfg(not(windows))]
pub use stub::query_top_vram_processes;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_windows_pdh_gpu_process_instance_names() {
        assert_eq!(
            parse_gpu_process_instance_name("pid_1444_luid_0x00000000_0x00010cb7_phys_0"),
            Some((1444, Some("0x00000000_0x00010cb7".to_string())))
        );
        assert_eq!(
            parse_gpu_process_instance_name("PID_3748_LUID_0x00000000_0x00010D00"),
            Some((3748, Some("0x00000000_0x00010D00".to_string())))
        );
        assert_eq!(parse_gpu_process_instance_name("pid_42"), Some((42, None)));
        assert_eq!(parse_gpu_process_instance_name("_Total"), None);
        assert_eq!(parse_gpu_process_instance_name("pid_abc_phys_0"), None);
    }

    #[test]
    fn aggregates_sorts_and_truncates_per_process_vram() {
        let mib = 1024u64 * 1024;
        let dedicated = vec![
            ("pid_100_luid_0x0_0x1_phys_0".to_string(), 200 * mib),
            ("pid_100_luid_0x0_0x2_phys_0".to_string(), 50 * mib),
            ("pid_200_luid_0x0_0x1_phys_0".to_string(), 1_200 * mib),
            ("pid_300_luid_0x0_0x1_phys_0".to_string(), 0),
        ];
        let shared = vec![
            ("pid_200_luid_0x0_0x1_phys_0".to_string(), 64 * mib),
            ("pid_400_luid_0x0_0x1_phys_0".to_string(), 128 * mib),
            ("pid_300_luid_0x0_0x1_phys_0".to_string(), 0),
        ];
        let mut names = HashMap::new();
        names.insert(100, "explorer.exe".to_string());
        names.insert(200, "chrome.exe".to_string());

        let top = aggregate_process_vram_samples(&dedicated, &shared, &names, 2);
        assert_eq!(top.len(), 2);

        assert_eq!(top[0].pid, 200);
        assert_eq!(top[0].process_name, "chrome.exe");
        assert_eq!(top[0].dedicated_vram_mib, 1200);
        assert_eq!(top[0].shared_vram_mib, 64);
        assert_eq!(top[0].adapter_luid.as_deref(), Some("0x0_0x1"));

        assert_eq!(top[1].pid, 100);
        assert_eq!(top[1].process_name, "explorer.exe");
        assert_eq!(top[1].dedicated_vram_mib, 250);
        assert_eq!(top[1].shared_vram_mib, 0);
        assert_eq!(top[1].adapter_luid.as_deref(), Some("0x0_0x1"));

        // PID 400 (0 dedicated, 128 MiB shared) stays behind dedicated consumers, PID 300 omitted.
        let all = aggregate_process_vram_samples(&dedicated, &shared, &names, 10);
        assert_eq!(all.len(), 3);
        assert_eq!(all[2].pid, 400);
        assert_eq!(all[2].process_name, "pid-400");
        assert_eq!(all[2].shared_vram_mib, 128);
    }
}
