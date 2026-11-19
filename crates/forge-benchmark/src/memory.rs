#[cfg(windows)]
pub fn peak_process_rss_bytes() -> Option<u64> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    unsafe {
        let mut counters: PROCESS_MEMORY_COUNTERS = zeroed();
        counters.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
            Some(counters.PeakWorkingSetSize as u64)
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
pub fn peak_process_rss_bytes() -> Option<u64> {
    None
}
