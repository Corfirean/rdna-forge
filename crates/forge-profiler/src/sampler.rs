use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use forge_core::ForgeResult;
use tracing::debug;

use crate::adlx_ffi::AdlxSession;
use crate::tuning::{GpuTuningReport, PowerLimitChangeResult, TuningResetResult};
use crate::types::{summarize_samples, GpuTelemetrySample, TelemetrySummary};

/// Captures a single ADLX telemetry snapshot across all visible AMD GPUs.
pub fn snapshot_all_gpus() -> ForgeResult<Vec<GpuTelemetrySample>> {
    let session = AdlxSession::open()?;
    session.read_all_samples()
}

/// Captures a single ADLX telemetry snapshot for the GPU matching `pci_device_id` (or discrete fallback).
pub fn snapshot_for_device(pci_device_id: Option<u32>) -> ForgeResult<Option<GpuTelemetrySample>> {
    let session = AdlxSession::open()?;
    session.read_matching_sample(pci_device_id)
}

/// Reads complete ADLX tuning capabilities and current settings across all AMD GPUs (strictly read-only).
pub fn read_all_gpu_tuning() -> ForgeResult<Vec<GpuTuningReport>> {
    let session = AdlxSession::open()?;
    session.read_all_tuning()
}

/// Validates and optionally applies an ADLX manual power limit percentage change (`dry_run = true` never mutates).
pub fn set_gpu_power_limit(
    pci_device_id: Option<u32>,
    limit_pct: i32,
    dry_run: bool,
) -> ForgeResult<PowerLimitChangeResult> {
    let session = AdlxSession::open()?;
    session.set_power_limit(pci_device_id, limit_pct, dry_run)
}

/// Validates and optionally resets GPU tuning to ADLX factory defaults (`dry_run = true` never mutates).
pub fn reset_gpu_tuning_to_factory(
    pci_device_id: Option<u32>,
    dry_run: bool,
) -> ForgeResult<TuningResetResult> {
    let session = AdlxSession::open()?;
    session.reset_tuning_to_factory(pci_device_id, dry_run)
}

/// Background ADLX telemetry sampler active over the lifetime of a benchmark run.
pub struct TelemetryGuard {
    interval_ms: u64,
    pre_run: Option<GpuTelemetrySample>,
    stop_signal: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<JoinHandle<Vec<GpuTelemetrySample>>>,
}

impl TelemetryGuard {
    /// Starts background ADLX sampling for `pci_device_id`, or returns `None` if ADLX is unavailable.
    pub fn start(pci_device_id: Option<u32>, interval: Duration) -> Option<Self> {
        let interval = interval.max(Duration::from_millis(25));
        let interval_ms = u64::try_from(interval.as_millis()).unwrap_or(u64::MAX);
        let stop_signal = Arc::new((Mutex::new(false), Condvar::new()));
        let signal_clone = Arc::clone(&stop_signal);
        let (tx, rx) = mpsc::sync_channel::<Option<GpuTelemetrySample>>(1);

        let handle = thread::Builder::new()
            .name("forge-adlx-telemetry".to_string())
            .spawn(move || {
                let Ok(session) = AdlxSession::open() else {
                    let _ = tx.send(None);
                    return Vec::new();
                };

                let initial = session.read_matching_sample(pci_device_id).ok().flatten();
                let _ = tx.send(initial.clone());

                let Some(first) = initial else {
                    return Vec::new();
                };

                let mut samples = vec![first];
                let (lock, cvar) = &*signal_clone;
                let mut stopped = match lock.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };

                while !*stopped {
                    let (guard, timeout) = match cvar.wait_timeout(stopped, interval) {
                        Ok(res) => res,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    stopped = guard;
                    if *stopped {
                        break;
                    }
                    if timeout.timed_out() {
                        if let Ok(Some(sample)) = session.read_matching_sample(pci_device_id) {
                            samples.push(sample);
                        }
                    }
                }

                // Capture one final sample right before closing the ADLX session.
                if let Ok(Some(final_sample)) = session.read_matching_sample(pci_device_id) {
                    samples.push(final_sample);
                }

                samples
            })
            .ok()?;

        let pre_run = match rx.recv() {
            Ok(Some(sample)) => Some(sample),
            _ => {
                let _ = handle.join();
                debug!("ADLX telemetry unavailable or no matching GPU found; skipping sampling");
                return None;
            }
        };

        Some(Self {
            interval_ms,
            pre_run,
            stop_signal,
            handle: Some(handle),
        })
    }

    /// Stops background polling, terminates the ADLX session cleanly, and summarizes recorded metrics.
    pub fn stop_and_summarize(mut self) -> Option<TelemetrySummary> {
        let samples = self.stop_inner();
        summarize_samples(&samples, self.interval_ms, self.pre_run.as_ref())
    }

    fn stop_inner(&mut self) -> Vec<GpuTelemetrySample> {
        {
            let (lock, cvar) = &*self.stop_signal;
            let mut stopped = match lock.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            *stopped = true;
            cvar.notify_all();
        }

        if let Some(handle) = self.handle.take() {
            handle.join().unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        let _ = self.stop_inner();
    }
}
