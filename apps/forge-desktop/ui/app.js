import { patchMonitor, renderBenchmarks, renderMonitor, renderOverview, renderTuning, renderVideo } from "./views.js";

const VIEW_TITLES = {
  overview: "Overview",
  monitor: "Monitor",
  tuning: "Tuning",
  video: "Video (AMF)",
  benchmarks: "Benchmarks",
};

const CSV_HEADER =
  "timestamp_ms,gpu_name,pci_device_id,gpu_usage_pct,gpu_clock_mhz,vram_clock_mhz,gpu_temp_c,hotspot_temp_c,gpu_power_w,fan_rpm,vram_used_mib,total_vram_mib,voltage_mv";

const MAX_HISTORY_SAMPLES = 60;

const state = {
  activeTab: "overview",
  selectedGpuIndex: 0,
  busyOp: false,
  statusNote: "",

  diagnostics: null,
  verifiedCapabilities: null,
  loadingDiagnostics: false,
  verifying: false,
  overviewError: null,

  monitor: {
    running: true,
    intervalMs: 1000,
    pausedReason: null,
    lastError: null,
    lastSamples: [],
    warnings: [],
    historyByDevice: {},
    rawLog: [],
    timerId: null,
    inFlight: false,
  },

  tuningReports: null,
  loadingTuning: false,
  tuningError: null,
  tuningUi: {
    powerInputPct: null,
    unlockWrites: false,
    pendingConfirm: null,
    actionResult: null,
    actionError: null,
  },

  videoCaps: null,
  loadingVideo: false,
  videoError: null,

  benchForm: {
    device_index: 0,
    kernel: "gemv_w8",
    batch_dispatches: 16,
    warmup_iters: 5,
    measured_iters: 20,
    stability_runs: 0,
  },
  runningBench: false,
  benchReport: null,
  benchError: null,
};

function invokeCommand(cmd, args) {
  const tauri = window.__TAURI__;
  if (!tauri || !tauri.core || typeof tauri.core.invoke !== "function") {
    return Promise.reject(new Error("Tauri IPC bridge is unavailable."));
  }
  return tauri.core.invoke(cmd, args);
}

function setStatusNote(msg) {
  state.statusNote = msg || "";
  const noteEl = document.getElementById("status-message");
  if (noteEl) noteEl.textContent = state.statusNote;
}

function downloadTextFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatusNote(`Exported ${filename}`);
}

function csvCell(val) {
  if (val == null) return "";
  const str = String(val);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function updateHeaderAndRail() {
  const gpus = state.diagnostics?.report?.gpus || [];
  const selector = document.getElementById("gpu-selector");
  const railMeta = document.getElementById("rail-meta");
  const headerChip = document.getElementById("header-device-chip");

  if (selector && selector.options.length !== gpus.length) {
    selector.replaceChildren();
    gpus.forEach((g) => {
      const opt = document.createElement("option");
      opt.value = String(g.index);
      const gfx = g.gfx_target?.base || g.architecture;
      const shortName = g.name.replace(/^AMD\s+/i, "");
      opt.textContent = `#${g.index} ${shortName} (${gfx})`;
      if (g.index === state.selectedGpuIndex) opt.selected = true;
      selector.appendChild(opt);
    });
  }

  const activeGpu = gpus.find((g) => g.index === state.selectedGpuIndex) || gpus[0] || null;
  if (activeGpu) {
    const gfx = activeGpu.gfx_target?.raw || activeGpu.architecture;
    const vramGiB = (activeGpu.vram_bytes / (1024 ** 3)).toFixed(1);
    const adrenalin = activeGpu.adrenalin_version ? `Adrenalin ${activeGpu.adrenalin_version}` : "AMD Driver";
    if (railMeta) railMeta.textContent = `${gfx} | ${vramGiB} GiB VRAM`;
    if (headerChip) headerChip.textContent = `${activeGpu.name} | ${gfx} | ${adrenalin}`;
  } else {
    if (railMeta) railMeta.textContent = "No adapter";
    if (headerChip) headerChip.textContent = "No adapter";
  }

  const badge = document.getElementById("polling-indicator");
  const text = document.getElementById("polling-text");
  if (badge && text) {
    badge.classList.remove("is-live", "is-paused");
    if (state.busyOp || state.monitor.pausedReason) {
      badge.classList.add("is-paused");
      text.textContent = "PAUSED (BUSY)";
    } else if (state.monitor.running) {
      badge.classList.add("is-live");
      text.textContent = `LIVE ${state.monitor.intervalMs}ms`;
    } else {
      text.textContent = "IDLE";
    }
  }
}

function renderActiveView() {
  const root = document.getElementById("view-root");
  const titleEl = document.getElementById("view-title");
  if (titleEl) titleEl.textContent = VIEW_TITLES[state.activeTab] || "RDNA Forge";

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.tab === state.activeTab);
  });
  updateHeaderAndRail();

  if (!root) return;
  switch (state.activeTab) {
    case "overview":
      renderOverview(root, state, actions);
      break;
    case "monitor":
      renderMonitor(root, state, actions);
      break;
    case "tuning":
      renderTuning(root, state, actions);
      break;
    case "video":
      renderVideo(root, state, actions);
      break;
    case "benchmarks":
      renderBenchmarks(root, state, actions);
      break;
    default:
      break;
  }
}

async function refreshDiagnostics() {
  if (state.loadingDiagnostics) return;
  state.loadingDiagnostics = true;
  state.overviewError = null;
  if (state.activeTab === "overview") renderActiveView();

  try {
    const resp = await invokeCommand("get_diagnostics");
    state.diagnostics = resp;
    const gpus = resp.report?.gpus || [];
    if (gpus.length > 0 && !gpus.some((g) => g.index === state.selectedGpuIndex)) {
      const preferred = gpus.find((g) => g.discrete) || gpus[0];
      state.selectedGpuIndex = preferred.index;
      state.benchForm.device_index = preferred.index;
    }
    if (resp.telemetry && resp.telemetry.length > 0) {
      ingestTelemetrySamples(resp.telemetry, [], null, null);
    }
    setStatusNote("Hardware diagnostics updated.");
  } catch (err) {
    state.overviewError = String(err);
  } finally {
    state.loadingDiagnostics = false;
    renderActiveView();
  }
}

async function runLiveVerify() {
  if (state.verifying || state.busyOp) return;
  state.verifying = true;
  state.busyOp = true;
  state.overviewError = null;
  setStatusNote("Running live HIP + Vulkan verification checks...");
  renderActiveView();

  try {
    state.verifiedCapabilities = await invokeCommand("run_live_verification");
    setStatusNote("Live hardware capability verification finished.");
  } catch (err) {
    state.overviewError = String(err);
  } finally {
    state.verifying = false;
    state.busyOp = false;
    renderActiveView();
  }
}

function ingestTelemetrySamples(samples, warnings, pausedReason, error) {
  state.monitor.pausedReason = pausedReason || null;
  state.monitor.lastError = error || null;
  state.monitor.warnings = warnings || [];

  if (samples && samples.length > 0) {
    state.monitor.lastSamples = samples;
    for (const s of samples) {
      const key = s.pci_device_id != null ? String(s.pci_device_id) : "default";
      if (!state.monitor.historyByDevice[key]) {
        state.monitor.historyByDevice[key] = [];
      }
      const hist = state.monitor.historyByDevice[key];
      hist.push(s);
      if (hist.length > MAX_HISTORY_SAMPLES) hist.shift();

      state.monitor.rawLog.push(s);
      if (state.monitor.rawLog.length > 3600) state.monitor.rawLog.shift();
    }
  }
}

async function pollTelemetryOnce() {
  if (!state.monitor.running || state.monitor.inFlight) return;
  if (state.busyOp) {
    state.monitor.pausedReason = "Telemetry polling paused while an active GPU task holds exclusive access.";
    updateHeaderAndRail();
    if (state.activeTab === "monitor") {
      patchMonitor(document.getElementById("view-root"), state);
    }
    return;
  }

  state.monitor.inFlight = true;
  try {
    const resp = await invokeCommand("get_telemetry_snapshot");
    if (resp.paused) {
      state.monitor.pausedReason = resp.pause_reason || "Paused while GPU busy.";
    } else {
      ingestTelemetrySamples(resp.samples, resp.warnings, null, resp.error);
    }
    updateHeaderAndRail();
    if (state.activeTab === "monitor") {
      patchMonitor(document.getElementById("view-root"), state);
    }
  } catch (err) {
    state.monitor.lastError = String(err);
    if (state.activeTab === "monitor") {
      patchMonitor(document.getElementById("view-root"), state);
    }
  } finally {
    state.monitor.inFlight = false;
  }
}

function restartMonitorTimer() {
  if (state.monitor.timerId != null) {
    clearInterval(state.monitor.timerId);
    state.monitor.timerId = null;
  }
  if (state.monitor.running) {
    state.monitor.timerId = setInterval(pollTelemetryOnce, state.monitor.intervalMs);
  }
}

async function refreshTuning() {
  if (state.loadingTuning || state.busyOp) return;
  state.loadingTuning = true;
  state.tuningError = null;
  if (state.activeTab === "tuning") renderActiveView();

  try {
    const reports = await invokeCommand("get_tuning_report");
    state.tuningReports = reports;
    const gpu = (state.diagnostics?.report?.gpus || []).find((g) => g.index === state.selectedGpuIndex);
    const activeRep =
      (gpu && reports.find((r) => r.pci_device_id === gpu.device_id)) ||
      reports.find((r) => r.discrete) ||
      reports[0];
    if (activeRep?.power?.power_limit_pct != null) {
      state.tuningUi.powerInputPct = activeRep.power.power_limit_pct;
    }
    setStatusNote("ADLX tuning parameters loaded (read-only).");
  } catch (err) {
    state.tuningError = String(err);
  } finally {
    state.loadingTuning = false;
    if (state.activeTab === "tuning") renderActiveView();
  }
}

async function submitPowerLimit(pciDeviceId, dryRun) {
  if (state.busyOp) return;
  state.busyOp = true;
  state.tuningUi.pendingConfirm = null;
  state.tuningUi.actionResult = null;
  state.tuningUi.actionError = null;
  renderActiveView();

  try {
    const data = await invokeCommand("set_power_limit", {
      request: {
        pci_device_id: pciDeviceId ?? null,
        limit_pct: Number(state.tuningUi.powerInputPct),
        dry_run: Boolean(dryRun),
      },
    });
    state.tuningUi.actionResult = { kind: "power", data };
    if (data.applied) {
      await refreshTuningAfterMutation();
    }
    setStatusNote(dryRun ? "Power limit dry-run passed." : "Power limit applied via ADLX.");
  } catch (err) {
    state.tuningUi.actionError = String(err);
  } finally {
    state.busyOp = false;
    renderActiveView();
  }
}

async function submitResetTuning(pciDeviceId, dryRun) {
  if (state.busyOp) return;
  state.busyOp = true;
  state.tuningUi.pendingConfirm = null;
  state.tuningUi.actionResult = null;
  state.tuningUi.actionError = null;
  renderActiveView();

  try {
    const data = await invokeCommand("reset_tuning", {
      request: {
        pci_device_id: pciDeviceId ?? null,
        dry_run: Boolean(dryRun),
      },
    });
    state.tuningUi.actionResult = { kind: "reset", data };
    if (data.applied) {
      await refreshTuningAfterMutation();
    }
    setStatusNote(dryRun ? "Factory reset dry-run complete." : "GPU tuning reset to factory defaults.");
  } catch (err) {
    state.tuningUi.actionError = String(err);
  } finally {
    state.busyOp = false;
    renderActiveView();
  }
}

async function refreshTuningAfterMutation() {
  try {
    state.busyOp = false;
    await refreshTuning();
  } catch (_) {
    // Error handled inside refreshTuning
  }
}

async function refreshVideoCaps() {
  if (state.loadingVideo) return;
  state.loadingVideo = true;
  state.videoError = null;
  if (state.activeTab === "video") renderActiveView();

  try {
    state.videoCaps = await invokeCommand("get_video_caps");
    setStatusNote(`AMF ${state.videoCaps.runtime_version} codec capabilities probed.`);
  } catch (err) {
    state.videoError = String(err);
  } finally {
    state.loadingVideo = false;
    if (state.activeTab === "video") renderActiveView();
  }
}

async function runBenchmark() {
  if (state.runningBench || state.busyOp) return;
  state.runningBench = true;
  state.busyOp = true;
  state.benchError = null;
  setStatusNote(`Running Vulkan benchmark '${state.benchForm.kernel}'...`);
  renderActiveView();

  try {
    state.benchReport = await invokeCommand("run_vulkan_benchmark", {
      request: {
        device_index: Number(state.benchForm.device_index),
        kernel: state.benchForm.kernel,
        warmup_iters: Number(state.benchForm.warmup_iters),
        measured_iters: Number(state.benchForm.measured_iters),
        stability_runs: Number(state.benchForm.stability_runs),
        batch_dispatches:
          state.benchForm.batch_dispatches == null
            ? null
            : Number(state.benchForm.batch_dispatches),
      },
    });
    setStatusNote(`Completed Vulkan benchmark '${state.benchForm.kernel}'.`);
  } catch (err) {
    state.benchError = String(err);
  } finally {
    state.runningBench = false;
    state.busyOp = false;
    renderActiveView();
  }
}

const actions = {
  refreshDiagnostics,
  runLiveVerify,
  exportDiagnosticsJson: () => {
    if (!state.diagnostics) return;
    downloadTextFile(
      "rdna-forge-diagnostics.json",
      JSON.stringify(state.diagnostics, null, 2),
      "application/json"
    );
  },

  toggleMonitorRunning: () => {
    state.monitor.running = !state.monitor.running;
    restartMonitorTimer();
    renderActiveView();
    if (state.monitor.running) pollTelemetryOnce();
  },
  setMonitorInterval: (ms) => {
    state.monitor.intervalMs = ms;
    restartMonitorTimer();
    updateHeaderAndRail();
  },
  clearMonitorHistory: () => {
    state.monitor.historyByDevice = {};
    state.monitor.rawLog = [];
    renderActiveView();
  },
  exportMonitorCsv: () => {
    const lines = [CSV_HEADER];
    for (const s of state.monitor.rawLog) {
      lines.push(
        [
          s.timestamp_ms,
          s.gpu_name,
          s.pci_device_id != null ? `0x${s.pci_device_id.toString(16).toUpperCase()}` : "",
          s.gpu_usage_pct?.toFixed(1),
          s.gpu_clock_mhz,
          s.vram_clock_mhz,
          s.gpu_temp_c?.toFixed(1),
          s.hotspot_temp_c?.toFixed(1),
          s.gpu_power_w?.toFixed(1),
          s.fan_rpm,
          s.vram_used_mib,
          s.total_vram_mib,
          s.voltage_mv,
        ]
          .map(csvCell)
          .join(",")
      );
    }
    downloadTextFile("rdna-forge-telemetry.csv", lines.join("\n"), "text/csv");
  },
  exportMonitorJsonl: () => {
    const lines = state.monitor.rawLog.map((s) => JSON.stringify(s));
    downloadTextFile("rdna-forge-telemetry.jsonl", lines.join("\n"), "application/x-ndjson");
  },

  refreshTuning,
  setTuningPowerInput: (val, source) => {
    state.tuningUi.powerInputPct = val;
    if (source !== "slider") {
      const s = document.getElementById("tuning-power-slider");
      if (s) s.value = String(val);
    }
    if (source !== "num") {
      const n = document.getElementById("tuning-power-num");
      if (n) n.value = String(val);
    }
  },
  setTuningUnlock: (unlocked) => {
    state.tuningUi.unlockWrites = Boolean(unlocked);
    if (!unlocked) state.tuningUi.pendingConfirm = null;
    renderActiveView();
  },
  requestConfirm: (kind) => {
    state.tuningUi.pendingConfirm = kind;
    renderActiveView();
  },
  submitPowerLimit,
  submitResetTuning,

  refreshVideoCaps,
  exportVideoJson: () => {
    if (!state.videoCaps) return;
    downloadTextFile(
      "rdna-forge-video-caps.json",
      JSON.stringify(state.videoCaps, null, 2),
      "application/json"
    );
  },

  updateBenchField: (key, value) => {
    state.benchForm[key] = value;
    if (key === "kernel") {
      state.benchForm.batch_dispatches = ["gemm", "gemm_opt", "gemm_i8", "flash_attn", "all"].includes(value) ? 1 : 16;
      renderActiveView();
    }
  },
  runBenchmark,
  exportBenchJson: () => {
    if (!state.benchReport) return;
    downloadTextFile(
      `rdna-forge-bench-${state.benchForm.kernel}.json`,
      JSON.stringify(state.benchReport, null, 2),
      "application/json"
    );
  },
};

function switchTab(tab) {
  if (!VIEW_TITLES[tab]) return;
  state.activeTab = tab;
  renderActiveView();

  if (tab === "tuning" && !state.tuningReports && !state.loadingTuning) {
    refreshTuning();
  } else if (tab === "video" && !state.videoCaps && !state.loadingVideo) {
    refreshVideoCaps();
  } else if (tab === "monitor") {
    pollTelemetryOnce();
  }
}

function init() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  const selector = document.getElementById("gpu-selector");
  if (selector) {
    selector.addEventListener("change", (e) => {
      state.selectedGpuIndex = Number(e.target.value);
      state.benchForm.device_index = state.selectedGpuIndex;
      state.tuningUi.powerInputPct = null;
      state.tuningUi.pendingConfirm = null;
      state.tuningUi.actionResult = null;
      state.tuningUi.actionError = null;
      renderActiveView();
    });
  }

  window.addEventListener("keydown", (e) => {
    if (!e.altKey) return;
    const tabs = ["overview", "monitor", "tuning", "video", "benchmarks"];
    const idx = Number(e.key) - 1;
    if (idx >= 0 && idx < tabs.length) {
      e.preventDefault();
      switchTab(tabs[idx]);
    }
  });

  window.addEventListener("resize", () => {
    if (state.activeTab === "monitor") {
      patchMonitor(document.getElementById("view-root"), state);
    }
  });

  renderActiveView();
  refreshDiagnostics();
  restartMonitorTimer();
}

document.addEventListener("DOMContentLoaded", init);
