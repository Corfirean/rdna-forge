import {
  patchMonitor,
  renderBenchmarks,
  renderLlmOptimizer,
  renderMonitor,
  renderOverview,
  renderTuning,
  renderVideo,
} from "./views.js";

const VIEW_TITLES = {
  overview: "Overview",
  monitor: "Monitor",
  tuning: "Tuning",
  video: "Video (AMF)",
  benchmarks: "Benchmarks",
  llm: "LLM Optimizer",
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
    topVramProcs: [],
    vramProcTick: 0,
    vramProcsError: null,
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
  videoBenchForm: {
    codec: "all",
    resolution: "1920x1080",
    fps: 60,
    bitrate_mbps: 12,
    preset: "speed",
    warmup_frames: 15,
    measured_frames: 120,
  },
  runningVideoBench: false,
  videoBenchReport: null,
  videoBenchError: null,

  videoDecodeForm: {
    codec: "all",
    resolution: "1920x1080",
    fps: 60,
    bitrate_mbps: 12,
    warmup_frames: 15,
    measured_frames: 120,
  },
  runningVideoDecodeBench: false,
  videoDecodeBenchReport: null,
  videoDecodeBenchError: null,

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

  llmAssets: null,
  llmForm: {
    model_path: "",
    context_tokens: 4096,
    safety_margin_mib: 512,
    vulkan_bin: "",
    hip_bin: "",
    context_depth: 2048,
    unlockConfigWrites: false,
  },
  loadingLlmPlan: false,
  llmPlan: null,
  llmRecommendations: null,
  llmError: null,
  runningLlmAutotune: false,
  llmAutotuneReport: null,
  llmAutotuneError: null,
  patchingLlm: false,
  llmPatchResult: null,
  llmPatchError: null,
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
    case "llm":
      renderLlmOptimizer(root, state, actions);
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

async function refreshVramProcesses() {
  try {
    state.monitor.topVramProcs = await invokeCommand("get_top_vram_processes", { limit: 10 });
    state.monitor.vramProcsError = null;
  } catch (err) {
    state.monitor.vramProcsError = String(err);
  }
  if (state.activeTab === "monitor") {
    patchMonitor(document.getElementById("view-root"), state);
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
    if (state.activeTab === "monitor") {
      state.monitor.vramProcTick = (state.monitor.vramProcTick + 1) % 3;
      if (state.monitor.vramProcTick === 1 || state.monitor.topVramProcs.length === 0) {
        try {
          state.monitor.topVramProcs = await invokeCommand("get_top_vram_processes", { limit: 10 });
          state.monitor.vramProcsError = null;
        } catch (procErr) {
          state.monitor.vramProcsError = String(procErr);
        }
      }
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

function parseVideoResolution(res) {
  const [wStr, hStr] = String(res || "1920x1080").split("x");
  return {
    width: Number(wStr) || 1920,
    height: Number(hStr) || 1080,
  };
}

async function runVideoBenchmark() {
  if (state.runningVideoBench || state.busyOp) return;
  state.runningVideoBench = true;
  state.busyOp = true;
  state.videoBenchError = null;
  const { width, height } = parseVideoResolution(state.videoBenchForm.resolution);
  setStatusNote(`Running VCN hardware encode benchmark (${state.videoBenchForm.codec}, ${width}x${height})...`);
  renderActiveView();

  try {
    state.videoBenchReport = await invokeCommand("run_video_benchmark", {
      request: {
        codec: state.videoBenchForm.codec,
        width,
        height,
        fps: Number(state.videoBenchForm.fps) || 60,
        bitrate_mbps: Number(state.videoBenchForm.bitrate_mbps) || 12,
        preset: state.videoBenchForm.preset,
        warmup_frames: Number(state.videoBenchForm.warmup_frames),
        measured_frames: Number(state.videoBenchForm.measured_frames),
      },
    });
    setStatusNote(`Completed VCN hardware encode benchmark (${width}x${height}).`);
  } catch (err) {
    state.videoBenchError = String(err);
  } finally {
    state.runningVideoBench = false;
    state.busyOp = false;
    renderActiveView();
  }
}

async function runVideoDecodeBenchmark() {
  if (state.runningVideoDecodeBench || state.busyOp) return;
  state.runningVideoDecodeBench = true;
  state.busyOp = true;
  state.videoDecodeBenchError = null;
  const { width, height } = parseVideoResolution(state.videoDecodeForm.resolution);
  setStatusNote(`Running VCN hardware decode benchmark (${state.videoDecodeForm.codec}, ${width}x${height})...`);
  renderActiveView();

  try {
    state.videoDecodeBenchReport = await invokeCommand("run_video_decode_benchmark", {
      request: {
        codec: state.videoDecodeForm.codec,
        width,
        height,
        fps: Number(state.videoDecodeForm.fps) || 60,
        bitrate_mbps: Number(state.videoDecodeForm.bitrate_mbps) || 12,
        warmup_frames: Number(state.videoDecodeForm.warmup_frames),
        measured_frames: Number(state.videoDecodeForm.measured_frames),
      },
    });
    setStatusNote(`Completed VCN hardware decode benchmark (${width}x${height}).`);
  } catch (err) {
    state.videoDecodeBenchError = String(err);
  } finally {
    state.runningVideoDecodeBench = false;
    state.busyOp = false;
    renderActiveView();
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
  copyGithubReport: async () => {
    setStatusNote("Generating sanitized GitHub hardware report...");
    try {
      const rep = await invokeCommand("generate_hardware_report", { verify: false });
      const content = `<!-- Suggested Title: ${rep.issue_title} -->\n\n${rep.markdown}`;
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(content);
        setStatusNote(`Copied anonymized GitHub hardware report (${rep.issue_title}).`);
      } else {
        downloadTextFile("rdna-forge-hardware-report.md", content, "text/markdown");
      }
    } catch (err) {
      try {
        // Fallback if clipboard permissions fail on WebView2 focus loss
        const rep = await invokeCommand("generate_hardware_report", { verify: false });
        downloadTextFile(
          "rdna-forge-hardware-report.md",
          `<!-- Suggested Title: ${rep.issue_title} -->\n\n${rep.markdown}`,
          "text/markdown"
        );
      } catch (e2) {
        state.overviewError = String(e2 || err);
        renderActiveView();
      }
    }
  },
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
  refreshVramProcesses,
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
  updateVideoBenchField: (key, value) => {
    state.videoBenchForm[key] = value;
  },
  runVideoBenchmark,
  exportVideoBenchJson: () => {
    if (!state.videoBenchReport) return;
    downloadTextFile(
      `rdna-forge-video-bench-${state.videoBenchForm.resolution}.json`,
      JSON.stringify(state.videoBenchReport, null, 2),
      "application/json"
    );
  },
  updateVideoDecodeField: (key, value) => {
    state.videoDecodeForm[key] = value;
  },
  runVideoDecodeBenchmark,
  exportVideoDecodeJson: () => {
    if (!state.videoDecodeBenchReport) return;
    downloadTextFile(
      `rdna-forge-video-decode-bench-${state.videoDecodeForm.resolution}.json`,
      JSON.stringify(state.videoDecodeBenchReport, null, 2),
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

  updateLlmField: (key, value, autoRefresh) => {
    state.llmForm[key] = value;
    if (autoRefresh) {
      refreshLlmPlan();
    } else if (key === "unlockConfigWrites") {
      renderActiveView();
    }
  },
  refreshLlmPlan,
  runLlmAutotune,
  applyLmStudioPatch,
  applyEnvPatch,
  rollbackLmStudioPatch,
  rollbackEnvPatch,
  copyLlmCommand: (cmdText) => {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      navigator.clipboard.writeText(cmdText).then(
        () => setStatusNote("Copied AMD Radeon llama-server launch command to clipboard."),
        () => setStatusNote("Clipboard copy failed; select and copy command text manually.")
      );
    }
  },
  exportLlmPlanJson: () => {
    if (!state.llmPlan) return;
    downloadTextFile(
      "rdna-forge-llm-plan.json",
      JSON.stringify(state.llmPlan, null, 2),
      "application/json"
    );
  },
};

async function refreshLlmPlan() {
  if (state.loadingLlmPlan) return;
  state.loadingLlmPlan = true;
  state.llmError = null;
  if (state.activeTab === "llm") renderActiveView();

  const reqDevice = Number(state.selectedGpuIndex) || 0;
  const reqCtx = Number(state.llmForm.context_tokens) || 4096;
  const reqMargin = Number(state.llmForm.safety_margin_mib) || 512;

  try {
    state.llmRecommendations = await invokeCommand("get_llm_recommendations", {
      request: {
        device_index: reqDevice,
        context_tokens: reqCtx,
        safety_margin_mib: reqMargin,
      },
    });
  } catch (recErr) {
    // Non-fatal: still attempt GGUF plan below
    state.llmRecommendations = null;
  }

  try {
    if (!state.llmAssets) {
      state.llmAssets = await invokeCommand("get_llm_assets");
      if (!state.llmForm.model_path && state.llmAssets.gguf_models?.length > 0) {
        state.llmForm.model_path = state.llmAssets.gguf_models[0];
      }
      if (!state.llmForm.vulkan_bin && state.llmAssets.vulkan_bench_bin) {
        state.llmForm.vulkan_bin = state.llmAssets.vulkan_bench_bin;
      }
      if (!state.llmForm.hip_bin && state.llmAssets.hip_bench_bin) {
        state.llmForm.hip_bin = state.llmAssets.hip_bench_bin;
      }
    }
    state.llmPlan = await invokeCommand("get_llm_plan", {
      request: {
        model_path: state.llmForm.model_path || null,
        device_index: reqDevice,
        context_tokens: reqCtx,
        safety_margin_mib: reqMargin,
      },
    });
    setStatusNote(`Analyzed GGUF model '${state.llmPlan.model.file_name}'.`);
  } catch (err) {
    state.llmError = String(err);
  } finally {
    state.loadingLlmPlan = false;
    if (state.activeTab === "llm") renderActiveView();
  }
}

async function runLlmAutotune() {
  if (state.runningLlmAutotune || state.busyOp) return;
  state.runningLlmAutotune = true;
  state.busyOp = true;
  state.llmAutotuneError = null;
  setStatusNote("Running empirical llama-bench AMD Radeon autotuner...");
  renderActiveView();

  try {
    state.llmAutotuneReport = await invokeCommand("run_llm_autotune", {
      request: {
        model_path: state.llmForm.model_path || null,
        device_index: Number(state.selectedGpuIndex) || 0,
        vulkan_bin: state.llmForm.vulkan_bin || null,
        hip_bin: state.llmForm.hip_bin || null,
        rocm_lib_dir: state.llmAssets?.rocm_lib_dir || null,
        prompt_tokens: 128,
        gen_tokens: 32,
        context_depth: Number(state.llmForm.context_depth) || 2048,
        reps: 2,
      },
    });
    setStatusNote(
      `Empirical autotuning finished. Best balanced profile: ${state.llmAutotuneReport.best_balanced_profile_id || "n/a"}`
    );
  } catch (err) {
    state.llmAutotuneError = String(err);
  } finally {
    state.runningLlmAutotune = false;
    state.busyOp = false;
    renderActiveView();
  }
}

async function applyLmStudioPatch(profileId, dryRun) {
  if (state.patchingLlm) return;
  state.patchingLlm = true;
  state.llmPatchError = null;
  setStatusNote(
    dryRun
      ? `Generating LM Studio config dry-run preview for profile '${profileId}'...`
      : `Writing LM Studio config & backup for profile '${profileId}'...`
  );
  renderActiveView();

  try {
    const res = await invokeCommand("apply_lm_studio_patch", {
      request: {
        model_path: state.llmForm.model_path || null,
        device_index: Number(state.selectedGpuIndex) || 0,
        profile_id: profileId || null,
        context_tokens: Number(state.llmForm.context_tokens) || 4096,
        safety_margin_mib: Number(state.llmForm.safety_margin_mib) || 512,
        dry_run: Boolean(dryRun),
      },
    });
    state.llmPatchResult = { kind: "lmstudio", data: res };
    setStatusNote(
      res.applied
        ? `Applied LM Studio config (${res.profile_id}). Backup created if prior file existed.`
        : `LM Studio config dry-run preview ready (${res.profile_id}).`
    );
  } catch (err) {
    state.llmPatchError = String(err);
  } finally {
    state.patchingLlm = false;
    renderActiveView();
  }
}

async function applyEnvPatch(dryRun) {
  if (state.patchingLlm) return;
  state.patchingLlm = true;
  state.llmPatchError = null;
  setStatusNote(
    dryRun
      ? "Previewing Windows HKCU environment variables & Ollama Modelfile (dry-run)..."
      : "Applying Windows HKCU environment variables & generating Ollama Modelfile..."
  );
  renderActiveView();

  try {
    const res = await invokeCommand("apply_env_and_ollama_patch", {
      request: {
        model_path: state.llmForm.model_path || null,
        device_index: Number(state.selectedGpuIndex) || 0,
        context_tokens: Number(state.llmForm.context_tokens) || 4096,
        kv_quant: "q8_0",
        dry_run: Boolean(dryRun),
      },
    });
    state.llmPatchResult = { kind: "env", data: res };
    setStatusNote(
      res.applied
        ? "Applied HKCU\\Environment variables for llama.cpp / LM Studio / Ollama."
        : "Windows & Ollama environment dry-run preview ready."
    );
  } catch (err) {
    state.llmPatchError = String(err);
  } finally {
    state.patchingLlm = false;
    renderActiveView();
  }
}

async function rollbackLmStudioPatch(dryRun) {
  if (state.patchingLlm) return;
  state.patchingLlm = true;
  state.llmPatchError = null;
  setStatusNote(
    dryRun
      ? "Scanning LM Studio .json.bak backups (dry-run preview)..."
      : "Restoring LM Studio configs from .json.bak backups..."
  );
  renderActiveView();

  try {
    const res = await invokeCommand("rollback_lm_studio_patch", {
      request: {
        model_path: null,
        dry_run: Boolean(dryRun),
      },
    });
    state.llmPatchResult = { kind: "rollback_lms", data: res };
    setStatusNote(
      res.applied
        ? `Restored ${res.actions?.length || 0} LM Studio config file(s) from .bak backups.`
        : `LM Studio rollback preview: ${res.actions?.length || 0} backup file(s) found.`
    );
  } catch (err) {
    state.llmPatchError = String(err);
  } finally {
    state.patchingLlm = false;
    renderActiveView();
  }
}

async function rollbackEnvPatch(dryRun) {
  if (state.patchingLlm) return;
  state.patchingLlm = true;
  state.llmPatchError = null;
  setStatusNote(
    dryRun
      ? "Inspecting RDNA Forge HKCU\\Environment variables (dry-run preview)..."
      : "Removing RDNA Forge-managed keys from HKCU\\Environment..."
  );
  renderActiveView();

  try {
    const res = await invokeCommand("rollback_env_patch", {
      request: {
        dry_run: Boolean(dryRun),
      },
    });
    state.llmPatchResult = { kind: "rollback_env", data: res };
    const activeCount = (res.keys || []).filter((k) => k.current_value != null).length;
    setStatusNote(
      res.applied
        ? `Removed RDNA Forge HKCU\\Environment variables (${activeCount} active key(s) reset).`
        : `Environment rollback preview: ${activeCount} RDNA Forge key(s) currently set.`
    );
  } catch (err) {
    state.llmPatchError = String(err);
  } finally {
    state.patchingLlm = false;
    renderActiveView();
  }
}

function switchTab(tab) {
  if (!VIEW_TITLES[tab]) return;
  state.activeTab = tab;
  renderActiveView();

  if (tab === "tuning" && !state.tuningReports && !state.loadingTuning) {
    refreshTuning();
  } else if (tab === "video" && !state.videoCaps && !state.loadingVideo) {
    refreshVideoCaps();
  } else if (tab === "llm" && !state.llmPlan && !state.loadingLlmPlan) {
    refreshLlmPlan();
  } else if (tab === "monitor") {
    pollTelemetryOnce();
    refreshVramProcesses();
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
      state.llmPlan = null;
      state.llmRecommendations = null;
      state.llmPatchResult = null;
      state.llmPatchError = null;
      renderActiveView();
      if (state.activeTab === "llm") {
        refreshLlmPlan();
      }
    });
  }

  window.addEventListener("keydown", (e) => {
    if (!e.altKey) return;
    const tabs = ["overview", "monitor", "tuning", "video", "benchmarks", "llm"];
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
