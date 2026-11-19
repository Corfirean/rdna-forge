import { drawFanCurveChart, drawStripChart } from "./charts.js";

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, val] of Object.entries(props)) {
    if (val === undefined || val === null) continue;
    if (key === "className") {
      node.className = val;
    } else if (key === "dataset") {
      Object.assign(node.dataset, val);
    } else if (key.startsWith("on") && typeof val === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), val);
    } else if (key === "disabled" || key === "checked" || key === "selected") {
      node[key] = Boolean(val);
    } else {
      node.setAttribute(key, String(val));
    }
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.appendChild(
      typeof child === "string" || typeof child === "number"
        ? document.createTextNode(String(child))
        : child
    );
  }
  return node;
}

function fmtBytesGiB(bytes) {
  if (bytes == null) return "n/a";
  return `${(bytes / (1024 ** 3)).toFixed(2)} GiB`;
}

function fmtHex(value, width = 4) {
  if (value == null) return "n/a";
  return `0x${value.toString(16).toUpperCase().padStart(width, "0")}`;
}

function fmtNum(val, digits = 1, unit = "") {
  if (val == null || !Number.isFinite(val)) return "n/a";
  const s = digits === 0 ? String(Math.round(val)) : val.toFixed(digits);
  return unit ? `${s} ${unit}` : s;
}

function fmtRange(r, unit = "") {
  if (!r) return "n/a";
  const u = unit ? ` ${unit}` : "";
  return `${r.min}..=${r.max}${u} (step ${r.step})`;
}

function formatBackendName(kind) {
  const map = {
    vulkan: "Vulkan",
    hip: "HIP",
    triton: "Triton",
    direct_ml: "DirectML",
    open_cl: "OpenCL",
    adlx: "ADLX",
    amf: "AMF",
    llama_cpp: "llama.cpp",
  };
  return map[kind] || kind;
}

function makeStatusBadge(status) {
  const labels = {
    verified: ["VERIFIED", "badge badge-ok"],
    partially_verified: ["PARTIAL", "badge badge-warn"],
    experimental: ["EXPERIMENTAL", "badge badge-warn"],
    unsupported: ["UNSUPPORTED", "badge badge-err"],
    not_tested: ["NOT TESTED", "badge badge-idle"],
  };
  const [text, cls] = labels[status] || [String(status).toUpperCase(), "badge badge-idle"];
  return el("span", { className: cls }, text);
}

function makeAmfAccelBadge(accel) {
  if (accel === "hardware") return el("span", { className: "badge badge-ok" }, "HARDWARE");
  if (accel === "gpu") return el("span", { className: "badge badge-ok" }, "GPU");
  if (accel === "software") return el("span", { className: "badge badge-warn" }, "SOFTWARE");
  if (accel === "not_supported") return el("span", { className: "badge badge-err" }, "NOT SUPPORTED");
  const code = typeof accel === "object" && accel !== null ? accel.unknown : accel;
  return el("span", { className: "badge badge-idle" }, `UNKNOWN(${code})`);
}

function makePanel(title, meta, bodyNode, flush = false) {
  return el(
    "section",
    { className: "panel" },
    el(
      "div",
      { className: "panel-header" },
      el("h2", { className: "panel-title" }, title),
      meta ? el("span", { className: "panel-meta mono" }, meta) : null
    ),
    el("div", { className: flush ? "panel-body panel-body-flush" : "panel-body" }, bodyNode)
  );
}

function makeKvTable(rows) {
  return el(
    "table",
    { className: "kv-table" },
    el(
      "tbody",
      {},
      rows.map(([label, value, mono = true]) =>
        el(
          "tr",
          {},
          el("th", {}, label),
          el("td", { className: mono ? "mono" : "" }, value ?? "n/a")
        )
      )
    )
  );
}

function selectedGpuFromDiagnostics(state) {
  const gpus = state.diagnostics?.report?.gpus || [];
  return (
    gpus.find((g) => g.index === state.selectedGpuIndex) ||
    gpus[0] ||
    null
  );
}

/* ---------------- OVERVIEW TAB ---------------- */

export function renderOverview(root, state, actions) {
  root.replaceChildren();

  const diag = state.diagnostics;
  const toolbar = el(
    "div",
    { className: "toolbar" },
    el(
      "div",
      { className: "toolbar-group" },
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: state.loadingDiagnostics || state.busyOp,
          onClick: actions.refreshDiagnostics,
        },
        state.loadingDiagnostics ? "Refreshing..." : "Refresh Diagnostics"
      ),
      el(
        "button",
        {
          type: "button",
          className: "btn btn-primary",
          disabled: state.verifying || state.busyOp || !diag,
          onClick: actions.runLiveVerify,
        },
        state.verifying ? "Verifying GPUs..." : "Verify Live"
      )
    ),
    el(
      "div",
      { className: "toolbar-group" },
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: !diag,
          onClick: actions.exportDiagnosticsJson,
        },
        "Export JSON"
      )
    )
  );
  root.appendChild(toolbar);

  if (state.overviewError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.overviewError));
  }

  if (!diag) {
    root.appendChild(el("div", { className: "empty-state" }, "Collecting hardware diagnostics..."));
    return;
  }

  const { report } = diag;
  const gpu = selectedGpuFromDiagnostics(state);
  const gfxLabel = gpu?.gfx_target ? `${gpu.gfx_target.raw} (${gpu.architecture.toUpperCase()})` : gpu?.architecture || "n/a";

  const gpuSpecRows = gpu
    ? [
        ["Adapter", gpu.name, false],
        ["Architecture / Target", gfxLabel],
        ["PCI Vendor : Device", `${fmtHex(gpu.vendor_id)}:${fmtHex(gpu.device_id)}`],
        ["PCI Bus ID", gpu.pci_bus_id || "n/a"],
        ["Type", gpu.discrete ? "Discrete GPU" : "Integrated GPU", false],
        ["Dedicated VRAM", fmtBytesGiB(gpu.vram_bytes)],
        ["Resizable BAR", gpu.rebar_enabled === true ? "ENABLED" : gpu.rebar_enabled === false ? "DISABLED (256 MiB BAR)" : "n/a"],
        [
          "Compute Units / WGPs",
          gpu.compute_units != null
            ? `${gpu.compute_units} CUs / ${gpu.wgp_count ?? "--"} WGPs (Wave ${gpu.warp_size ?? "--"})`
            : "n/a",
        ],
        [
          "Max Buffer / Alloc",
          `${fmtBytesGiB(gpu.max_buffer_size_bytes)} / ${fmtBytesGiB(gpu.max_memory_allocation_bytes)}`,
        ],
      ]
    : [["Adapter", "No GPU detected", false]];

  const hostRows = [
    ["OS", `${report.system.os_name} (Build ${report.system.os_build})`, false],
    ["CPU", `${report.system.cpu_name} (${report.system.logical_cores} threads)`, false],
    ["System RAM", fmtBytesGiB(report.system.total_ram_bytes)],
    ["Adrenalin / Driver", gpu ? `${gpu.adrenalin_version || "n/a"} (${gpu.driver_version || "n/a"})` : "n/a"],
    ["Vulkan API", gpu?.vulkan_api_version || "n/a"],
    ["Vulkan Compiler", gpu?.vulkan_driver_info || "n/a", false],
    ["HIP Runtime", gpu?.hip_available ? gpu.hip_runtime_version || "Detected" : "Unavailable"],
    [
      "Vulkan Capabilities",
      gpu
        ? `FP16=${gpu.vulkan_fp16}  INT8=${gpu.vulkan_int8}  DP4A=${gpu.vulkan_int8_dot_product}  CoopMat=${gpu.vulkan_cooperative_matrix}`
        : "n/a",
    ],
  ];

  root.appendChild(
    el(
      "div",
      { className: "grid-2" },
      makePanel("Hardware Target", gpu ? `#${gpu.index}` : "", makeKvTable(gpuSpecRows)),
      makePanel("Host & Runtime Flags", report.system.architecture, makeKvTable(hostRows))
    )
  );

  // Backends Table
  const backendRows = report.backends.map((b) =>
    el(
      "tr",
      {},
      el("td", { className: "mono" }, formatBackendName(b.kind)),
      el("td", {}, makeStatusBadge(b.status)),
      el("td", { className: "mono muted-text" }, b.version || "--"),
      el(
        "td",
        {},
        el("div", {}, b.summary),
        b.details && b.details.length
          ? el("div", { className: "sub-detail mono" }, b.details.join(" | "))
          : null
      )
    )
  );

  const backendsTable = el(
    "div",
    { className: "data-table-wrap" },
    el(
      "table",
      { className: "data-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Backend"),
          el("th", {}, "Status"),
          el("th", {}, "Version"),
          el("th", {}, "Summary & Probe Details")
        )
      ),
      el("tbody", {}, backendRows)
    )
  );
  root.appendChild(makePanel("Detected Backends", `${report.backends.length} audited`, backendsTable, true));

  // Capabilities Verification Table
  const caps = state.verifiedCapabilities || report.capabilities;
  const capRows = caps.map((c) =>
    el(
      "tr",
      {},
      el("td", {}, c.name),
      el("td", {}, makeStatusBadge(c.status)),
      el("td", { className: "mono muted-text" }, c.detail || "--")
    )
  );

  const capsTable = el(
    "div",
    { className: "data-table-wrap" },
    el(
      "table",
      { className: "data-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Capability Check"),
          el("th", {}, "Status"),
          el("th", {}, "Measured Detail")
        )
      ),
      el("tbody", {}, capRows)
    )
  );
  root.appendChild(
    makePanel(
      "Live Hardware Capabilities",
      state.verifiedCapabilities ? "Live verified" : "Pending live verification",
      capsTable,
      true
    )
  );

  if (report.missing_dependencies && report.missing_dependencies.length > 0) {
    root.appendChild(
      makePanel(
        "Notes & Missing Optional Dependencies",
        `${report.missing_dependencies.length} items`,
        el(
          "ul",
          { style: "margin:0;padding-left:18px;color:var(--text-secondary)" },
          report.missing_dependencies.map((note) => el("li", {}, note))
        )
      )
    );
  }
}

/* ---------------- MONITOR TAB ---------------- */

function getMatchingSample(samples, selectedGpu) {
  if (!samples || samples.length === 0) return null;
  if (selectedGpu) {
    const exact = samples.find((s) => s.pci_device_id === selectedGpu.device_id);
    if (exact) return exact;
  }
  return samples.find((s) => s.discrete) || samples[0];
}

function buildMonitorChartSeries(history) {
  return {
    clocks: [
      { label: "SCLK", color: "#e6e8ec", unit: " MHz", data: history.map((h) => h.gpu_clock_mhz ?? null) },
      { label: "MCLK", color: "#4ca866", unit: " MHz", data: history.map((h) => h.vram_clock_mhz ?? null) },
    ],
    thermals: [
      { label: "Core", color: "#c9963b", unit: " C", data: history.map((h) => h.gpu_temp_c ?? null) },
      { label: "Hotspot", color: "#c93b3b", unit: " C", data: history.map((h) => h.hotspot_temp_c ?? null) },
    ],
    power: [
      { label: "GPU Power", color: "#c93b3b", unit: " W", data: history.map((h) => h.gpu_power_w ?? null) },
      { label: "Board", color: "#7a9dc9", unit: " W", data: history.map((h) => h.board_power_w ?? null) },
    ],
    utilization: [
      { label: "GPU Load", color: "#6c9bd2", unit: "%", data: history.map((h) => h.gpu_usage_pct ?? null) },
      {
        label: "VRAM %",
        color: "#c9963b",
        unit: "%",
        data: history.map((h) =>
          h.vram_used_mib != null && h.total_vram_mib > 0
            ? (h.vram_used_mib / h.total_vram_mib) * 100
            : null
        ),
      },
    ],
  };
}

export function renderMonitor(root, state, actions) {
  root.replaceChildren();

  const toolbar = el(
    "div",
    { className: "toolbar" },
    el(
      "div",
      { className: "toolbar-group" },
      el(
        "button",
        {
          type: "button",
          className: state.monitor.running ? "btn" : "btn btn-primary",
          onClick: actions.toggleMonitorRunning,
        },
        state.monitor.running ? "Pause Polling" : "Resume Polling"
      ),
      el(
        "select",
        {
          className: "select-input mono",
          value: String(state.monitor.intervalMs),
          onChange: (e) => actions.setMonitorInterval(Number(e.target.value)),
        },
        el("option", { value: "500", selected: state.monitor.intervalMs === 500 }, "500 ms"),
        el("option", { value: "1000", selected: state.monitor.intervalMs === 1000 }, "1000 ms"),
        el("option", { value: "2000", selected: state.monitor.intervalMs === 2000 }, "2000 ms")
      ),
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: state.monitor.rawLog.length === 0,
          onClick: actions.clearMonitorHistory,
        },
        "Clear History"
      )
    ),
    el(
      "div",
      { className: "toolbar-group" },
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: state.monitor.rawLog.length === 0,
          onClick: actions.exportMonitorCsv,
        },
        "Export CSV"
      ),
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: state.monitor.rawLog.length === 0,
          onClick: actions.exportMonitorJsonl,
        },
        "Export JSONL"
      )
    )
  );
  root.appendChild(toolbar);

  const alertSlot = el("div", { id: "monitor-alerts" });
  root.appendChild(alertSlot);

  const kpiNames = [
    ["sclk", "GPU Clock (SCLK)"],
    ["mclk", "VRAM Clock (MCLK)"],
    ["load", "GPU Utilization"],
    ["temp", "Core / Hotspot"],
    ["power", "GPU Power"],
    ["fan", "Fan Speed"],
    ["volt", "Core Voltage"],
    ["vram", "VRAM Used (WDDM)"],
  ];
  const kpiStrip = el(
    "div",
    { className: "kpi-strip" },
    kpiNames.map(([id, label]) =>
      el(
        "div",
        { className: "kpi-tile" },
        el("div", { className: "kpi-label" }, label),
        el("div", { className: "kpi-value", id: `kpi-val-${id}` }, "--"),
        el("div", { className: "kpi-sub", id: `kpi-sub-${id}` }, "\u00a0")
      )
    )
  );
  root.appendChild(kpiStrip);

  const cClocks = el("canvas", { className: "chart-canvas", id: "chart-clocks" });
  const cThermals = el("canvas", { className: "chart-canvas", id: "chart-thermals" });
  const cPower = el("canvas", { className: "chart-canvas", id: "chart-power" });
  const cUtil = el("canvas", { className: "chart-canvas", id: "chart-util" });

  root.appendChild(
    el(
      "div",
      { className: "grid-2" },
      makePanel("Clocks (MHz)", "60 samples", cClocks),
      makePanel("Temperatures (C)", "60 samples", cThermals),
      makePanel("Power Draw (W)", "60 samples", cPower),
      makePanel("Load & WDDM VRAM Occupancy (%)", "50% WDDM threshold", cUtil)
    )
  );

  patchMonitor(root, state);
}

export function patchMonitor(root, state) {
  const alertSlot = root.querySelector("#monitor-alerts");
  if (!alertSlot) return;

  const banners = [];
  if (state.monitor.pausedReason) {
    banners.push(el("div", { className: "banner banner-warn" }, state.monitor.pausedReason));
  }
  if (state.monitor.lastError) {
    banners.push(el("div", { className: "banner banner-err" }, state.monitor.lastError));
  }
  for (const w of state.monitor.warnings || []) {
    banners.push(el("div", { className: "banner banner-warn" }, w));
  }
  alertSlot.replaceChildren(...banners);

  const gpu = selectedGpuFromDiagnostics(state);
  const sample = getMatchingSample(state.monitor.lastSamples, gpu);

  const setKpi = (id, val, sub = "\u00a0") => {
    const vEl = root.querySelector(`#kpi-val-${id}`);
    const sEl = root.querySelector(`#kpi-sub-${id}`);
    if (vEl) vEl.textContent = val;
    if (sEl) sEl.textContent = sub;
  };

  if (sample) {
    setKpi("sclk", sample.gpu_clock_mhz != null ? `${sample.gpu_clock_mhz} MHz` : "n/a", sample.gpu_name);
    setKpi("mclk", sample.vram_clock_mhz != null ? `${sample.vram_clock_mhz} MHz` : "n/a", fmtHex(sample.pci_device_id));
    setKpi("load", sample.gpu_usage_pct != null ? `${sample.gpu_usage_pct.toFixed(1)} %` : "n/a");
    setKpi(
      "temp",
      sample.gpu_temp_c != null ? `${sample.gpu_temp_c.toFixed(1)} C` : "n/a",
      sample.hotspot_temp_c != null ? `Hotspot ${sample.hotspot_temp_c.toFixed(1)} C` : "Hotspot n/a"
    );
    setKpi(
      "power",
      sample.gpu_power_w != null ? `${sample.gpu_power_w.toFixed(1)} W` : "n/a",
      sample.board_power_w != null ? `Board ${sample.board_power_w.toFixed(1)} W` : "ASIC Power"
    );
    setKpi("fan", sample.fan_rpm != null ? `${sample.fan_rpm} RPM` : "n/a");
    setKpi("volt", sample.voltage_mv != null ? `${sample.voltage_mv} mV` : "n/a");
    const vramPct =
      sample.vram_used_mib != null && sample.total_vram_mib > 0
        ? `${((sample.vram_used_mib / sample.total_vram_mib) * 100).toFixed(1)}% occupied`
        : "n/a";
    setKpi(
      "vram",
      sample.vram_used_mib != null ? `${sample.vram_used_mib} MiB` : "n/a",
      sample.total_vram_mib != null ? `of ${sample.total_vram_mib} MiB (${vramPct})` : ""
    );
  }

  const key = gpu ? String(gpu.device_id) : "default";
  const history = state.monitor.historyByDevice[key] || [];
  const series = buildMonitorChartSeries(history);

  const cClocks = root.querySelector("#chart-clocks");
  const cThermals = root.querySelector("#chart-thermals");
  const cPower = root.querySelector("#chart-power");
  const cUtil = root.querySelector("#chart-util");

  if (cClocks) drawStripChart(cClocks, series.clocks, { minY: 0 });
  if (cThermals) drawStripChart(cThermals, series.thermals, { minY: 20, maxY: 110 });
  if (cPower) drawStripChart(cPower, series.power, { minY: 0 });
  if (cUtil) {
    drawStripChart(cUtil, series.utilization, {
      minY: 0,
      maxY: 100,
      thresholdY: 50,
      thresholdLabel: "50% WDDM limit",
    });
  }
}

/* ---------------- TUNING TAB ---------------- */

function findTuningReport(state) {
  const reports = state.tuningReports || [];
  if (reports.length === 0) return null;
  const gpu = selectedGpuFromDiagnostics(state);
  if (gpu) {
    const match = reports.find((r) => r.pci_device_id === gpu.device_id);
    if (match) return match;
  }
  return reports.find((r) => r.discrete) || reports[0];
}

export function renderTuning(root, state, actions) {
  root.replaceChildren();

  root.appendChild(
    el(
      "div",
      { className: "toolbar" },
      el(
        "div",
        { className: "toolbar-group" },
        el(
          "button",
          {
            type: "button",
            className: "btn",
            disabled: state.loadingTuning || state.busyOp,
            onClick: actions.refreshTuning,
          },
          state.loadingTuning ? "Reading ADLX..." : "Refresh ADLX Tuning"
        ),
        el(
          "span",
          { className: "muted-text" },
          "All changes default to hardware range dry-run unless explicitly unlocked."
        )
      )
    )
  );

  if (state.tuningError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.tuningError));
  }

  if (!state.tuningReports) {
    root.appendChild(el("div", { className: "empty-state" }, "Click 'Refresh ADLX Tuning' or wait for inspection..."));
    return;
  }

  const rep = findTuningReport(state);
  if (!rep) {
    root.appendChild(el("div", { className: "empty-state" }, "No ADLX tuning interfaces detected."));
    return;
  }

  const caps = rep.capabilities;
  const capSummary = [
    ["Adapter", `${rep.gpu_name} (${fmtHex(rep.pci_device_id)})`, false],
    ["At Factory Defaults", rep.at_factory == null ? "n/a" : rep.at_factory ? "YES" : "MODIFIED"],
    ["Active Preset", rep.preset?.current_preset || "None / Custom", false],
    [
      "Supported Subsystems",
      `Power=${caps.manual_power}  GFX=${caps.manual_gfx}  VRAM=${caps.manual_vram}  Fan=${caps.manual_fan}  Preset=${caps.preset_tuning}  Auto=${caps.auto_tuning}`,
    ],
  ];
  root.appendChild(makePanel("ADLX Tuning Overview", rep.discrete ? "Discrete" : "Integrated", makeKvTable(capSummary)));

  // Left: Power Limit + GFX/VRAM Readonly
  const power = rep.power;
  const powerSupported = Boolean(power && power.power_limit_range);
  const pRange = power?.power_limit_range;
  const currentSliderVal =
    state.tuningUi.powerInputPct != null
      ? state.tuningUi.powerInputPct
      : power?.power_limit_pct ?? pRange?.min ?? 0;

  const powerPanelContent = el("div", {});
  if (!powerSupported) {
    powerPanelContent.appendChild(
      el(
        "div",
        { className: "muted-text" },
        "Manual power tuning is not exposed by ADLX for this adapter."
      )
    );
  } else {
    powerPanelContent.appendChild(
      makeKvTable([
        ["Current Power Limit", `${power.power_limit_pct ?? "--"} %`],
        ["Allowed Hardware Range", fmtRange(pRange, "%")],
        ["Factory Default", `${power.power_limit_default_pct ?? "--"} %`],
        [
          "TDC Limit",
          power.tdc_limit_supported ? `${power.tdc_limit_a ?? "--"} A (${fmtRange(power.tdc_limit_range, "A")})` : "Unsupported",
        ],
      ])
    );

    const slider = el("input", {
      type: "range",
      id: "tuning-power-slider",
      className: "range-slider",
      min: String(pRange.min),
      max: String(pRange.max),
      step: String(Math.max(1, pRange.step)),
      value: String(currentSliderVal),
      onInput: (e) => actions.setTuningPowerInput(Number(e.target.value), "slider"),
    });
    const numInput = el("input", {
      type: "number",
      id: "tuning-power-num",
      className: "text-input mono",
      style: "width:76px",
      min: String(pRange.min),
      max: String(pRange.max),
      step: String(Math.max(1, pRange.step)),
      value: String(currentSliderVal),
      onInput: (e) => actions.setTuningPowerInput(Number(e.target.value), "num"),
    });

    powerPanelContent.appendChild(
      el(
        "div",
        { className: "slider-row" },
        el("span", { className: "mono muted-text" }, "Target %"),
        slider,
        numInput
      )
    );

    powerPanelContent.appendChild(
      el(
        "div",
        { className: "toolbar-group", style: "margin-top:8px" },
        el(
          "button",
          {
            type: "button",
            className: "btn",
            disabled: state.busyOp,
            onClick: () => actions.submitPowerLimit(rep.pci_device_id, true),
          },
          "Dry Run Validate"
        ),
        el(
          "label",
          { className: "checkbox-label" },
          el("input", {
            type: "checkbox",
            checked: state.tuningUi.unlockWrites,
            onChange: (e) => actions.setTuningUnlock(e.target.checked),
          }),
          "Unlock Hardware Writes"
        ),
        el(
          "button",
          {
            type: "button",
            className: "btn btn-primary",
            disabled: state.busyOp || !state.tuningUi.unlockWrites,
            onClick: () => actions.requestConfirm("power_limit"),
          },
          "Apply Power Limit"
        ),
        el(
          "button",
          {
            type: "button",
            className: "btn btn-danger",
            disabled: state.busyOp || !state.tuningUi.unlockWrites,
            onClick: () => actions.requestConfirm("reset"),
          },
          "Reset to Factory"
        )
      )
    );

    if (state.tuningUi.pendingConfirm && state.tuningUi.unlockWrites) {
      const msg =
        state.tuningUi.pendingConfirm === "power_limit"
          ? `Confirm hardware write: SetPowerLimit(${currentSliderVal}%) on ${rep.gpu_name}?`
          : `Confirm hardware write: ResetToFactory() on ${rep.gpu_name}?`;
      powerPanelContent.appendChild(
        el(
          "div",
          { className: "confirm-box" },
          el("span", { className: "mono" }, msg),
          el(
            "div",
            { className: "toolbar-group" },
            el(
              "button",
              {
                type: "button",
                className: "btn btn-danger",
                disabled: state.busyOp,
                onClick: () =>
                  state.tuningUi.pendingConfirm === "power_limit"
                    ? actions.submitPowerLimit(rep.pci_device_id, false)
                    : actions.submitResetTuning(rep.pci_device_id, false),
              },
              "Confirm Hardware Write"
            ),
            el(
              "button",
              {
                type: "button",
                className: "btn",
                onClick: () => actions.requestConfirm(null),
              },
              "Cancel"
            )
          )
        )
      );
    }
  }

  if (state.tuningUi.actionResult) {
    const res = state.tuningUi.actionResult;
    const text =
      res.kind === "power"
        ? `Power Limit ${res.data.applied ? "APPLIED" : "DRY-RUN VALIDATED"}: current=${res.data.previous_limit_pct}%, requested=${res.data.requested_limit_pct}%, verified=${res.data.verified_limit_pct ?? "n/a"}%, range=${res.data.allowed_range.min}..=${res.data.allowed_range.max}%`
        : `Factory Reset ${res.data.applied ? "APPLIED" : "DRY-RUN"}: was_at_factory=${res.data.was_at_factory}, now_at_factory=${res.data.now_at_factory}`;
    powerPanelContent.appendChild(
      el("div", { className: "banner banner-ok mono", style: "margin-top:10px;margin-bottom:0" }, text)
    );
  }
  if (state.tuningUi.actionError) {
    powerPanelContent.appendChild(
      el("div", { className: "banner banner-err mono", style: "margin-top:10px;margin-bottom:0" }, state.tuningUi.actionError)
    );
  }

  // Clocks & VRAM Readonly Panel
  const gfx = rep.gfx;
  const vram = rep.vram;
  const clockRows = [
    ["GFX Min Frequency", gfx?.min_freq_mhz != null ? `${gfx.min_freq_mhz} MHz (${fmtRange(gfx.min_freq_range)})` : "n/a"],
    ["GFX Max Frequency", gfx?.max_freq_mhz != null ? `${gfx.max_freq_mhz} MHz (${fmtRange(gfx.max_freq_range)})` : "n/a"],
    ["GFX Voltage", gfx?.voltage_mv != null ? `${gfx.voltage_mv} mV (${fmtRange(gfx.voltage_range)})` : "n/a"],
    ["VRAM Max Frequency", vram?.max_freq_mhz != null ? `${vram.max_freq_mhz} MHz (${fmtRange(vram.max_freq_range)})` : "n/a"],
    ["Memory Timing Mode", vram?.memory_timing_supported ? String(vram.memory_timing ?? "n/a") : "Unsupported"],
    ["Supported Presets", rep.preset?.supported_presets?.join(", ") || "None", false],
  ];

  // Right: Fan Curve & Canvas
  const fan = rep.fan;
  const fanContent = el("div", {});
  if (!fan) {
    fanContent.appendChild(el("div", { className: "muted-text" }, "Manual fan control is not exposed for this adapter."));
  } else {
    fanContent.appendChild(
      makeKvTable([
        ["Zero-RPM State", fan.zero_rpm_supported ? `${fan.zero_rpm_enabled ? "Enabled" : "Disabled"} (default ${fan.zero_rpm_default})` : "Unsupported"],
        ["Min Acoustic Limit", fan.min_acoustic_limit_mhz != null ? `${fan.min_acoustic_limit_mhz} MHz` : "n/a"],
        ["Fan Speed Range", fmtRange(fan.speed_range_pct, "%")],
        ["Temperature Range", fmtRange(fan.temp_range_c, "C")],
      ])
    );

    if (fan.fan_curve && fan.fan_curve.length > 0) {
      const fanCanvas = el("canvas", { className: "fan-canvas", style: "margin-top:10px" });
      fanContent.appendChild(fanCanvas);
      fanContent.appendChild(
        el(
          "table",
          { className: "data-table", style: "margin-top:8px" },
          el(
            "thead",
            {},
            el("tr", {}, el("th", {}, "Point"), el("th", { className: "num" }, "Temperature"), el("th", { className: "num" }, "Fan PWM"))
          ),
          el(
            "tbody",
            {},
            fan.fan_curve.map((pt, idx) =>
              el(
                "tr",
                {},
                el("td", { className: "mono muted-text" }, `P${idx + 1}`),
                el("td", { className: "num mono" }, `${pt.temp_c} C`),
                el("td", { className: "num mono" }, `${pt.speed_pct} %`)
              )
            )
          )
        )
      );
      requestAnimationFrame(() => drawFanCurveChart(fanCanvas, fan.fan_curve, fan.temp_range_c, fan.speed_range_pct));
    }
  }

  root.appendChild(
    el(
      "div",
      { className: "grid-asymmetric" },
      el(
        "div",
        { style: "display:flex;flex-direction:column;gap:12px" },
        makePanel("Power Limit & Hardware Safety Gate", "Read-only / Dry-run default", powerPanelContent),
        makePanel("GFX & VRAM Tuning State (Read-Only)", "Hardware inspected", makeKvTable(clockRows))
      ),
      makePanel("Fan Curve & Acoustic Telemetry", "ADLX ManualFanTuning", fanContent)
    )
  );
}

/* ---------------- VIDEO (AMF) TAB ---------------- */

export function renderVideo(root, state, actions) {
  root.replaceChildren();

  root.appendChild(
    el(
      "div",
      { className: "toolbar" },
      el(
        "div",
        { className: "toolbar-group" },
        el(
          "button",
          {
            type: "button",
            className: "btn btn-primary",
            disabled: state.loadingVideo,
            onClick: actions.refreshVideoCaps,
          },
          state.loadingVideo ? "Probing AMF..." : "Probe VCN Codecs"
        )
      ),
      el(
        "div",
        { className: "toolbar-group" },
        el(
          "button",
          {
            type: "button",
            className: "btn",
            disabled: !state.videoCaps,
            onClick: actions.exportVideoJson,
          },
          "Export JSON"
        )
      )
    )
  );

  if (state.videoError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.videoError));
  }

  const caps = state.videoCaps;
  if (!caps) {
    root.appendChild(el("div", { className: "empty-state" }, "Probing amfrt64.dll via DX11 COM..."));
    return;
  }

  const hwEnc = caps.codecs.filter((c) => c.direction === "encode" && c.acceleration === "hardware").length;
  const hwDec = caps.codecs.filter((c) => c.direction === "decode" && c.acceleration === "hardware").length;

  root.appendChild(
    el(
      "div",
      { className: "kpi-strip" },
      el("div", { className: "kpi-tile" }, el("div", { className: "kpi-label" }, "AMF Runtime"), el("div", { className: "kpi-value" }, caps.runtime_version)),
      el("div", { className: "kpi-tile" }, el("div", { className: "kpi-label" }, "DX11 Context"), el("div", { className: "kpi-value" }, caps.dx11_initialized ? "READY" : "FAILED")),
      el("div", { className: "kpi-tile" }, el("div", { className: "kpi-label" }, "HW Encoders"), el("div", { className: "kpi-value" }, String(hwEnc))),
      el("div", { className: "kpi-tile" }, el("div", { className: "kpi-label" }, "HW Decoders"), el("div", { className: "kpi-value" }, String(hwDec)))
    )
  );

  const codecRows = caps.codecs.map((c) => {
    const resRange =
      c.min_width != null && c.max_width != null
        ? `${c.min_width}x${c.min_height}..${c.max_width}x${c.max_height}`
        : "n/a";
    const tenBit = (c.supported_formats || []).some((f) =>
      ["P010", "P012", "P016", "Y210", "Y410"].includes(f)
    );
    return el(
      "tr",
      {},
      el("td", { className: "mono" }, c.codec, el("div", { className: "sub-detail mono" }, c.component_id)),
      el("td", { className: "mono" }, c.direction === "encode" ? "Encode" : "Decode"),
      el("td", {}, makeAmfAccelBadge(c.acceleration)),
      el("td", { className: "mono" }, resRange),
      el("td", { className: "mono num" }, c.vert_align ?? "--"),
      el("td", { className: "mono" }, tenBit ? "yes" : "no"),
      el("td", { className: "mono" }, c.interlaced_supported == null ? "n/a" : c.interlaced_supported ? "yes" : "no"),
      el("td", { className: "mono" }, c.native_formats?.length ? c.native_formats.join(", ") : "--"),
      el(
        "td",
        { className: "mono muted-text" },
        c.detail ? c.detail : c.supported_formats?.length ? c.supported_formats.join(", ") : "--"
      )
    );
  });

  const codecTable = el(
    "div",
    { className: "data-table-wrap" },
    el(
      "table",
      { className: "data-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Codec / Component"),
          el("th", {}, "Dir"),
          el("th", {}, "Acceleration"),
          el("th", {}, "Resolution Range"),
          el("th", { className: "num" }, "Align"),
          el("th", {}, "10-bit"),
          el("th", {}, "Interlaced"),
          el("th", {}, "Native Formats"),
          el("th", {}, "Supported Formats / Detail")
        )
      ),
      el("tbody", {}, codecRows)
    )
  );

  root.appendChild(makePanel("AMD AMF Hardware Video Capabilities", `${caps.codecs.length} components probed`, codecTable, true));
}

/* ---------------- BENCHMARKS TAB ---------------- */

const KERNEL_OPTIONS = [
  ["gemv_w8", "gemv_w8 (Wave32 W8A32 GEMV)"],
  ["gemv_q4", "gemv_q4 (Wave32 Q4_0 GEMV)"],
  ["gemv_q4_k", "gemv_q4_k (Wave32 GGUF Q4_K GEMV)"],
  ["fused_rms_rope", "fused_rms_rope (Fused Subgroup RMSNorm+RoPE)"],
  ["flash_attn", "flash_attn (Wave32 Tiled Causal FlashAttention)"],
  ["rmsnorm", "rmsnorm (FP32 RMSNorm)"],
  ["rope", "rope (FP32 RoPE)"],
  ["gemm_opt", "gemm_opt (Tiled vec4 FP32 GEMM)"],
  ["gemm_i8", "gemm_i8 (INT8 DP4A OpSDot GEMM)"],
  ["gemm", "gemm (Baseline FP32 GEMM)"],
  ["all", "all (Full Vulkan Compute Suite)"],
];

export function renderBenchmarks(root, state, actions) {
  root.replaceChildren();
  const cfg = state.benchForm;
  const gpus = state.diagnostics?.report?.gpus || [];

  const formNode = el(
    "div",
    { className: "form-row" },
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Device Index"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.runningBench,
          onChange: (e) => actions.updateBenchField("device_index", Number(e.target.value)),
        },
        gpus.length
          ? gpus.map((g) =>
              el(
                "option",
                { value: String(g.index), selected: cfg.device_index === g.index },
                `#${g.index} ${g.name}`
              )
            )
          : el("option", { value: "0", selected: true }, "#0 Primary Vulkan GPU")
      )
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Vulkan Kernel"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.runningBench,
          onChange: (e) => actions.updateBenchField("kernel", e.target.value),
        },
        KERNEL_OPTIONS.map(([val, label]) =>
          el("option", { value: val, selected: cfg.kernel === val }, label)
        )
      )
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Batch (1..256)"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:84px",
        min: "1",
        max: "256",
        disabled: state.runningBench,
        value: String(cfg.batch_dispatches ?? ""),
        placeholder: "auto (16)",
        onChange: (e) =>
          actions.updateBenchField(
            "batch_dispatches",
            e.target.value.trim() === "" ? null : Number(e.target.value)
          ),
      })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Warmup"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:70px",
        min: "0",
        max: "100",
        disabled: state.runningBench,
        value: String(cfg.warmup_iters),
        onChange: (e) => actions.updateBenchField("warmup_iters", Number(e.target.value)),
      })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Iters"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:70px",
        min: "1",
        max: "200",
        disabled: state.runningBench,
        value: String(cfg.measured_iters),
        onChange: (e) => actions.updateBenchField("measured_iters", Number(e.target.value)),
      })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Stability"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:74px",
        min: "0",
        max: "200",
        disabled: state.runningBench,
        value: String(cfg.stability_runs),
        onChange: (e) => actions.updateBenchField("stability_runs", Number(e.target.value)),
      })
    ),
    el(
      "div",
      { className: "form-field", style: "align-self:flex-end" },
      el(
        "button",
        {
          type: "button",
          className: "btn btn-primary",
          disabled: state.runningBench || state.busyOp,
          onClick: actions.runBenchmark,
        },
        state.runningBench ? "Running Benchmark..." : "Run Benchmark"
      )
    ),
    el(
      "div",
      { className: "form-field", style: "align-self:flex-end" },
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: !state.benchReport,
          onClick: actions.exportBenchJson,
        },
        "Export JSON"
      )
    )
  );

  root.appendChild(
    makePanel(
      "Vulkan Compute Benchmark Runner",
      "Decode/elementwise default batch=16; GEMM stays batch=1 unless overridden",
      formNode
    )
  );

  if (state.benchError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.benchError));
  }

  const report = state.benchReport;
  if (!report) {
    root.appendChild(
      el(
        "div",
        { className: "empty-state" },
        "Select a Vulkan compute kernel and click 'Run Benchmark' to execute timed GPU dispatches with CPU reference validation."
      )
    );
    return;
  }

  const rows = report.benchmarks.map((b) => {
    const l = b.latency;
    const acc = b.accuracy;
    return el(
      "tr",
      {},
      el(
        "td",
        { className: "mono" },
        b.name,
        b.notes ? el("div", { className: "sub-detail" }, b.notes) : null
      ),
      el("td", { className: "mono" }, b.shape),
      el("td", { className: "mono" }, b.dtype),
      el("td", { className: "num mono" }, `${l.median_ms.toFixed(4)} ms`),
      el(
        "td",
        { className: "num mono muted-text" },
        `${l.min_ms.toFixed(4)} / ${l.p95_ms.toFixed(4)} / ${l.max_ms.toFixed(4)}`
      ),
      el(
        "td",
        { className: "num mono muted-text" },
        `${l.iqr_ms.toFixed(4)} / ${l.mad_ms.toFixed(4)}`
      ),
      el(
        "td",
        { className: "num mono" },
        `${b.throughput_median.toFixed(2)} ${b.throughput_unit}`
      ),
      el("td", { className: "num mono muted-text" }, acc ? acc.max_abs_err.toExponential(2) : "--"),
      el("td", { className: "num mono muted-text" }, acc ? acc.rmse.toExponential(2) : "--"),
      el(
        "td",
        {},
        acc
          ? acc.passed
            ? el("span", { className: "badge badge-ok" }, "PASS")
            : el("span", { className: "badge badge-err" }, "FAIL")
          : el("span", { className: "badge badge-idle" }, "N/A")
      )
    );
  });

  const table = el(
    "div",
    { className: "data-table-wrap" },
    el(
      "table",
      { className: "data-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Kernel & Notes"),
          el("th", {}, "Shape"),
          el("th", {}, "Dtype"),
          el("th", { className: "num" }, "Median"),
          el("th", { className: "num" }, "Min / P95 / Max (ms)"),
          el("th", { className: "num" }, "IQR / MAD (ms)"),
          el("th", { className: "num" }, "Throughput"),
          el("th", { className: "num" }, "Max Abs Err"),
          el("th", { className: "num" }, "RMSE"),
          el("th", {}, "Ref")
        )
      ),
      el("tbody", {}, rows)
    )
  );
  root.appendChild(
    makePanel(
      `Measured Results on ${report.device.name}`,
      `Peak RSS: ${report.peak_rss_bytes ? (report.peak_rss_bytes / (1024 ** 2)).toFixed(1) + " MiB" : "n/a"}`,
      table,
      true
    )
  );

  if (report.telemetry) {
    const t = report.telemetry;
    const telRows = [
      ["ADLX Samples", `${t.sample_count} (${t.interval_ms} ms)`],
      ["GPU Clock (med / max)", `${fmtNum(t.gpu_clock_mhz?.median, 0)} / ${fmtNum(t.gpu_clock_mhz?.max, 0, "MHz")}`],
      ["VRAM Clock (med / max)", `${fmtNum(t.vram_clock_mhz?.median, 0)} / ${fmtNum(t.vram_clock_mhz?.max, 0, "MHz")}`],
      ["Core / Hotspot Temp (max)", `${fmtNum(t.gpu_temp_c?.max, 1, "C")} / ${fmtNum(t.hotspot_temp_c?.max, 1, "C")}`],
      ["GPU Power (med / max)", `${fmtNum(t.gpu_power_w?.median, 1)} / ${fmtNum(t.gpu_power_w?.max, 1, "W")}`],
      ["VRAM Used (max)", t.vram_used_mib ? `${Math.round(t.vram_used_mib.max)} / ${t.total_vram_mib ?? "--"} MiB` : "n/a"],
    ];
    root.appendChild(makePanel("Run ADLX Telemetry Summary", t.gpu_name, makeKvTable(telRows)));
    for (const w of t.warnings || []) {
      root.appendChild(el("div", { className: "banner banner-warn" }, w));
    }
  }

  if (report.stability) {
    const st = report.stability;
    root.appendChild(
      makePanel(
        "Repeated Workload Stability",
        `${st.iterations} iterations`,
        makeKvTable([
          ["Workload", st.workload],
          ["Failures", String(st.failures)],
          ["All Finite / Correct", `${st.all_finite} / ${st.all_correct}`],
          ["Median Latency", `${st.latency.median_ms.toFixed(4)} ms`],
        ])
      )
    );
  }
}
