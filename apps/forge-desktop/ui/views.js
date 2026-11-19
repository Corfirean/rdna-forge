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
    const pwmMode =
      fan.flat_pwm_override_pct != null
        ? `Flat Override (${fan.flat_pwm_override_pct}% across all 5 points)`
        : "5-Point Curve";
    fanContent.appendChild(
      makeKvTable([
        ["Zero-RPM State", fan.zero_rpm_supported ? `${fan.zero_rpm_enabled ? "Enabled" : "Disabled"} (default ${fan.zero_rpm_default})` : "Unsupported"],
        ["External Fan Manager", fan.external_controller || "None detected"],
        ["ADLX PWM Mode", pwmMode],
        ["Fan Speed Range", fmtRange(fan.speed_range_pct, "%")],
        ["Temperature Range", fmtRange(fan.temp_range_c, "C")],
      ])
    );

    if (fan.fan_note) {
      fanContent.appendChild(
        el("div", { className: "banner banner-warn", style: "margin-top:8px;margin-bottom:4px" }, fan.fan_note)
      );
    }

    if (fan.fan_curve && fan.fan_curve.length > 0) {
      const fanCanvas = el("canvas", { className: "fan-canvas", style: "margin-top:10px" });
      fanContent.appendChild(fanCanvas);
      const defCurve = fan.default_fan_curve || [];
      fanContent.appendChild(
        el(
          "table",
          { className: "data-table", style: "margin-top:8px" },
          el(
            "thead",
            {},
            el(
              "tr",
              {},
              el("th", {}, "Point"),
              el("th", { className: "num" }, "Temperature"),
              el("th", { className: "num" }, "Active ADLX PWM"),
              el("th", { className: "num" }, "Factory VBIOS PWM")
            )
          ),
          el(
            "tbody",
            {},
            fan.fan_curve.map((pt, idx) => {
              const defPt = defCurve[idx];
              return el(
                "tr",
                {},
                el("td", { className: "mono muted-text" }, `P${idx + 1}`),
                el("td", { className: "num mono" }, `${pt.temp_c} C`),
                el("td", { className: "num mono" }, `${pt.speed_pct} %`),
                el("td", { className: "num mono muted-text" }, defPt ? `${defPt.speed_pct} % (${defPt.temp_c} C)` : "--")
              );
            })
          )
        )
      );
      requestAnimationFrame(() =>
        drawFanCurveChart(fanCanvas, fan.fan_curve, fan.temp_range_c, fan.speed_range_pct, defCurve)
      );
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

  const vcfg = state.videoBenchForm || {
    codec: "all",
    resolution: "1920x1080",
    fps: 60,
    bitrate_mbps: 12,
    preset: "speed",
    warmup_frames: 15,
    measured_frames: 120,
  };

  const benchFormNode = el(
    "div",
    { className: "form-row" },
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "VCN Encoder Codec"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.runningVideoBench,
          onChange: (e) => actions.updateVideoBenchField("codec", e.target.value),
        },
        [
          ["all", "all (H.264 + HEVC + HEVC 10-bit)"],
          ["h264", "h264 (AMFVideoEncoderVCE_AVC NV12)"],
          ["hevc", "hevc (AMFVideoEncoderHW_HEVC NV12)"],
          ["hevc10", "hevc10 (AMFVideoEncoderHW_HEVC P010)"],
        ].map(([val, label]) =>
          el("option", { value: val, selected: vcfg.codec === val }, label)
        )
      )
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Resolution"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.runningVideoBench,
          onChange: (e) => actions.updateVideoBenchField("resolution", e.target.value),
        },
        [
          ["1280x720", "1280x720 (720p HD)"],
          ["1920x1080", "1920x1080 (1080p FHD)"],
          ["2560x1440", "2560x1440 (1440p QHD)"],
          ["3840x2160", "3840x2160 (4K UHD)"],
        ].map(([val, label]) =>
          el("option", { value: val, selected: vcfg.resolution === val }, label)
        )
      )
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Preset"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.runningVideoBench,
          onChange: (e) => actions.updateVideoBenchField("preset", e.target.value),
        },
        [
          ["speed", "speed"],
          ["balanced", "balanced"],
          ["quality", "quality"],
        ].map(([val, label]) =>
          el("option", { value: val, selected: vcfg.preset === val }, label)
        )
      )
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Bitrate (Mbps)"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:78px",
        min: "1",
        max: "500",
        disabled: state.runningVideoBench,
        value: String(vcfg.bitrate_mbps),
        onChange: (e) => actions.updateVideoBenchField("bitrate_mbps", Number(e.target.value)),
      })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Warmup"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:68px",
        min: "0",
        max: "120",
        disabled: state.runningVideoBench,
        value: String(vcfg.warmup_frames),
        onChange: (e) => actions.updateVideoBenchField("warmup_frames", Number(e.target.value)),
      })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Frames"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:74px",
        min: "10",
        max: "600",
        disabled: state.runningVideoBench,
        value: String(vcfg.measured_frames),
        onChange: (e) => actions.updateVideoBenchField("measured_frames", Number(e.target.value)),
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
          disabled: state.runningVideoBench || state.busyOp,
          onClick: actions.runVideoBenchmark,
        },
        state.runningVideoBench ? "Encoding on VCN..." : "Run Encode Benchmark"
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
          disabled: !state.videoBenchReport,
          onClick: actions.exportVideoBenchJson,
        },
        "Export Bench JSON"
      )
    )
  );

  root.appendChild(
    makePanel(
      "VCN Hardware Encode Benchmark Runner",
      "Pre-uploads 16 DX11 surfaces to VRAM before timing VCN hardware encode",
      benchFormNode
    )
  );

  if (state.videoBenchError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.videoBenchError));
  }

  const vrep = state.videoBenchReport;
  if (vrep && Array.isArray(vrep.results)) {
    const vrows = vrep.results.map((r) =>
      el(
        "tr",
        {},
        el(
          "td",
          { className: "mono" },
          r.codec,
          el("div", { className: "sub-detail mono" }, r.component_id)
        ),
        el("td", { className: "mono" }, r.surface_format),
        el("td", { className: "mono" }, `${r.width}x${r.height}`),
        el("td", { className: "mono" }, r.preset),
        el("td", { className: "num mono" }, `${r.encode_fps.toFixed(2)} FPS`),
        el("td", { className: "num mono" }, `${r.megapixels_per_sec.toFixed(2)} MPix/s`),
        el("td", { className: "num mono muted-text" }, `${r.avg_frame_ms.toFixed(3)} ms`),
        el("td", { className: "num mono muted-text" }, `${r.effective_bitrate_mbps.toFixed(2)} Mbps`),
        el("td", { className: "num mono muted-text" }, `${r.output_packets}/${r.submitted_frames}`),
        el(
          "td",
          {},
          r.valid_annex_b
            ? el("span", { className: "badge badge-ok" }, "VALID")
            : el("span", { className: "badge badge-err" }, "INVALID")
        )
      )
    );

    const vtable = el(
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
            el("th", {}, "Format"),
            el("th", {}, "Resolution"),
            el("th", {}, "Preset"),
            el("th", { className: "num" }, "Encode FPS"),
            el("th", { className: "num" }, "Throughput"),
            el("th", { className: "num" }, "Avg Frame"),
            el("th", { className: "num" }, "Effective Bitrate"),
            el("th", { className: "num" }, "Packets"),
            el("th", {}, "Annex-B NAL")
          )
        ),
        el("tbody", {}, vrows)
      )
    );

    root.appendChild(
      makePanel(
        `VCN Hardware Encode Results (AMF ${vrep.runtime_version})`,
        vrep.peak_rss_bytes ? `Peak RSS: ${(vrep.peak_rss_bytes / (1024 ** 2)).toFixed(1)} MiB` : "",
        vtable,
        true
      )
    );
  }
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

/* ---------------- LLM OPTIMIZER TAB ---------------- */

export function renderLlmOptimizer(root, state, actions) {
  root.replaceChildren();
  const form = state.llmForm || {
    model_path: "",
    context_tokens: 4096,
    safety_margin_mib: 512,
    context_depth: 2048,
    prompt_tokens: 128,
    gen_tokens: 32,
  };
  const assets = state.llmAssets || { gguf_models: [] };

  const modelControls = el(
    "div",
    { className: "form-row" },
    el(
      "div",
      { className: "form-field", style: "flex:1;min-width:260px" },
      el("label", { className: "field-label" }, "Discovered Local GGUF Model"),
      assets.gguf_models.length > 0
        ? el(
            "select",
            {
              className: "select-input mono",
              style: "width:100%",
              disabled: state.loadingLlmPlan || state.runningLlmAutotune,
              onChange: (e) => actions.updateLlmField("model_path", e.target.value, true),
            },
            assets.gguf_models.map((p) => {
              const parts = p.split(/[\\/]/);
              const shortName = parts[parts.length - 1] || p;
              return el(
                "option",
                { value: p, selected: form.model_path === p },
                shortName
              );
            })
          )
        : el("input", {
            type: "text",
            className: "text-input mono",
            style: "width:100%",
            placeholder: "C:\\path\\to\\model.gguf",
            value: form.model_path || "",
            onChange: (e) => actions.updateLlmField("model_path", e.target.value, false),
          })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Target Context"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.loadingLlmPlan || state.runningLlmAutotune,
          onChange: (e) => actions.updateLlmField("context_tokens", Number(e.target.value), true),
        },
        [2048, 4096, 8192, 16384, 32768].map((ctx) =>
          el("option", { value: String(ctx), selected: Number(form.context_tokens) === ctx }, `${ctx} tokens`)
        )
      )
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Safety Margin (MiB)"),
      el("input", {
        type: "number",
        className: "text-input mono",
        style: "width:84px",
        min: "0",
        max: "4096",
        disabled: state.loadingLlmPlan || state.runningLlmAutotune,
        value: String(form.safety_margin_mib),
        onChange: (e) => actions.updateLlmField("safety_margin_mib", Number(e.target.value), true),
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
          disabled: state.loadingLlmPlan || state.runningLlmAutotune,
          onClick: actions.refreshLlmPlan,
        },
        state.loadingLlmPlan ? "Analyzing GGUF..." : "Analyze & Plan VRAM"
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
          disabled: !state.llmPlan,
          onClick: actions.exportLlmPlanJson,
        },
        "Export Plan JSON"
      )
    )
  );

  root.appendChild(
    makePanel(
      "GGUF Model Selector & Anti-WDDM-Spill Planner",
      "Reads GGUF v2/v3 header & live ADLX VRAM occupancy to prevent silent Windows PCIe spilling",
      modelControls
    )
  );

  if (state.llmError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.llmError));
  }

  const plan = state.llmPlan;
  if (!plan) {
    root.appendChild(
      el(
        "div",
        { className: "empty-state" },
        state.loadingLlmPlan
          ? "Inspecting GGUF model header and live ADLX WDDM VRAM..."
          : "Select a local GGUF model and click 'Analyze & Plan VRAM' to calculate non-spilling GPU layer budgets and RDNA2 tweaks."
      )
    );
    return;
  }

  for (const w of plan.warnings || []) {
    root.appendChild(el("div", { className: "banner banner-warn" }, w));
  }

  const m = plan.model;
  const gib = (m.file_size_bytes / (1024 ** 3)).toFixed(2);
  const mibPerLayer = (m.bytes_per_layer / (1024 ** 2)).toFixed(1);

  root.appendChild(
    el(
      "div",
      { className: "kpi-strip" },
      el(
        "div",
        { className: "kpi-tile" },
        el("div", { className: "kpi-label" }, "Model & Quant"),
        el("div", { className: "kpi-value" }, m.model_name || m.file_name),
        el("div", { className: "kpi-sub" }, `${m.quantization_name} (${gib} GiB)`)
      ),
      el(
        "div",
        { className: "kpi-tile" },
        el("div", { className: "kpi-label" }, "Architecture & Layers"),
        el(
          "div",
          { className: "kpi-value" },
          m.is_moe
            ? `${m.architecture} MoE (${m.block_count}L, ${m.expert_used_count ?? "?"}/${m.expert_count}E)`
            : `${m.architecture} (${m.block_count}L)`
        ),
        el(
          "div",
          { className: "kpi-sub" },
          m.full_attention_interval
            ? `Hybrid SSM+Attn (1/${m.full_attention_interval} full attn, ~${mibPerLayer} MiB/L)`
            : `~${mibPerLayer} MiB/layer`
        )
      ),
      el(
        "div",
        { className: "kpi-tile" },
        el("div", { className: "kpi-label" }, "Free Physical VRAM"),
        el("div", { className: "kpi-value" }, `${plan.free_vram_mib} MiB`),
        el("div", { className: "kpi-sub" }, `Used: ${plan.used_vram_mib} / ${plan.total_vram_mib} MiB`)
      ),
      el(
        "div",
        { className: "kpi-tile" },
        el("div", { className: "kpi-label" }, "VRAM App Recovery"),
        el("div", { className: "kpi-value" }, `+${plan.recoverable_layers_if_apps_closed} layers`),
        el("div", { className: "kpi-sub" }, `Physical CPU Threads: -t ${plan.physical_cpu_cores}`)
      )
    )
  );

  // KV Cache Memory Budget Table
  const kvRows = (plan.kv_budgets || []).map((b) =>
    el(
      "tr",
      {},
      el("td", { className: "mono" }, b.kv_quant),
      el("td", { className: "mono" }, b.requires_flash_attn ? "-fa 1 required" : "optional"),
      el("td", { className: "num mono" }, `${b.model_weights_mib} MiB`),
      el("td", { className: "num mono" }, `${b.kv_and_state_mib} MiB`),
      el("td", { className: "num mono" }, `${b.total_required_mib} MiB`),
      el(
        "td",
        {},
        b.fits_in_free_vram
          ? el("span", { className: "badge badge-ok" }, "FITS FREE VRAM")
          : el("span", { className: "badge badge-warn" }, "EXCEEDS FREE VRAM")
      ),
      el(
        "td",
        {},
        b.fits_with_wddm_memory_priority
          ? el("span", { className: "badge badge-ok" }, "FITS WITH PRIO=1")
          : el("span", { className: "badge badge-warn" }, "PARTIAL OFFLOAD")
      ),
      el(
        "td",
        { className: "num mono" },
        b.recommended_n_cpu_moe > 0
          ? `${b.strict_zero_spill_ngl} / ${b.memory_priority_ngl} of ${m.block_count} (-ncmoe ${b.recommended_n_cpu_moe})`
          : `${b.strict_zero_spill_ngl} / ${b.memory_priority_ngl} of ${m.block_count}`
      )
    )
  );

  root.appendChild(
    makePanel(
      `KV Cache & VRAM Fit Analysis (@ ${plan.target_context_tokens} Context Tokens)`,
      "Strict NGL fits unallocated VRAM; Priority NGL uses GGML_VK_ENABLE_MEMORY_PRIORITY=1",
      el(
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
              el("th", {}, "KV Cache Type"),
              el("th", {}, "Flash Attention"),
              el("th", { className: "num" }, "Weights"),
              el("th", { className: "num" }, "KV + SSM State"),
              el("th", { className: "num" }, "Total Required"),
              el("th", {}, "Unallocated VRAM Fit"),
              el("th", {}, "WDDM Priority=1 Fit"),
              el("th", { className: "num" }, "Strict / Prio NGL")
            )
          ),
          el("tbody", {}, kvRows)
        )
      ),
      true
    )
  );

  // Recommended RDNA2 Profiles & One-Click Copy / LM Studio Patcher Commands
  const profileCards = (plan.recommended_profiles || []).map((p) => {
    const envStr = (p.env_vars || []).map(([k, v]) => `${k}=${v}`).join(" | ") || "none";
    return el(
      "div",
      {
        style:
          "border:1px solid #2a2e35;background:#16181b;padding:10px 12px;margin-bottom:8px;display:flex;flex-direction:column;gap:6px",
      },
      el(
        "div",
        { style: "display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap" },
        el("strong", { style: "color:#e6e8ec;font-size:12.5px" }, p.title),
        el(
          "div",
          { style: "display:flex;gap:6px;align-items:center;flex-wrap:wrap" },
          el("span", { className: "badge badge-ok mono" }, `${p.backend.toUpperCase()} (${p.device_selector})`),
          el("span", { className: "badge badge-idle mono" }, `ngl=${p.n_gpu_layers}`),
          p.n_cpu_moe > 0
            ? el("span", { className: "badge badge-warn mono" }, `ncmoe=${p.n_cpu_moe}`)
            : null,
          el("span", { className: "badge badge-idle mono" }, `fa=${p.flash_attn ? "on" : "off"}`),
          el("span", { className: "badge badge-idle mono" }, `kv=${p.cache_type_k}/${p.cache_type_v}`),
          el(
            "button",
            {
              type: "button",
              className: "btn",
              disabled: state.patchingLlm,
              onClick: () => actions.applyLmStudioPatch(p.id, true),
            },
            "Preview LM Studio Patch"
          ),
          el(
            "button",
            {
              type: "button",
              className: "btn btn-primary",
              disabled: state.patchingLlm || !form.unlockConfigWrites,
              onClick: () => actions.applyLmStudioPatch(p.id, false),
            },
            "Apply to LM Studio"
          ),
          el(
            "button",
            {
              type: "button",
              className: "btn",
              onClick: () => actions.copyLlmCommand(p.llama_server_cmd),
            },
            "Copy Launch Cmd"
          )
        )
      ),
      el("div", { className: "muted-text", style: "font-size:11.5px;line-height:1.4" }, p.rationale),
      el("div", { className: "mono muted-text", style: "font-size:11px" }, `Env Tweaks: ${envStr}`),
      el(
        "pre",
        {
          className: "mono",
          style:
            "margin:2px 0 0 0;padding:7px 9px;background:#111214;border:1px solid #24272d;font-size:11px;overflow-x:auto;color:#cfd4dc",
        },
        p.llama_server_cmd
      )
    );
  });

  root.appendChild(
    makePanel(
      `Verified AMD RDNA2 (${plan.gfx_target}) Tweaks & Launch Profiles`,
      `${plan.gpu_name}`,
      el("div", {}, profileCards)
    )
  );

  // 1-Click LM Studio, Ollama & Windows Environment Patcher Panel
  const patchPanelBody = el(
    "div",
    { style: "display:flex;flex-direction:column;gap:10px" },
    el(
      "div",
      { className: "toolbar-group" },
      el(
        "label",
        { className: "checkbox-label" },
        el("input", {
          type: "checkbox",
          checked: Boolean(form.unlockConfigWrites),
          onChange: (e) => actions.updateLlmField("unlockConfigWrites", e.target.checked, false),
        }),
        "Unlock Config & Registry Writes (creates .bak backups before modifying LM Studio JSONs)"
      ),
      el(
        "button",
        {
          type: "button",
          className: "btn",
          disabled: state.patchingLlm,
          onClick: () => actions.applyEnvPatch(true),
        },
        "Preview Windows & Ollama Env (Dry-Run)"
      ),
      el(
        "button",
        {
          type: "button",
          className: "btn btn-primary",
          disabled: state.patchingLlm || !form.unlockConfigWrites,
          onClick: () => actions.applyEnvPatch(false),
        },
        "Apply Windows & Ollama Env (HKCU)"
      )
    )
  );

  if (state.llmPatchError) {
    patchPanelBody.appendChild(el("div", { className: "banner banner-err mono" }, state.llmPatchError));
  }

  if (state.llmPatchResult) {
    const pr = state.llmPatchResult;
    if (pr.kind === "lmstudio") {
      const d = pr.data;
      patchPanelBody.appendChild(
        el(
          "div",
          {
            className: `banner ${d.applied ? "banner-ok" : "banner-warn"} mono`,
            style: "margin-bottom:0",
          },
          `[${d.applied ? "APPLIED" : "DRY-RUN PREVIEW"}] Profile: ${d.profile_id} (${d.profile_title}) | Runtime: ${d.selected_runtime}`
        )
      );
      patchPanelBody.appendChild(
        makeKvTable([
          ["Model Config Path", d.model_config_path],
          ["Model Backup (.bak)", d.model_backup_path || "(none — new file or dry-run)"],
          ["Backend Preferences Path", d.backend_prefs_path],
          ["Backend Backup (.bak)", d.backend_backup_path || "(none)"],
        ])
      );
      patchPanelBody.appendChild(
        el(
          "pre",
          {
            className: "mono",
            style:
              "margin:0;padding:8px 10px;background:#111214;border:1px solid #24272d;font-size:11px;max-height:220px;overflow:auto;color:#cfd4dc",
          },
          d.model_config_json
        )
      );
    } else if (pr.kind === "env") {
      const d = pr.data;
      patchPanelBody.appendChild(
        el(
          "div",
          {
            className: `banner ${d.applied ? "banner-ok" : "banner-warn"} mono`,
            style: "margin-bottom:0",
          },
          `[${d.applied ? "APPLIED TO HKCU\\Environment" : "DRY-RUN PREVIEW"}] ${d.note}`
        )
      );
      const envRows = (d.env_changes || []).map((c) =>
        el(
          "tr",
          {},
          el("td", { className: "mono" }, c.name),
          el("td", { className: "mono muted-text" }, c.previous_value || "(unset)"),
          el("td", { className: "mono" }, c.new_value),
          el("td", { className: "muted-text" }, c.purpose)
        )
      );
      patchPanelBody.appendChild(
        el(
          "table",
          { className: "data-table" },
          el(
            "thead",
            {},
            el(
              "tr",
              {},
              el("th", {}, "Variable (HKCU\\Environment)"),
              el("th", {}, "Previous"),
              el("th", {}, "Target"),
              el("th", {}, "RDNA2 Purpose")
            )
          ),
          el("tbody", {}, envRows)
        )
      );
      patchPanelBody.appendChild(
        el(
          "div",
          { style: "display:flex;justify-content:space-between;align-items:center" },
          el("span", { className: "mono muted-text", style: "font-size:11.5px" }, "Generated Ollama Modelfile:"),
          el(
            "button",
            {
              type: "button",
              className: "btn",
              onClick: () => actions.copyLlmCommand(d.ollama_modelfile),
            },
            "Copy Ollama Modelfile"
          )
        )
      );
      patchPanelBody.appendChild(
        el(
          "pre",
          {
            className: "mono",
            style:
              "margin:0;padding:8px 10px;background:#111214;border:1px solid #24272d;font-size:11px;max-height:180px;overflow:auto;color:#cfd4dc",
          },
          d.ollama_modelfile
        )
      );
    }
  }

  root.appendChild(
    makePanel(
      "1-Click LM Studio, Ollama & Windows Environment Auto-Patcher",
      "Dry-Run by default; automatically creates .bak backups and preserves custom Prompt Templates",
      patchPanelBody
    )
  );

  // LM Studio Checklist
  if (plan.lm_studio_tips && plan.lm_studio_tips.length > 0) {
    root.appendChild(
      makePanel(
        "LM Studio & Local Launcher Optimization Checklist",
        "Applies directly to your installed LM Studio 2.73.0 Vulkan & ROCm backends",
        el(
          "div",
          { style: "display:flex;flex-direction:column;gap:6px" },
          plan.lm_studio_tips.map((tip) =>
            el("div", { style: "font-size:12px;line-height:1.45;color:#cfd4dc" }, tip)
          )
        )
      )
    );
  }

  // Empirical llama-bench Autotuner Card
  const autotuneControls = el(
    "div",
    { className: "form-row" },
    el(
      "div",
      { className: "form-field", style: "flex:1;min-width:210px" },
      el("label", { className: "field-label" }, "Vulkan llama-bench.exe Path"),
      el("input", {
        type: "text",
        className: "text-input mono",
        style: "width:100%",
        placeholder: "Path to Vulkan llama-bench.exe",
        disabled: state.runningLlmAutotune,
        value: form.vulkan_bin || "",
        onChange: (e) => actions.updateLlmField("vulkan_bin", e.target.value, false),
      })
    ),
    el(
      "div",
      { className: "form-field", style: "flex:1;min-width:210px" },
      el("label", { className: "field-label" }, "ROCm/HIP llama-bench.exe Path (Optional)"),
      el("input", {
        type: "text",
        className: "text-input mono",
        style: "width:100%",
        placeholder: "Path to ROCm llama-bench.exe",
        disabled: state.runningLlmAutotune,
        value: form.hip_bin || "",
        onChange: (e) => actions.updateLlmField("hip_bin", e.target.value, false),
      })
    ),
    el(
      "div",
      { className: "form-field" },
      el("label", { className: "field-label" }, "Context Depth (-d)"),
      el(
        "select",
        {
          className: "select-input mono",
          disabled: state.runningLlmAutotune,
          onChange: (e) => actions.updateLlmField("context_depth", Number(e.target.value), false),
        },
        [0, 1024, 2048, 4096].map((d) =>
          el("option", { value: String(d), selected: Number(form.context_depth) === d }, `d=${d}`)
        )
      )
    ),
    el(
      "div",
      { className: "form-field", style: "align-self:flex-end" },
      el(
        "button",
        {
          type: "button",
          className: "btn btn-primary",
          disabled: state.runningLlmAutotune || state.busyOp,
          onClick: actions.runLlmAutotune,
        },
        state.runningLlmAutotune ? "Autotuning Model..." : "Run Empirical Autotune"
      )
    )
  );

  root.appendChild(
    makePanel(
      "Empirical llama-bench Autotuner",
      "Benchmarks Baseline vs WDDM Priority=1 (-fa 0) vs Q4_0 KV Cache at selected context depth",
      autotuneControls
    )
  );

  if (state.llmAutotuneError) {
    root.appendChild(el("div", { className: "banner banner-err" }, state.llmAutotuneError));
  }

  const autoRep = state.llmAutotuneReport;
  if (autoRep && Array.isArray(autoRep.candidates)) {
    const arows = autoRep.candidates.map((c) => {
      const isWinner = c.profile_id === autoRep.best_balanced_profile_id;
      return el(
        "tr",
        {},
        el(
          "td",
          { className: "mono" },
          c.profile_id,
          isWinner ? el("span", { className: "badge badge-ok", style: "margin-left:6px" }, "WINNER") : null,
          el("div", { className: "sub-detail" }, c.title)
        ),
        el("td", { className: "mono" }, c.backend.toUpperCase()),
        el("td", { className: "mono" }, c.flash_attn ? "on" : "off"),
        el("td", { className: "mono" }, `${c.cache_type_k}/${c.cache_type_v}`),
        el("td", { className: "mono muted-text" }, c.env_summary),
        el("td", { className: "num mono" }, c.pp_tok_s != null ? `${c.pp_tok_s.toFixed(2)} tok/s` : "ERR"),
        el("td", { className: "num mono" }, c.tg_tok_s != null ? `${c.tg_tok_s.toFixed(2)} tok/s` : "ERR"),
        el(
          "td",
          { className: "num mono" },
          c.tg_speedup_vs_baseline != null ? `${c.tg_speedup_vs_baseline.toFixed(2)}x` : "n/a"
        ),
        el(
          "td",
          {},
          el(
            "button",
            {
              type: "button",
              className: "btn",
              onClick: () => actions.copyLlmCommand(c.llama_server_cmd),
            },
            "Copy Cmd"
          )
        )
      );
    });

    root.appendChild(
      makePanel(
        `Empirical Autotune Results (pp${autoRep.prompt_tokens} + tg${autoRep.gen_tokens} @ d=${autoRep.context_depth})`,
        `Best Balanced: ${autoRep.best_balanced_profile_id || "n/a"}`,
        el(
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
                el("th", {}, "Profile"),
                el("th", {}, "Backend"),
                el("th", {}, "FA"),
                el("th", {}, "KV Cache"),
                el("th", {}, "Env Tweaks"),
                el("th", { className: "num" }, "Prefill (pp)"),
                el("th", { className: "num" }, "Decode (tg)"),
                el("th", { className: "num" }, "TG Speedup"),
                el("th", {}, "Launch")
              )
            ),
            el("tbody", {}, arows)
          )
        ),
        true
      )
    );
  }
}
