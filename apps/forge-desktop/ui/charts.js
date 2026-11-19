// Crisp high-DPI Canvas 2D chart renderers for telemetry strip plots and fan curves.

function setupCanvas(canvas, cssHeight) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(240, Math.floor(rect.width || canvas.parentElement?.clientWidth || 420));
  const height = cssHeight || Math.max(140, Math.floor(rect.height || 170));
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

/**
 * Draws a multi-series rolling time-series line chart onto a `<canvas>`.
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{label: string, color: string, data: Array<number|null>, unit?: string}>} seriesList
 * @param {{minY?: number|null, maxY?: number|null, unit?: string, thresholdY?: number|null, thresholdLabel?: string}} [opts]
 */
export function drawStripChart(canvas, seriesList, opts = {}) {
  const { ctx, width, height } = setupCanvas(canvas, 170);
  const pad = { top: 22, right: 10, bottom: 18, left: 46 };
  const plotW = Math.max(40, width - pad.left - pad.right);
  const plotH = Math.max(40, height - pad.top - pad.bottom);

  // Determine numeric Y bounds
  const values = [];
  let maxSamples = 2;
  for (const s of seriesList) {
    if (s.data.length > maxSamples) maxSamples = s.data.length;
    for (const v of s.data) {
      if (typeof v === "number" && Number.isFinite(v)) {
        values.push(v);
      }
    }
  }

  let yMin = opts.minY != null ? opts.minY : values.length ? Math.min(...values) : 0;
  let yMax = opts.maxY != null ? opts.maxY : values.length ? Math.max(...values) : 100;
  if (yMax <= yMin) {
    yMax = yMin + 10;
  } else if (opts.minY == null || opts.maxY == null) {
    const span = yMax - yMin;
    if (opts.minY == null) yMin = Math.max(0, Math.floor(yMin - span * 0.1));
    if (opts.maxY == null) yMax = Math.ceil(yMax + span * 0.12);
  }

  // Horizontal gridlines & Y ticks
  const ticks = 4;
  ctx.font = '10px "Cascadia Mono", Consolas, monospace';
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= ticks; i++) {
    const frac = i / ticks;
    const y = pad.top + plotH - frac * plotH;
    const val = yMin + frac * (yMax - yMin);

    ctx.strokeStyle = "#24272d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, Math.round(y) + 0.5);
    ctx.lineTo(pad.left + plotW, Math.round(y) + 0.5);
    ctx.stroke();

    ctx.fillStyle = "#6c737f";
    const label = Math.abs(val) >= 100 ? Math.round(val).toString() : val.toFixed(val % 1 === 0 ? 0 : 1);
    ctx.fillText(label, pad.left - 6, y);
  }

  // Plot frame baseline
  ctx.strokeStyle = "#2d3138";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left + 0.5, pad.top + 0.5, plotW - 1, plotH - 1);

  // Optional threshold line (e.g. 50% WDDM VRAM pressure)
  if (typeof opts.thresholdY === "number" && opts.thresholdY >= yMin && opts.thresholdY <= yMax) {
    const ty = pad.top + plotH - ((opts.thresholdY - yMin) / (yMax - yMin)) * plotH;
    ctx.save();
    ctx.strokeStyle = "rgba(201, 150, 59, 0.55)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.left, ty);
    ctx.lineTo(pad.left + plotW, ty);
    ctx.stroke();
    ctx.restore();

    if (opts.thresholdLabel) {
      ctx.fillStyle = "#c9963b";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(opts.thresholdLabel, pad.left + 5, ty - 2);
    }
  }

  // Series lines + top legend with latest readout
  let legendX = pad.left;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  for (const s of seriesList) {
    let latest = null;
    for (let i = s.data.length - 1; i >= 0; i--) {
      if (typeof s.data[i] === "number" && Number.isFinite(s.data[i])) {
        latest = s.data[i];
        break;
      }
    }

    const latestTxt =
      latest == null
        ? "--"
        : `${Number.isInteger(latest) ? latest : latest.toFixed(1)}${s.unit || opts.unit || ""}`;
    const itemText = `${s.label}: ${latestTxt}`;

    ctx.fillStyle = s.color;
    ctx.fillRect(legendX, 7, 7, 7);
    ctx.fillStyle = "#9aa1ac";
    ctx.fillText(itemText, legendX + 11, 5);
    legendX += ctx.measureText(itemText).width + 24;

    if (s.data.length < 2) continue;

    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "miter";
    ctx.beginPath();
    let penDown = false;
    const windowSize = Math.max(maxSamples - 1, 1);
    const startOffset = Math.max(0, maxSamples - s.data.length);

    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      if (typeof v !== "number" || !Number.isFinite(v)) {
        penDown = false;
        continue;
      }
      const x = pad.left + ((startOffset + i) / windowSize) * plotW;
      const clamped = Math.min(yMax, Math.max(yMin, v));
      const y = pad.top + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
      if (!penDown) {
        ctx.moveTo(x, y);
        penDown = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  if (values.length === 0) {
    ctx.fillStyle = "#6c737f";
    ctx.font = '11.5px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Waiting for ADLX samples...", pad.left + plotW / 2, pad.top + plotH / 2);
  }
}

/**
 * Draws a 5-point temperature-vs-PWM fan curve chart onto a `<canvas>`.
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{temp_c: number, speed_pct: number}>} points
 * @param {{min: number, max: number}|null} tempRange
 * @param {{min: number, max: number}|null} speedRange
 */
export function drawFanCurveChart(canvas, points, tempRange, speedRange) {
  const { ctx, width, height } = setupCanvas(canvas, 185);
  const pad = { top: 14, right: 14, bottom: 26, left: 42 };
  const plotW = Math.max(40, width - pad.left - pad.right);
  const plotH = Math.max(40, height - pad.top - pad.bottom);

  const tMin = tempRange ? tempRange.min : 20;
  const tMax = tempRange ? tempRange.max : 100;
  const sMin = speedRange ? Math.min(0, speedRange.min) : 0;
  const sMax = speedRange ? Math.max(100, speedRange.max) : 100;

  ctx.font = '10px "Cascadia Mono", Consolas, monospace';

  // Grid
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + plotH - (i / 4) * plotH;
    const pct = sMin + (i / 4) * (sMax - sMin);
    ctx.strokeStyle = "#24272d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, Math.round(y) + 0.5);
    ctx.lineTo(pad.left + plotW, Math.round(y) + 0.5);
    ctx.stroke();

    ctx.fillStyle = "#6c737f";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`${Math.round(pct)}%`, pad.left - 6, y);
  }

  for (let i = 0; i <= 4; i++) {
    const x = pad.left + (i / 4) * plotW;
    const temp = tMin + (i / 4) * (tMax - tMin);
    ctx.fillStyle = "#6c737f";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`${Math.round(temp)}C`, x, pad.top + plotH + 6);
  }

  ctx.strokeStyle = "#2d3138";
  ctx.strokeRect(pad.left + 0.5, pad.top + 0.5, plotW - 1, plotH - 1);

  if (!points || points.length === 0) {
    ctx.fillStyle = "#6c737f";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No fan curve points reported", pad.left + plotW / 2, pad.top + plotH / 2);
    return;
  }

  const coords = points.map((p) => {
    const tx = Math.max(tMin, Math.min(tMax, p.temp_c));
    const sy = Math.max(sMin, Math.min(sMax, p.speed_pct));
    return {
      x: pad.left + ((tx - tMin) / Math.max(1, tMax - tMin)) * plotW,
      y: pad.top + plotH - ((sy - sMin) / Math.max(1, sMax - sMin)) * plotH,
      label: `${p.temp_c}C/${p.speed_pct}%`,
    };
  });

  ctx.strokeStyle = "#c93b3b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  coords.forEach((pt, idx) => {
    if (idx === 0) ctx.moveTo(pt.x, pt.y);
    else ctx.lineTo(pt.x, pt.y);
  });
  ctx.stroke();

  for (const pt of coords) {
    ctx.fillStyle = "#e6e8ec";
    ctx.strokeStyle = "#c93b3b";
    ctx.lineWidth = 1.5;
    ctx.fillRect(pt.x - 3, pt.y - 3, 6, 6);
    ctx.strokeRect(pt.x - 3, pt.y - 3, 6, 6);
  }
}
