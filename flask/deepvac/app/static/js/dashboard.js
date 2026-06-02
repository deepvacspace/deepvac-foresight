const colors = ["#8bd66f", "#4f7cff", "#51d6c7", "#f2bd52", "#ff6f7d", "#b792ff", "#f48fb1", "#7ec8ff"];

const state = {
  runs: [],
  activeRun: null,
  detail: null,
  selectedColumns: new Set(["temp", "temp_ref"]),
  compareMode: false,
  compareRuns: new Set(),
  showSetpoint: true,
  setpointValue: 0,
  series: null,
  simSeries: null,
  charts: new Map(),
  viewports: new Map(),
  drag: null,
  sidebarCollapsed: true,
};

const el = {
  runList: document.getElementById("runList"),
  runSearch: document.getElementById("runSearch"),
  refreshRuns: document.getElementById("refreshRuns"),
  sidebarToggle: document.getElementById("sidebarToggle"),
  themeToggle: document.getElementById("themeToggle"),
  dataRoot: document.getElementById("dataRoot"),
  summaryStrip: document.getElementById("summaryStrip"),
  activeRunTitle: document.getElementById("activeRunTitle"),
  channelList: document.getElementById("channelList"),
  toggleChannels: document.getElementById("toggleChannels"),
  showSetpoint: document.getElementById("showSetpoint"),
  setpointValue: document.getElementById("setpointValue"),
  bandMetrics: document.getElementById("bandMetrics"),
  samplePanel: document.getElementById("samplePanel"),
  sampleTable: document.getElementById("sampleTable"),
  chartMode: document.getElementById("chartMode"),
  resetZoom: document.getElementById("resetZoom"),
  downloadToggle: document.getElementById("downloadToggle"),
  downloadMenu: document.getElementById("downloadMenu"),
  exportChart: document.getElementById("exportChart"),
  exportRunCsv: document.getElementById("exportRunCsv"),
  exportCompareCsv: document.getElementById("exportCompareCsv"),
  openReport: document.getElementById("openReport"),
  legend: document.getElementById("legend"),
  comparisonTable: document.getElementById("comparisonTable"),
  canvas: document.getElementById("mainChart"),
  mainTooltip: document.getElementById("mainTooltip"),
  mainSpinner: document.getElementById("mainSpinner"),
  runsTool: document.getElementById("runsTool"),
  simulatorTool: document.getElementById("simulatorTool"),
  runsSidebar: document.getElementById("runsSidebar"),
  toolTabs: document.querySelectorAll(".tool-tab"),
  simForm: document.getElementById("simForm"),
  runSimulation: document.getElementById("runSimulation"),
  simSummaryStrip: document.getElementById("simSummaryStrip"),
  simTitle: document.getElementById("simTitle"),
  simChannel: document.getElementById("simChannel"),
  simCanvas: document.getElementById("simChart"),
  simTooltip: document.getElementById("simTooltip"),
  simSpinner: document.getElementById("simSpinner"),
  simLegend: document.getElementById("simLegend"),
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtAxis(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return Math.round(number).toLocaleString();
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch (_) {
      // Keep the HTTP message if the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

function apiRunPath(runKey, suffix = "") {
  return `/dashboard-api/runs/${encodeURIComponent(runKey)}${suffix}`;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function downloadText(filename, content, type = "text/csv") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function setDownloadMenu(open) {
  el.downloadMenu.classList.toggle("open", open);
  el.downloadToggle.setAttribute("aria-expanded", open ? "true" : "false");
}

function toggleDownloadMenu() {
  setDownloadMenu(!el.downloadMenu.classList.contains("open"));
}

function setMainLoading(loading) {
  el.mainSpinner.classList.toggle("active", loading);
}

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : "dark";
  document.body.classList.toggle("light-mode", resolved === "light");
  el.themeToggle.querySelector(".theme-icon").textContent = resolved === "light" ? "☀" : "◐";
  el.themeToggle.title = resolved === "light" ? "Switch to dark mode" : "Switch to light mode";
  el.themeToggle.setAttribute("aria-label", el.themeToggle.title);
  localStorage.setItem("deepvac-theme", resolved);
  redrawVisibleCharts();
}

function applySidebarState() {
  document.body.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  el.sidebarToggle.title = state.sidebarCollapsed ? "Show sidebar" : "Collapse sidebar";
  el.sidebarToggle.setAttribute("aria-label", el.sidebarToggle.title);
  setTimeout(redrawVisibleCharts, 0);
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  applySidebarState();
}

function toggleTheme() {
  applyTheme(document.body.classList.contains("light-mode") ? "dark" : "light");
}

function redrawVisibleCharts() {
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
  drawSimulation();
}

function payloadForCanvas(canvas) {
  if (canvas === el.simCanvas) {
    if (!state.simSeries) return null;
    const channel = el.simChannel.value;
    return {
      payload: {
        columns: channel === "temp" ? ["temp", "temp_ref"] : [channel],
        points: state.simSeries.points,
      },
      legend: el.simLegend,
      tooltip: el.simTooltip,
      mode: "line",
    };
  }
  return {
    payload: state.series,
    legend: el.legend,
    tooltip: el.mainTooltip,
    mode: el.chartMode.value,
  };
}

function redrawCanvas(canvas) {
  const config = payloadForCanvas(canvas);
  if (!config) return;
  drawChart(canvas, config.legend, config.tooltip, config.payload, config.mode);
}

function activeSetpoint() {
  const value = Number(state.setpointValue);
  if (!state.compareMode || !state.showSetpoint || !Number.isFinite(value)) return null;
  return value;
}

function syncCompareMode() {
  state.compareMode = state.compareRuns.size > 1;
  updateSetpointControls();
}

function renderRuns() {
  const query = el.runSearch.value.trim().toLowerCase();
  const runs = state.runs.filter((run) => JSON.stringify(run).toLowerCase().includes(query));
  el.runList.innerHTML = "";

  runs.forEach((run) => {
    const row = document.createElement("div");
    row.className = `run-item ${state.activeRun === run.key ? "active" : ""}`;
    row.innerHTML = `
      <input class="run-select" type="checkbox" title="Select run for charting or comparison" ${state.compareRuns.has(run.key) ? "checked" : ""}>
      <button type="button" class="run-open">
        <span class="run-id">${run.id}</span>
        <span class="run-meta">${fmt(run.samples, 0)} pts</span>
        <span class="run-meta">MAE ${fmt(run.mae)}</span>
      </button>
    `;

    row.querySelector(".run-select").addEventListener("change", (event) => {
      if (event.target.checked) state.compareRuns.add(run.key);
      else state.compareRuns.delete(run.key);
      if (state.compareRuns.size === 0) {
        state.compareRuns.add(run.key);
        event.target.checked = true;
      }
      syncCompareMode();
      if (!state.compareRuns.has(state.activeRun)) {
        const nextRun = Array.from(state.compareRuns)[0];
        if (nextRun) {
          loadRun(nextRun, { preserveSelection: true });
          return;
        }
      }
      renderRuns();
      renderSummary();
      renderComparisonTable();
      loadSeries();
      renderTable();
    });
    row.querySelector(".run-open").addEventListener("click", () => loadRun(run.key, { replaceSelection: true }));
    el.runList.appendChild(row);
  });
}

function renderSummary() {
  const run = state.detail?.run;
  const summary = state.detail?.summary || {};
  const stats = state.compareMode
    ? [
        ["Selected Runs", state.compareRuns.size],
        ["Channel", firstSelectedColumn()],
        ["Active Run", run?.id || "-"],
        ["Samples", run?.samples],
        ["MAE", summary.mae],
      ]
    : [
        ["Samples", run?.samples],
        ["Duration", run?.duration_s ? `${fmt(run.duration_s, 1)} s` : "-"],
        ["MAE", summary.mae],
        ["Cost", summary.cost],
        ["Tail MAE", summary.tail_mae],
      ];

  el.summaryStrip.innerHTML = stats.map(([label, value]) => `
    <div class="stat"><span>${label}</span><strong>${fmt(value)}</strong></div>
  `).join("");
}

function comparisonRows() {
  const keys = state.compareMode ? Array.from(state.compareRuns) : [state.activeRun].filter(Boolean);
  if (state.compareMode && state.activeRun && !keys.includes(state.activeRun)) keys.unshift(state.activeRun);
  return keys
    .map((key) => state.runs.find((run) => run.key === key))
    .filter(Boolean);
}

function renderComparisonTable() {
  const rows = comparisonRows();
  if (!rows.length) {
    el.comparisonTable.innerHTML = "<tbody><tr><td>No run selected.</td></tr></tbody>";
    return;
  }

  const columns = [
    ["Run", (run) => run.id],
    ["Cost", (run) => fmt(run.cost)],
    ["MAE", (run) => fmt(run.mae)],
    ["Tail MAE", (run) => fmt(run.tail_mae)],
    ["Overshoot", (run) => fmt(run.overshoot)],
  ];
  el.comparisonTable.innerHTML = `
    <thead><tr>${columns.map(([label]) => `<th>${label}</th>`).join("")}</tr></thead>
    <tbody>
      ${rows.map((run) => `<tr>${columns.map(([, getter]) => `<td>${getter(run)}</td>`).join("")}</tr>`).join("")}
    </tbody>
  `;
}

function renderChannels() {
  const columns = state.detail?.numeric_columns || [];
  el.channelList.innerHTML = columns.map((column) => `
    <label class="channel">
      <span>${column}</span>
      <input type="checkbox" value="${column}" ${state.selectedColumns.has(column) ? "checked" : ""}>
    </label>
  `).join("");

  el.channelList.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.selectedColumns.add(input.value);
      else state.selectedColumns.delete(input.value);
      renderChannelToggleLabel();
      loadSeries();
    });
  });
  renderChannelToggleLabel();
}

function renderChannelToggleLabel() {
  const total = state.detail?.numeric_columns?.length || 0;
  el.toggleChannels.textContent = state.selectedColumns.size >= total && total > 0 ? "Clear All" : "Select All";
}

function toggleAllChannels() {
  const columns = state.detail?.numeric_columns || [];
  if (state.selectedColumns.size >= columns.length) state.selectedColumns.clear();
  else state.selectedColumns = new Set(columns);
  renderChannels();
  loadSeries();
}

function renderBands() {
  const bands = state.detail?.bands || [];
  if (!bands.length) {
    el.bandMetrics.innerHTML = "<p class=\"eyebrow\" style=\"padding:14px\">No band metrics found</p>";
    return;
  }
  const columns = ["band", "kp", "ki", "kd", "cost", "n_samples", "far_mae", "mid_mae", "tail_mae", "overshoot"];
  el.bandMetrics.innerHTML = `
    <table>
      <thead><tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr></thead>
      <tbody>${bands.map((row) => `<tr>${columns.map((column) => `<td>${fmt(row[column])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

async function renderTable() {
  el.samplePanel.hidden = state.compareMode;
  if (state.compareMode) {
    el.sampleTable.innerHTML = "";
    return;
  }

  const table = await getJSON(apiRunPath(state.activeRun, "/table"));
  el.sampleTable.innerHTML = `
    <thead><tr>${table.columns.map((column) => `<th>${column}</th>`).join("")}</tr></thead>
    <tbody>
      ${table.rows.map((row) => `<tr>${table.columns.map((column) => `<td>${fmt(row[column])}</td>`).join("")}</tr>`).join("")}
    </tbody>
  `;
}

function firstSelectedColumn() {
  return Array.from(state.selectedColumns)[0] || null;
}

function defaultSelectedColumns(columns) {
  const preferred = ["temp", "temp_ref"].filter((column) => columns.includes(column));
  if (preferred.length) return preferred;
  return columns.filter((column) => column !== "temp_u").slice(0, 3);
}

function makeDatasets(seriesPayload) {
  if (!seriesPayload) return [];
  if (seriesPayload.series) {
    const channel = firstSelectedColumn();
    if (!channel) return [];
    return seriesPayload.series.map((item, index) => ({
      label: `${item.label} / ${channel}`,
      color: colors[index % colors.length],
      points: normalizePoints(item.points, channel, true),
    }));
  }

  return seriesPayload.columns.map((column, index) => ({
    label: column,
    color: colors[index % colors.length],
    points: normalizePoints(seriesPayload.points, column, false),
  }));
}

function normalizePoints(points, column, elapsedMode) {
  const firstT = points.find((point) => Number.isFinite(point.t))?.t || 0;
  return points
    .map((point) => ({
      x: Number.isFinite(point.t) ? point.t - firstT : point.i,
      y: point.values[column],
      rawX: point.t ?? point.i,
      label: point.label,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
}

function bounds(xs, ys, options = {}) {
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padY = (maxY - minY || 1) * 0.08;
  return {
    minX: options.startXAtZero ? 0 : minX,
    maxX,
    minY: minY - padY,
    maxY: Number.isFinite(options.maxY) ? options.maxY : maxY + padY,
  };
}

function viewportFor(canvas, dataBounds) {
  const current = state.viewports.get(canvas);
  if (!current) return dataBounds;
  return {
    minX: Number.isFinite(current.minX) ? current.minX : dataBounds.minX,
    maxX: Number.isFinite(current.maxX) ? current.maxX : dataBounds.maxX,
    minY: Number.isFinite(current.minY) ? current.minY : dataBounds.minY,
    maxY: Number.isFinite(current.maxY) ? current.maxY : dataBounds.maxY,
  };
}

function clearViewport(canvas) {
  state.viewports.delete(canvas);
  const chart = state.charts.get(canvas);
  if (chart) chart.hovered = null;
  redrawCanvas(canvas);
}

function drawChart(canvas, legend, tooltip, payload, mode) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--chart-bg").trim() || "#101116";
  ctx.fillRect(0, 0, width, height);

  const datasets = makeDatasets(payload);
  if (!datasets.length || !datasets.some((dataset) => dataset.points.length)) {
    state.charts.delete(canvas);
    if (tooltip) tooltip.style.display = "none";
    legend.innerHTML = "";
    return;
  }

  const allX = datasets.flatMap((dataset) => dataset.points.map((point) => point.x));
  const allY = datasets.flatMap((dataset) => dataset.points.map((point) => point.y));
  const setpoint = canvas === el.canvas ? activeSetpoint() : null;
  if (setpoint !== null) allY.push(setpoint);
  const hasTemperature = datasets.some((dataset) => dataset.label.toLowerCase().includes("temp"));
  const dataBounds = bounds(allX, allY, {
    startXAtZero: canvas === el.canvas,
    maxY: canvas === el.canvas && hasTemperature ? 30 : undefined,
  });
  const b = viewportFor(canvas, dataBounds);
  const margin = { left: 92, right: 24, top: 22, bottom: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xScale = (value) => margin.left + ((value - b.minX) / (b.maxX - b.minX || 1)) * plotW;
  const yScale = (value) => margin.top + plotH - ((value - b.minY) / (b.maxY - b.minY || 1)) * plotH;

  drawGrid(ctx, width, height, margin, plotW, plotH, b, datasets);
  let hitPoints = [];
  withPlotClip(ctx, margin, width, height, () => {
    drawAnnotations(ctx, canvas, xScale, yScale, margin, width, height, "region");
    if (setpoint !== null) drawSetpoint(ctx, setpoint, yScale, margin, width);
    hitPoints = drawSeries(ctx, datasets, mode, xScale, yScale);
    drawAnnotations(ctx, canvas, xScale, yScale, margin, width, height, "marker");
  });
  const previous = state.charts.get(canvas);
  if (previous?.hovered && isInPlot(previous.hovered, margin, width, height)) drawHoverMarker(ctx, previous.hovered);
  renderLegend(legend, datasets);
  const visibleHitPoints = hitPoints.filter((point) => isInPlot(point, margin, width, height));
  state.charts.set(canvas, { hitPoints: visibleHitPoints, tooltip, mode, bounds: b, dataBounds, margin, plotW, plotH, hovered: previous?.hovered || null });
}

function withPlotClip(ctx, margin, width, height, draw) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, width - margin.left - margin.right, height - margin.top - margin.bottom);
  ctx.clip();
  draw();
  ctx.restore();
}

function isInPlot(point, margin, width, height) {
  return (
    point.x >= margin.left &&
    point.x <= width - margin.right &&
    point.y >= margin.top &&
    point.y <= height - margin.bottom
  );
}

function drawGrid(ctx, width, height, margin, plotW, plotH, b, datasets) {
  const styles = getComputedStyle(document.body);
  ctx.strokeStyle = styles.getPropertyValue("--line").trim() || "#2d313b";
  ctx.lineWidth = 1;
  ctx.fillStyle = styles.getPropertyValue("--muted").trim() || "#8f98a8";
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";

  for (let i = 0; i <= 5; i += 1) {
    const y = margin.top + (plotH / 5) * i;
    const value = b.maxY - ((b.maxY - b.minY) / 5) * i;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillText(fmtAxis(value), margin.left - 12, y + 4);
  }

  ctx.textAlign = "start";
  for (let i = 0; i <= 4; i += 1) {
    const x = margin.left + (plotW / 4) * i;
    const value = b.minX + ((b.maxX - b.minX) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, height - margin.bottom);
    ctx.stroke();
    ctx.fillText(fmtAxis(value), x - 12, height - margin.bottom + 20);
  }

  const channelText = datasets.map((dataset) => dataset.label.toLowerCase()).join(" ");
  const yLabel = channelText.includes("temp") ? "Temperature [deg C]" : "Value";
  ctx.fillStyle = styles.getPropertyValue("--muted").trim() || "#8f98a8";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("Time elapsed [s]", margin.left + plotW / 2, height - 14);
  ctx.save();
  ctx.translate(16, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.textAlign = "start";
}

function drawSeries(ctx, datasets, mode, xScale, yScale) {
  const hitPoints = [];
  datasets.forEach((dataset) => {
    ctx.strokeStyle = dataset.color;
    ctx.fillStyle = dataset.color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();

    dataset.points.forEach((point, index) => {
      const x = xScale(point.x);
      const y = yScale(point.y);
      hitPoints.push({ x, y, dataset: dataset.label, color: dataset.color, value: point.y, xValue: point.x, label: point.label });
      if (mode === "scatter") {
        ctx.fillRect(x - 2, y - 2, 4, 4);
      } else if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    if (mode !== "scatter") ctx.stroke();
  });
  return hitPoints;
}

function drawHoverMarker(ctx, point) {
  if (!point) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--chart-bg").trim() || "#101116";
  ctx.fill();
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = point.color;
  ctx.stroke();
  ctx.restore();
}

function drawSetpoint(ctx, value, yScale, margin, width) {
  const styles = getComputedStyle(document.body);
  const y = yScale(value);
  ctx.save();
  ctx.strokeStyle = styles.getPropertyValue("--muted").trim() || "#8f98a8";
  ctx.fillStyle = styles.getPropertyValue("--muted").trim() || "#8f98a8";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([7, 5]);
  ctx.beginPath();
  ctx.moveTo(margin.left, y);
  ctx.lineTo(width - margin.right, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = "11px system-ui";
  ctx.fillText(`Setpoint ${fmt(value, 2)}`, margin.left + 8, y - 7);
  ctx.restore();
}

function annotationColor(kind) {
  if (kind === "overshoot") return "#ff6f7d";
  if (kind === "pid") return "#f2bd52";
  if (kind === "invalid") return "#ff6f7d";
  if (kind === "settling") return "#8bd66f";
  return "#8f98a8";
}

function chartAnnotations(canvas) {
  if (canvas !== el.canvas || state.compareMode) return [];
  return state.detail?.annotations || [];
}

function drawAnnotations(ctx, canvas, xScale, yScale, margin, width, height, phase) {
  const annotations = chartAnnotations(canvas);
  if (!annotations.length) return;
  const plotTop = margin.top;
  const plotBottom = height - margin.bottom;
  const plotLeft = margin.left;
  const plotRight = width - margin.right;

  ctx.save();
  annotations.forEach((annotation) => {
    const color = annotationColor(annotation.kind);
    if (phase === "region" && annotation.type === "region-x") {
      const x0 = Math.max(plotLeft, Math.min(plotRight, xScale(annotation.x0)));
      const x1 = Math.max(plotLeft, Math.min(plotRight, xScale(annotation.x1)));
      ctx.fillStyle = annotation.kind === "invalid" ? "rgba(255, 111, 125, .13)" : "rgba(139, 214, 111, .10)";
      ctx.fillRect(Math.min(x0, x1), plotTop, Math.abs(x1 - x0), plotBottom - plotTop);
      ctx.fillStyle = color;
      ctx.font = "11px system-ui";
      ctx.fillText(annotation.label, Math.min(x0, x1) + 6, plotTop + 16);
    }

    if (phase !== "marker") return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.2;
    ctx.font = "11px system-ui";
    if (annotation.type === "line-y" && Number.isFinite(annotation.y)) {
      const y = yScale(annotation.y);
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(annotation.label, plotLeft + 8, y - 7);
    } else if (annotation.type === "line-x" && Number.isFinite(annotation.x)) {
      const x = xScale(annotation.x);
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(x, plotTop);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(annotation.label, x + 5, plotTop + 30);
    } else if (annotation.type === "point" && Number.isFinite(annotation.x) && Number.isFinite(annotation.y)) {
      const x = xScale(annotation.x);
      const y = yScale(annotation.y);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(annotation.label, x + 7, y - 7);
    }
  });
  ctx.restore();
}

function renderLegend(legend, datasets) {
  legend.innerHTML = datasets.map((dataset) => `
    <span class="legend-item"><span class="swatch" style="background:${dataset.color}"></span>${dataset.label}</span>
  `).join("");
}

function showNearestPoint(canvas, event) {
  const chart = state.charts.get(canvas);
  if (state.drag?.active) return;
  if (!chart || !chart.tooltip || !chart.hitPoints.length) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = null;
  let bestDistance = Infinity;

  chart.hitPoints.forEach((point) => {
    const dx = point.x - x;
    const dy = point.y - y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < bestDistance) {
      best = point;
      bestDistance = distance;
    }
  });

  if (!best || bestDistance > 28) {
    chart.tooltip.style.display = "none";
    if (chart.hovered) {
      chart.hovered = null;
      redrawCanvas(canvas);
    }
    return;
  }

  chart.hovered = best;
  redrawCanvas(canvas);
  const refreshed = state.charts.get(canvas);
  if (refreshed) {
    refreshed.hovered = best;
    drawHoverMarker(canvas.getContext("2d"), best);
  }

  chart.tooltip.innerHTML = `
    <strong style="color:${best.color}">${best.dataset}</strong>
    <span>Value: ${fmt(best.value, 2)}</span>
    <span>Time: ${fmtAxis(best.xValue)} s</span>
    ${best.label ? `<span>${best.label}</span>` : ""}
  `;
  chart.tooltip.style.display = "block";

  const parentRect = canvas.parentElement.getBoundingClientRect();
  const tooltipRect = chart.tooltip.getBoundingClientRect();
  const left = Math.min(Math.max(x + 14, 8), parentRect.width - tooltipRect.width - 8);
  const top = Math.min(Math.max(y + 14, 8), parentRect.height - tooltipRect.height - 8);
  chart.tooltip.style.left = `${left}px`;
  chart.tooltip.style.top = `${top}px`;
}

function hideTooltip(canvas) {
  const chart = state.charts.get(canvas);
  if (chart?.tooltip) chart.tooltip.style.display = "none";
  if (chart) {
    chart.hovered = null;
    redrawCanvas(canvas);
  }
}

function pointToData(canvas, clientX, clientY) {
  const chart = state.charts.get(canvas);
  if (!chart) return null;
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const { bounds: b, margin, plotW, plotH } = chart;
  const dataX = b.minX + ((x - margin.left) / (plotW || 1)) * (b.maxX - b.minX);
  const dataY = b.maxY - ((y - margin.top) / (plotH || 1)) * (b.maxY - b.minY);
  return { x, y, dataX, dataY };
}

function startPan(canvas, event) {
  if (event.button !== 0) return;
  const chart = state.charts.get(canvas);
  if (!chart) return;
  event.preventDefault();
  state.drag = {
    active: true,
    canvas,
    startX: event.clientX,
    startY: event.clientY,
    bounds: { ...chart.bounds },
    plotW: chart.plotW,
    plotH: chart.plotH,
  };
  canvas.classList.add("panning");
  hideTooltip(canvas);
}

function panChart(event) {
  if (!state.drag?.active) return;
  const { canvas, startX, startY, bounds: b, plotW, plotH } = state.drag;
  const dx = event.clientX - startX;
  const dy = event.clientY - startY;
  const spanX = b.maxX - b.minX;
  const spanY = b.maxY - b.minY;
  const shiftX = -(dx / (plotW || 1)) * spanX;
  const shiftY = (dy / (plotH || 1)) * spanY;
  state.viewports.set(canvas, {
    minX: b.minX + shiftX,
    maxX: b.maxX + shiftX,
    minY: b.minY + shiftY,
    maxY: b.maxY + shiftY,
  });
  redrawCanvas(canvas);
}

function endPan() {
  if (!state.drag?.active) return;
  state.drag.canvas.classList.remove("panning");
  state.drag = null;
}

function zoomChart(canvas, event) {
  const chart = state.charts.get(canvas);
  if (!chart) return;
  event.preventDefault();
  if (chart.tooltip) chart.tooltip.style.display = "none";
  chart.hovered = null;
  const point = pointToData(canvas, event.clientX, event.clientY);
  if (!point) return;
  const b = chart.bounds;
  const factor = Math.exp(event.deltaY * 0.001);
  const minSpanX = Math.max((chart.dataBounds.maxX - chart.dataBounds.minX) * 0.001, 1e-6);
  const minSpanY = Math.max((chart.dataBounds.maxY - chart.dataBounds.minY) * 0.001, 1e-6);
  const spanX = Math.max((b.maxX - b.minX) * factor, minSpanX);
  const spanY = Math.max((b.maxY - b.minY) * factor, minSpanY);
  const rx = (point.dataX - b.minX) / (b.maxX - b.minX || 1);
  const ry = (point.dataY - b.minY) / (b.maxY - b.minY || 1);
  state.viewports.set(canvas, {
    minX: point.dataX - spanX * rx,
    maxX: point.dataX + spanX * (1 - rx),
    minY: point.dataY - spanY * ry,
    maxY: point.dataY + spanY * (1 - ry),
  });
  redrawCanvas(canvas);
}

async function loadSeries() {
  if (!state.activeRun) return;
  const columns = Array.from(state.selectedColumns);
  if (!columns.length) {
    state.series = { columns: [], points: [] };
    renderSummary();
    drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
    return;
  }

  if (state.compareMode) {
    const keys = Array.from(state.compareRuns);
    if (!keys.includes(state.activeRun)) keys.unshift(state.activeRun);
    const channel = firstSelectedColumn();
    if (!channel) {
      state.series = { series: [] };
      renderSummary();
      drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
      return;
    }
    const series = await Promise.all(keys.map(async (key) => {
      const payload = await getJSON(apiRunPath(key, `/series?columns=${encodeURIComponent(channel)}`));
      const run = state.runs.find((item) => item.key === key);
      return { label: run ? run.id : key, points: payload.points };
    }));
    state.series = { series };
  } else {
    state.series = await getJSON(apiRunPath(state.activeRun, `/series?columns=${encodeURIComponent(columns.join(","))}`));
  }

  renderSummary();
  renderComparisonTable();
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
}

async function loadRun(runId, options = {}) {
  setMainLoading(true);
  state.activeRun = runId;
  localStorage.setItem("deepvac-active-run", runId);
  try {
    if (options.replaceSelection) {
      state.compareRuns = new Set([runId]);
    } else if (!options.preserveSelection) {
      state.compareRuns.add(runId);
    }
    syncCompareMode();
    state.viewports.delete(el.canvas);
    state.detail = await getJSON(apiRunPath(runId));
    if (!Array.from(state.selectedColumns).some((column) => state.detail.numeric_columns.includes(column))) {
      state.selectedColumns = new Set(defaultSelectedColumns(state.detail.numeric_columns));
    }
    el.activeRunTitle.textContent = state.detail.run.id;
    renderRuns();
    renderSummary();
    renderChannels();
    renderBands();
    renderComparisonTable();
    await Promise.all([loadSeries(), renderTable()]);
  } finally {
    setMainLoading(false);
  }
}

async function loadRuns() {
  const payload = await getJSON("/dashboard-api/runs");
  state.runs = payload.runs;
  el.dataRoot.textContent = payload.data_root;
  renderRuns();
  const savedRun = localStorage.getItem("deepvac-active-run");
  const initialRun = state.runs.find((run) => run.key === savedRun)?.key || state.runs[0]?.key;
  if (state.runs.length && !state.activeRun) await loadRun(initialRun);
  else renderComparisonTable();
}

function updateSetpointControls() {
  el.showSetpoint.disabled = !state.compareMode;
  el.setpointValue.disabled = !state.compareMode || !state.showSetpoint;
}

function exportChartPng() {
  setDownloadMenu(false);
  if (!state.series) return;
  el.canvas.toBlob((blob) => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${state.detail?.run?.id || "deepvac-chart"}.png`;
    document.body.appendChild(link);
    link.click();
    URL.revokeObjectURL(link.href);
    link.remove();
  }, "image/png");
}

async function exportRunCsv() {
  setDownloadMenu(false);
  if (!state.activeRun) return;
  const table = await getJSON(apiRunPath(state.activeRun, "/table"));
  const lines = [
    table.columns.map(csvEscape).join(","),
    ...table.rows.map((row) => table.columns.map((column) => csvEscape(row[column])).join(",")),
  ];
  downloadText(`${state.detail?.run?.id || "run"}-samples.csv`, lines.join("\n"));
}

function exportComparisonCsv() {
  setDownloadMenu(false);
  const rows = comparisonRows();
  const columns = ["Run", "Cost", "MAE", "Tail MAE", "Overshoot"];
  const lines = [
    columns.map(csvEscape).join(","),
    ...rows.map((run) => [
      run.id,
      run.cost,
      run.mae,
      run.tail_mae,
      run.overshoot,
    ].map(csvEscape).join(",")),
  ];
  downloadText("deepvac-comparison.csv", lines.join("\n"));
}

function openRunReport() {
  setDownloadMenu(false);
  if (!state.activeRun) return;
  window.open(apiRunPath(state.activeRun, "/report"), "_blank", "noopener");
}

function switchTool(tool) {
  el.toolTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tool === tool));
  el.runsTool.classList.toggle("active", tool === "runs");
  el.simulatorTool.classList.toggle("active", tool === "simulator");
  el.runsSidebar.style.display = tool === "runs" ? "" : "none";
  setTimeout(() => {
    if (tool === "runs") drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
    if (tool === "simulator") drawSimulation();
  }, 0);
}

async function runSimulation() {
  const payload = Object.fromEntries(new FormData(el.simForm).entries());
  Object.keys(payload).forEach((key) => {
    payload[key] = Number(payload[key]);
  });

  if (!el.simForm.reportValidity()) return;

  el.runSimulation.disabled = true;
  el.simSpinner.classList.add("active");
  el.simTitle.textContent = "Running simulation...";
  try {
    state.simSeries = await getJSON("/dashboard-api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    el.simTitle.textContent = `Kp ${payload.kp} / Ki ${payload.ki} / Kd ${payload.kd}`;
    renderSimSummary();
    drawSimulation();
  } catch (error) {
    el.simTitle.textContent = "Simulation failed";
    el.simSummaryStrip.innerHTML = `<div class="stat"><span>Error</span><strong>${error.message}</strong></div>`;
  } finally {
    el.runSimulation.disabled = false;
    el.simSpinner.classList.remove("active");
  }
}

function renderSimSummary() {
  const metrics = state.simSeries?.metrics || {};
  const stats = [
    ["Cost", metrics.cost],
    ["Tail MAE", metrics.tail_mae],
    ["End Temp", metrics.end_temp],
    ["Overshoot", metrics.overshoot_max],
    ["Settle s", metrics.time_to_settle_s],
  ];
  el.simSummaryStrip.innerHTML = stats.map(([label, value]) => `
    <div class="stat"><span>${label}</span><strong>${fmt(value)}</strong></div>
  `).join("");
}

function drawSimulation() {
  if (!state.simSeries) return;
  const channel = el.simChannel.value;
  const payload = {
    columns: channel === "temp" ? ["temp", "temp_ref"] : [channel],
    points: state.simSeries.points,
  };
  drawChart(el.simCanvas, el.simLegend, el.simTooltip, payload, "line");
}

el.runSearch.addEventListener("input", renderRuns);
el.refreshRuns.addEventListener("click", loadRuns);
el.sidebarToggle.addEventListener("click", toggleSidebar);
el.themeToggle.addEventListener("click", toggleTheme);
el.chartMode.addEventListener("change", () => drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value));
el.resetZoom.addEventListener("click", () => clearViewport(el.canvas));
el.downloadToggle.addEventListener("click", toggleDownloadMenu);
el.exportChart.addEventListener("click", exportChartPng);
el.exportRunCsv.addEventListener("click", exportRunCsv);
el.exportCompareCsv.addEventListener("click", exportComparisonCsv);
el.openReport.addEventListener("click", openRunReport);
el.toggleChannels.addEventListener("click", toggleAllChannels);
el.showSetpoint.addEventListener("change", () => {
  state.showSetpoint = el.showSetpoint.checked;
  updateSetpointControls();
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
});
el.setpointValue.addEventListener("input", () => {
  state.setpointValue = Number(el.setpointValue.value);
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
});
el.toolTabs.forEach((tab) => tab.addEventListener("click", () => switchTool(tab.dataset.tool)));
el.runSimulation.addEventListener("click", runSimulation);
el.simChannel.addEventListener("change", drawSimulation);
el.canvas.addEventListener("mousemove", (event) => showNearestPoint(el.canvas, event));
el.canvas.addEventListener("mouseleave", () => hideTooltip(el.canvas));
el.simCanvas.addEventListener("mousemove", (event) => showNearestPoint(el.simCanvas, event));
el.simCanvas.addEventListener("mouseleave", () => hideTooltip(el.simCanvas));
[el.canvas, el.simCanvas].forEach((canvas) => {
  canvas.addEventListener("mousedown", (event) => startPan(canvas, event));
  canvas.addEventListener("wheel", (event) => zoomChart(canvas, event), { passive: false });
});
window.addEventListener("mousemove", panChart);
window.addEventListener("mouseup", endPan);
document.addEventListener("click", (event) => {
  if (!event.target.closest(".download-menu")) setDownloadMenu(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setDownloadMenu(false);
});
window.addEventListener("resize", () => {
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
  drawSimulation();
});

applyTheme(localStorage.getItem("deepvac-theme") || "dark");
applySidebarState();
syncCompareMode();

loadRuns().catch((error) => {
  el.activeRunTitle.textContent = "Unable to load runs";
  el.summaryStrip.innerHTML = `<div class="stat"><span>Error</span><strong>${error.message}</strong></div>`;
});
