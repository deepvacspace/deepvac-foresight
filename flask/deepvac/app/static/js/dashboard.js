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
};

const el = {
  runList: document.getElementById("runList"),
  runSearch: document.getElementById("runSearch"),
  refreshRuns: document.getElementById("refreshRuns"),
  themeToggle: document.getElementById("themeToggle"),
  dataRoot: document.getElementById("dataRoot"),
  summaryStrip: document.getElementById("summaryStrip"),
  activeRunTitle: document.getElementById("activeRunTitle"),
  channelList: document.getElementById("channelList"),
  toggleChannels: document.getElementById("toggleChannels"),
  showSetpoint: document.getElementById("showSetpoint"),
  setpointValue: document.getElementById("setpointValue"),
  compareToggle: document.getElementById("compareToggle"),
  bandMetrics: document.getElementById("bandMetrics"),
  sampleTable: document.getElementById("sampleTable"),
  chartMode: document.getElementById("chartMode"),
  resetZoom: document.getElementById("resetZoom"),
  legend: document.getElementById("legend"),
  canvas: document.getElementById("mainChart"),
  mainTooltip: document.getElementById("mainTooltip"),
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

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : "dark";
  document.body.classList.toggle("light-mode", resolved === "light");
  el.themeToggle.querySelector(".theme-icon").textContent = resolved === "light" ? "☀" : "◐";
  el.themeToggle.title = resolved === "light" ? "Switch to dark mode" : "Switch to light mode";
  el.themeToggle.setAttribute("aria-label", el.themeToggle.title);
  localStorage.setItem("deepvac-theme", resolved);
  redrawVisibleCharts();
}

function toggleTheme() {
  applyTheme(document.body.classList.contains("light-mode") ? "dark" : "light");
}

function redrawVisibleCharts() {
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
  drawSimulation();
}

function activeSetpoint() {
  const value = Number(state.setpointValue);
  if (!state.compareMode || !state.showSetpoint || !Number.isFinite(value)) return null;
  return value;
}

function renderRuns() {
  const query = el.runSearch.value.trim().toLowerCase();
  const runs = state.runs.filter((run) => JSON.stringify(run).toLowerCase().includes(query));
  el.runList.innerHTML = "";

  runs.forEach((run) => {
    const row = document.createElement("div");
    row.className = `run-item ${state.activeRun === run.key ? "active" : ""}`;
    row.innerHTML = `
      <input class="run-select" type="checkbox" title="${state.compareMode ? "Compare run" : "Turn compare on to select multiple runs"}" ${state.compareMode ? "" : "disabled"} ${state.compareRuns.has(run.key) ? "checked" : ""}>
      <button type="button" class="run-open">
        <span class="run-id">${run.id}</span>
        <span class="run-meta">${fmt(run.samples, 0)} pts</span>
        <span class="run-meta">MAE ${fmt(run.mae)}</span>
      </button>
    `;

    row.querySelector(".run-select").addEventListener("change", (event) => {
      if (event.target.checked) state.compareRuns.add(run.key);
      else state.compareRuns.delete(run.key);
      if (state.compareMode) loadSeries();
    });
    row.querySelector(".run-open").addEventListener("click", () => loadRun(run.key));
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
  if (state.compareMode) {
    el.sampleTable.innerHTML = "<tbody><tr><td>Raw sample table is hidden while comparing runs.</td></tr></tbody>";
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
      x: elapsedMode && Number.isFinite(point.t) ? point.t - firstT : point.t ?? point.i,
      y: point.values[column],
      rawX: point.t ?? point.i,
      label: point.label,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
}

function bounds(xs, ys) {
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padY = (maxY - minY || 1) * 0.08;
  const padX = (maxX - minX || 1) * 0.02;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
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
  const b = bounds(allX, allY);
  const margin = { left: 60, right: 22, top: 20, bottom: 48 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xScale = (value) => margin.left + ((value - b.minX) / (b.maxX - b.minX || 1)) * plotW;
  const yScale = (value) => margin.top + plotH - ((value - b.minY) / (b.maxY - b.minY || 1)) * plotH;

  drawGrid(ctx, width, height, margin, plotW, plotH, b);
  if (setpoint !== null) drawSetpoint(ctx, setpoint, yScale, margin, width);
  const hitPoints = drawSeries(ctx, datasets, mode, xScale, yScale);
  renderLegend(legend, datasets);
  state.charts.set(canvas, { hitPoints, tooltip, mode });
}

function drawGrid(ctx, width, height, margin, plotW, plotH, b) {
  const styles = getComputedStyle(document.body);
  ctx.strokeStyle = styles.getPropertyValue("--line").trim() || "#2d313b";
  ctx.lineWidth = 1;
  ctx.fillStyle = styles.getPropertyValue("--muted").trim() || "#8f98a8";
  ctx.font = "11px system-ui";

  for (let i = 0; i <= 5; i += 1) {
    const y = margin.top + (plotH / 5) * i;
    const value = b.maxY - ((b.maxY - b.minY) / 5) * i;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillText(fmt(value, 2), 10, y + 4);
  }

  for (let i = 0; i <= 4; i += 1) {
    const x = margin.left + (plotW / 4) * i;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, height - margin.bottom);
    ctx.stroke();
  }
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

function renderLegend(legend, datasets) {
  legend.innerHTML = datasets.map((dataset) => `
    <span class="legend-item"><span class="swatch" style="background:${dataset.color}"></span>${dataset.label}</span>
  `).join("");
}

function showNearestPoint(canvas, event) {
  const chart = state.charts.get(canvas);
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
    return;
  }

  chart.tooltip.innerHTML = `
    <strong style="color:${best.color}">${best.dataset}</strong>
    <span>Value: ${fmt(best.value, 4)}</span>
    <span>X: ${fmt(best.xValue, 2)}</span>
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
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
}

async function loadRun(runId) {
  state.activeRun = runId;
  if (state.compareMode) state.compareRuns.add(runId);
  else state.compareRuns = new Set([runId]);
  state.detail = await getJSON(apiRunPath(runId));
  if (!Array.from(state.selectedColumns).some((column) => state.detail.numeric_columns.includes(column))) {
    state.selectedColumns = new Set(defaultSelectedColumns(state.detail.numeric_columns));
  }
  el.activeRunTitle.textContent = state.detail.run.id;
  renderRuns();
  renderSummary();
  renderChannels();
  renderBands();
  await Promise.all([loadSeries(), renderTable()]);
}

async function loadRuns() {
  const payload = await getJSON("/dashboard-api/runs");
  state.runs = payload.runs;
  el.dataRoot.textContent = payload.data_root;
  renderRuns();
  if (state.runs.length && !state.activeRun) await loadRun(state.runs[0].key);
}

function updateSetpointControls() {
  el.showSetpoint.disabled = !state.compareMode;
  el.setpointValue.disabled = !state.compareMode || !state.showSetpoint;
}

function setCompareMode(enabled) {
  state.compareMode = enabled;
  el.compareToggle.textContent = enabled ? "Compare On" : "Compare Off";
  el.compareToggle.classList.toggle("active", enabled);
  updateSetpointControls();
  if (!enabled && state.activeRun) {
    state.compareRuns = new Set([state.activeRun]);
  } else if (enabled && state.activeRun && state.compareRuns.size === 0) {
    state.compareRuns.add(state.activeRun);
  }
  renderRuns();
  renderSummary();
  if (state.activeRun) {
    loadSeries();
    renderTable();
  }
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
el.themeToggle.addEventListener("click", toggleTheme);
el.chartMode.addEventListener("change", () => drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value));
el.resetZoom.addEventListener("click", () => drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value));
el.toggleChannels.addEventListener("click", toggleAllChannels);
el.compareToggle.addEventListener("click", () => setCompareMode(!state.compareMode));
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
window.addEventListener("resize", () => {
  drawChart(el.canvas, el.legend, el.mainTooltip, state.series, el.chartMode.value);
  drawSimulation();
});

applyTheme(localStorage.getItem("deepvac-theme") || "dark");
setCompareMode(false);

loadRuns().catch((error) => {
  el.activeRunTitle.textContent = "Unable to load runs";
  el.summaryStrip.innerHTML = `<div class="stat"><span>Error</span><strong>${error.message}</strong></div>`;
});
