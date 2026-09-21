"use strict";

const state = {
  metadata: null,
  national: [],
  coverage: [],
  catalog: [],
  regional: [],
  stateMap: null,
  nationalPage: 1,
  nationalPageSize: 50,
  catalogPage: 1,
  catalogPageSize: 50,
};

const targetLabels = {
  "lower-5%": "Lower-5%",
  "lower-10%": "Lower-10%",
  "lower-20%": "Lower-20%",
};

const targetNumbers = {
  "lower-5%": 0.95,
  "lower-10%": 0.90,
  "lower-20%": 0.80,
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value, digits = 1) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
}

function percentagePoints(value, digits = 1) {
  return value == null ? "—" : `${Number(value).toFixed(digits)} pp`;
}

function scientificName(name) {
  return `<em class="scientific-name">${escapeHtml(name)}</em>`;
}

const supportLabels = {
  very_sparse_1_4: "Very limited (1-4 protective contexts)",
  sparse_5_19: "Limited (5-19 protective contexts)",
  moderate_20_49: "Moderate (20-49 protective contexts)",
  strong_50_plus: "Strong (50+ protective contexts)",
};

function supportLabel(tier) {
  return supportLabels[String(tier)] || String(tier).replaceAll("_", " ");
}

function supportBadge(tier) {
  const limited = String(tier).startsWith("sparse") || String(tier).startsWith("very_sparse");
  return `<span class="badge ${limited ? "limited" : ""}">${escapeHtml(supportLabel(tier))}</span>`;
}

function statusBadges(row) {
  const badges = [];
  if (row.top5) badges.push('<span class="badge top">Top-5</span>');
  else if (row.top10) badges.push('<span class="badge top">Top-10</span>');
  if (row.official_or_common_method_species) badges.push('<span class="badge">method species</span>');
  return badges.join(" ") || '<span class="muted">—</span>';
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function showSection(name) {
  document.querySelectorAll(".page-section").forEach((section) => {
    section.classList.toggle("active", section.id === `section-${name}`);
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    const active = button.dataset.section === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  window.scrollTo({ top: 0, behavior: "auto" });
}

function populateSelect(select, values, allLabel, formatter = (value) => value.replaceAll("_", " ")) {
  const current = select.value;
  select.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>`;
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = formatter(value);
    select.append(option);
  });
  select.value = current;
}

function top5Capture(label) {
  const target = targetNumbers[label];
  return state.coverage.find(
    (row) => Number(row.panel_size) === 5 && Math.abs(Number(row.evaluation_target) - target) < 1e-8,
  );
}

function renderOverview() {
  const metadata = state.metadata;
  byId("header-version").textContent = `${metadata.site_version} · ${metadata.analysis_version}`;
  byId("overview-analysis").textContent = metadata.analysis_version;
  byId("overview-freeze").textContent = metadata.data_freeze_date;
  byId("metric-lower5").textContent = percent(top5Capture("lower-5%")?.expected_capture);
  byId("metric-lower10").textContent = percent(top5Capture("lower-10%")?.expected_capture);
  byId("metric-lower20").textContent = percent(top5Capture("lower-20%")?.expected_capture);
  byId("metric-states").textContent = String(state.regional.filter((row) => row.status === "supported").length);
}

function selectedNationalRows() {
  const query = byId("national-search").value.trim().toLowerCase();
  const taxClass = byId("national-class").value;
  const support = byId("national-support").value;
  return state.national.filter((row) => {
    return (!query || row.scientific_name.toLowerCase().includes(query))
      && (!taxClass || row.class === taxClass)
      && (!support || row.evidence_support === support);
  });
}

function greedyGain(value) {
  if (value == null) return "—";
  const percentagePointGain = Number(value) * 100;
  if (percentagePointGain >= 0.001) return `${percentagePointGain.toFixed(3)} pp`;
  return `${percentagePointGain.toExponential(2)} pp`;
}

function syncPageJump(inputId, currentPage, totalPages) {
  const input = byId(inputId);
  input.max = String(totalPages);
  input.value = String(currentPage);
  input.setCustomValidity("");
}

function requestedPage(inputId, totalPages) {
  const input = byId(inputId);
  const page = Number(input.value);
  if (!Number.isInteger(page) || page < 1 || page > totalPages) {
    input.setCustomValidity(`Enter a whole-number page from 1 to ${totalPages}.`);
    input.reportValidity();
    input.focus();
    return null;
  }
  input.setCustomValidity("");
  return page;
}

function renderNational() {
  const target = byId("national-target").value;
  const panelSize = Number(byId("panel-size").value);
  byId("panel-size-output").value = String(panelSize);
  byId("panel-size-output").textContent = panelSize.toLocaleString();
  const curve = state.coverage.find(
    (row) => Number(row.panel_size) === panelSize
      && Math.abs(Number(row.evaluation_target) - targetNumbers[target]) < 1e-8,
  );
  byId("national-k").textContent = `Top-${panelSize.toLocaleString()}`;
  byId("national-capture").textContent = percent(curve?.expected_capture);
  byId("national-target-label").textContent = target;
  byId("national-added").textContent = curve?.added_species || "—";

  const rows = selectedNationalRows();
  const pages = Math.max(1, Math.ceil(rows.length / state.nationalPageSize));
  state.nationalPage = Math.min(state.nationalPage, pages);
  const start = (state.nationalPage - 1) * state.nationalPageSize;
  const visible = rows.slice(start, start + state.nationalPageSize);
  byId("national-count").textContent = rows.length.toLocaleString();
  byId("national-page").textContent = `Page ${state.nationalPage} of ${pages}`;
  syncPageJump("national-page-input", state.nationalPage, pages);
  byId("national-prev").disabled = state.nationalPage <= 1;
  byId("national-next").disabled = state.nationalPage >= pages;

  byId("national-table-body").innerHTML = visible.map((row) => {
    const selected = row.national_rank <= panelSize;
    const scope = selected
      ? '<span class="badge top">in selected prefix</span>'
      : row.national_rank <= 20
        ? '<span class="badge">frozen Top-20</span>'
        : '<span class="badge">extended evaluation</span>';
    return `<tr>
      <td>${row.national_rank}</td>
      <td>${scientificName(row.scientific_name)}</td>
      <td>${escapeHtml(row.class)}<br><span class="muted">${escapeHtml(row.family)}</span></td>
      <td>${supportBadge(row.evidence_support)}</td>
      <td>${row.direct_protective_context_count}</td>
      <td>${row.direct_protective_chemical_count}</td>
      <td>${greedyGain(row.selection_incremental_gain_lower5)}</td>
      <td>${percent(row.cumulative_expected_capture[target])}</td>
      <td>${scope}</td>
    </tr>`;
  }).join("");
}

function chartMarker(shape, x, y, color) {
  if (shape === "square") {
    return `<rect x="${x - 3.5}" y="${y - 3.5}" width="7" height="7" fill="white" stroke="${color}" stroke-width="2"/>`;
  }
  if (shape === "triangle") {
    return `<path d="M ${x} ${y - 4.5} L ${x + 4.2} ${y + 3.5} L ${x - 4.2} ${y + 3.5} Z" fill="white" stroke="${color}" stroke-width="2"/>`;
  }
  return `<circle cx="${x}" cy="${y}" r="3.7" fill="white" stroke="${color}" stroke-width="2"/>`;
}

function renderCoverageChart() {
  const width = 920;
  const height = 420;
  const margin = { left: 68, right: 28, top: 45, bottom: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxPanelSize = Math.max(...state.coverage.map((row) => Number(row.panel_size)));
  const logMaximum = Math.log(maxPanelSize);
  const xScale = (k) => margin.left + (Math.log(Number(k)) / logMaximum) * plotWidth;
  const yScale = (value) => margin.top + (1 - Number(value)) * plotHeight;
  const xTicks = [...new Set([1, 5, 20, 100, 500, 1000, maxPanelSize])]
    .filter((tick) => tick <= maxPanelSize);
  const specs = [
    { label: "lower-5%", color: "#b64342", dash: "", shape: "circle" },
    { label: "lower-10%", color: "#3775ba", dash: "9 5", shape: "square" },
    { label: "lower-20%", color: "#42949e", dash: "2 5", shape: "triangle" },
  ];
  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" aria-labelledby="coverage-title coverage-desc">
    <title id="coverage-title">Cumulative expected capture across the complete fixed national sequence</title>
    <desc id="coverage-desc">Three curves show lower-5%, lower-10%, and lower-20% evaluations of all 2,144 prefixes of the same lower-5% national sequence on a logarithmic panel-size axis.</desc>`;
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const y = yScale(tick);
    svg += `<line class="chart-grid" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"/>`;
    svg += `<text class="chart-label" x="${margin.left - 12}" y="${y + 4}" text-anchor="end">${Math.round(tick * 100)}%</text>`;
  });
  xTicks.forEach((tick) => {
    const x = xScale(tick);
    svg += `<line class="chart-axis" x1="${x}" y1="${height - margin.bottom}" x2="${x}" y2="${height - margin.bottom + 5}"/>`;
    svg += `<text class="chart-label" x="${x}" y="${height - margin.bottom + 22}" text-anchor="middle">${tick}</text>`;
  });
  svg += `<line class="chart-axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/>
    <line class="chart-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"/>
    <text class="chart-label" x="${margin.left + plotWidth / 2}" y="${height - 14}" text-anchor="middle">Fixed national sequence prefix size (log scale)</text>
    <text class="chart-label" transform="translate(18 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle">Expected capture</text>`;

  specs.forEach((spec, index) => {
    const rows = state.coverage
      .filter((row) => row.measured_tail === spec.label)
      .sort((a, b) => Number(a.panel_size) - Number(b.panel_size));
    const path = rows.map((row, rowIndex) => `${rowIndex === 0 ? "M" : "L"} ${xScale(Number(row.panel_size)).toFixed(2)} ${yScale(row.expected_capture).toFixed(2)}`).join(" ");
    svg += `<path class="chart-line" d="${path}" stroke="${spec.color}" ${spec.dash ? `stroke-dasharray="${spec.dash}"` : ""}/>`;
    rows.filter((row) => xTicks.includes(Number(row.panel_size))).forEach((row) => {
      svg += chartMarker(spec.shape, xScale(Number(row.panel_size)), yScale(row.expected_capture), spec.color);
    });
    const legendX = margin.left + index * 205;
    const legendY = 20;
    svg += `<line x1="${legendX}" y1="${legendY}" x2="${legendX + 34}" y2="${legendY}" stroke="${spec.color}" stroke-width="2.5" ${spec.dash ? `stroke-dasharray="${spec.dash}"` : ""}/>`;
    svg += chartMarker(spec.shape, legendX + 17, legendY, spec.color);
    svg += `<text class="chart-label" x="${legendX + 43}" y="${legendY + 4}">${targetLabels[spec.label]}</text>`;
  });
  svg += "</svg>";
  byId("coverage-chart").innerHTML = svg;
}

function filteredCatalog() {
  const query = byId("catalog-search").value.trim().toLowerCase();
  const taxClass = byId("catalog-class").value;
  const support = byId("catalog-support").value;
  const status = byId("catalog-status").value;
  return state.catalog.filter((row) => {
    let statusMatch = true;
    if (status === "top5") statusMatch = row.top5;
    if (status === "top10") statusMatch = row.top10;
    if (status === "top20") statusMatch = row.national_rank != null && row.national_rank <= 20;
    if (status === "extended") statusMatch = row.national_rank != null && row.national_rank > 20;
    return statusMatch
      && (!query || row.scientific_name.toLowerCase().includes(query))
      && (!taxClass || row.class === taxClass)
      && (!support || row.evidence_support === support);
  });
}

function renderCatalog() {
  const rows = filteredCatalog();
  const pages = Math.max(1, Math.ceil(rows.length / state.catalogPageSize));
  state.catalogPage = Math.min(state.catalogPage, pages);
  const start = (state.catalogPage - 1) * state.catalogPageSize;
  const visible = rows.slice(start, start + state.catalogPageSize);
  byId("catalog-count").textContent = rows.length.toLocaleString();
  byId("catalog-page").textContent = `Page ${state.catalogPage} of ${pages}`;
  syncPageJump("catalog-page-input", state.catalogPage, pages);
  byId("catalog-prev").disabled = state.catalogPage <= 1;
  byId("catalog-next").disabled = state.catalogPage >= pages;
  byId("catalog-table-body").innerHTML = visible.map((row) => `<tr>
    <td>${scientificName(row.scientific_name)} ${statusBadges(row)}</td>
    <td>${row.national_rank ?? "—"}</td>
    <td>${escapeHtml(row.class)}<br><span class="muted">${escapeHtml(row.family)}</span></td>
    <td>${row.protective_context_count}</td>
    <td>${row.protective_chemical_count}</td>
    <td>${row.warning_context_count}</td>
    <td>${row.eligible_state_count}</td>
    <td>${supportBadge(row.evidence_support)}</td>
    <td>${escapeHtml(row.note || "")}</td>
  </tr>`).join("");
}

function regionalRow(stateCode) {
  return state.regional.find((row) => row.state_code === stateCode);
}

function updateStateMapSelection() {
  const selectedCode = byId("state-select").value;
  document.querySelectorAll(".state-group").forEach((group) => {
    const selected = group.dataset.stateCode === selectedCode;
    group.classList.toggle("is-selected", selected);
    group.classList.toggle("is-dimmed", Boolean(selectedCode) && !selected);
    group.setAttribute("aria-pressed", selected ? "true" : "false");
  });
}

function positionMapTooltip(clientX, clientY) {
  const wrap = byId("state-map-wrap");
  const tooltip = byId("state-map-tooltip");
  const bounds = wrap.getBoundingClientRect();
  const tooltipWidth = tooltip.offsetWidth || 300;
  const tooltipHeight = tooltip.offsetHeight || 190;
  let left = clientX - bounds.left + 16;
  let top = clientY - bounds.top + 16;
  if (left + tooltipWidth > bounds.width - 8) left = clientX - bounds.left - tooltipWidth - 16;
  if (top + tooltipHeight > bounds.height - 8) top = bounds.height - tooltipHeight - 8;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function showMapTooltip(stateCode, eventOrElement) {
  const row = regionalRow(stateCode);
  if (!row) return;
  const tooltip = byId("state-map-tooltip");
  const species = row.display_top5?.length ? row.display_top5 : row.national_top5;
  const items = species.map((name) => `<li>${scientificName(name)}</li>`).join("");
  tooltip.innerHTML = `<strong>${escapeHtml(row.state_name)} (${escapeHtml(row.state_code)})</strong>
    <span class="tooltip-basis">${escapeHtml(row.top5_basis)}</span>
    <ol>${items}</ol>
    <span class="tooltip-support">${row.eligible_priority_chemical_count} eligible priority chemicals</span>`;
  tooltip.classList.remove("hidden");
  if (typeof eventOrElement?.clientX === "number" && eventOrElement.clientX > 0) {
    positionMapTooltip(eventOrElement.clientX, eventOrElement.clientY);
  } else {
    const bounds = eventOrElement.getBoundingClientRect();
    positionMapTooltip(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
  }
}

function hideMapTooltip() {
  byId("state-map-tooltip").classList.add("hidden");
}

function selectRegionalState(stateCode) {
  const select = byId("state-select");
  if (![...select.options].some((option) => option.value === stateCode)) return;
  select.value = stateCode;
  renderRegional();
}

function renderStateMap() {
  const map = state.stateMap;
  const regionalByCode = new Map(state.regional.map((row) => [row.state_code, row]));
  const [minX, minY, width, height] = map.view_box;
  const groups = map.states.map((geometry) => {
    const row = regionalByCode.get(geometry.state_code);
    const availability = row?.status === "supported" ? "supported" : "unavailable";
    const connector = geometry.callout
      ? `<line class="state-callout-line" x1="${geometry.centroid_x}" y1="${geometry.centroid_y}" x2="${geometry.label_x - 20}" y2="${geometry.label_y}"/>`
      : "";
    const label = geometry.callout
      ? `<rect class="state-callout-box" x="${geometry.label_x - 18}" y="${geometry.label_y - 10}" width="36" height="20" rx="5"/>
         <text class="state-label callout" x="${geometry.label_x}" y="${geometry.label_y + 4}">${escapeHtml(geometry.state_code)}</text>`
      : `<text class="state-label" x="${geometry.label_x}" y="${geometry.label_y + 4}">${escapeHtml(geometry.state_code)}</text>`;
    const basis = row?.status === "supported" ? "localized Top-5 available" : "national default Top-5 shown";
    return `<g class="state-group ${availability}" data-state-code="${escapeHtml(geometry.state_code)}" role="button" tabindex="0" aria-pressed="false" aria-label="${escapeHtml(geometry.state_name)}, ${basis}">
      <path class="state-shape" d="${geometry.path}" fill-rule="evenodd"/>
      ${connector}${label}
    </g>`;
  }).join("");
  byId("state-map").innerHTML = `<svg viewBox="${minX} ${minY} ${width} ${height}" xmlns="http://www.w3.org/2000/svg" aria-labelledby="state-map-title state-map-desc">
    <title id="state-map-title">Interactive United States regional localization map</title>
    <desc id="state-map-desc">Select any state or the District of Columbia. Hover or focus to view its localized or national-default Top-5 species.</desc>
    ${groups}
  </svg>`;

  document.querySelectorAll(".state-group").forEach((group) => {
    const stateCode = group.dataset.stateCode;
    group.addEventListener("click", () => selectRegionalState(stateCode));
    group.addEventListener("pointerenter", (event) => showMapTooltip(stateCode, event));
    group.addEventListener("pointermove", (event) => positionMapTooltip(event.clientX, event.clientY));
    group.addEventListener("pointerleave", hideMapTooltip);
    group.addEventListener("focus", () => showMapTooltip(stateCode, group));
    group.addEventListener("blur", hideMapTooltip);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRegionalState(stateCode);
      }
    });
  });
  updateStateMapSelection();
}

function populateStateSelect() {
  const select = byId("state-select");
  select.innerHTML = "";
  state.regional
    .slice()
    .sort((a, b) => a.state_name.localeCompare(b.state_name))
    .forEach((row) => {
      const option = document.createElement("option");
      option.value = row.state_code;
      option.textContent = `${row.state_name} (${row.state_code})${row.status === "supported" ? "" : " — insufficient support"}`;
      select.append(option);
    });
  if (state.regional.some((row) => row.state_code === "CT")) select.value = "CT";
}

function renderRegional() {
  const row = state.regional.find((item) => item.state_code === byId("state-select").value);
  if (!row) return;
  updateStateMapSelection();
  const status = byId("regional-status");
  if (row.status !== "supported") {
    status.className = "status-panel warning";
    status.innerHTML = `<strong>${escapeHtml(row.state_name)}:</strong> ${escapeHtml(row.reason)} Eligible priority chemicals: ${row.eligible_priority_chemical_count}.`;
    byId("regional-supported").classList.add("hidden");
    return;
  }
  status.className = "status-panel";
  status.innerHTML = `<strong>${escapeHtml(row.state_name)}:</strong> localized lower-5% Top-5 available. ${escapeHtml(row.weighting_summary)}`;
  byId("regional-supported").classList.remove("hidden");
  byId("regional-localized").textContent = percent(row.localized_expected_capture);
  byId("regional-national").textContent = percent(row.national_expected_capture);
  byId("regional-gain").textContent = percentagePoints(row.absolute_gain_percentage_points);
  byId("regional-overlap").textContent = `${row.overlap_count}/5`;
  byId("regional-chemicals").textContent = String(row.eligible_priority_chemical_count);
  byId("localized-top5").innerHTML = row.localized_top5.map((name) => `<li>${scientificName(name)}${row.overlap_species.includes(name) ? ' <span class="badge top">overlap</span>' : ""}</li>`).join("");
  byId("national-top5").innerHTML = row.national_top5.map((name) => `<li>${scientificName(name)}${row.overlap_species.includes(name) ? ' <span class="badge top">overlap</span>' : ""}</li>`).join("");
  byId("regional-table-body").innerHTML = row.localized_sequence.map((member) => `<tr>
    <td>${member.rank}</td>
    <td>${scientificName(member.scientific_name)}</td>
    <td>${escapeHtml(member.candidate_status.replaceAll("_", " "))}</td>
    <td>${Number(member.species_relevance_weight).toFixed(3)}</td>
    <td>${percent(member.mean_tail_probability, 2)}</td>
    <td>${percent(member.mean_reliability, 1)}</td>
    <td>${member.direct_support_chemical_count}</td>
  </tr>`).join("");
}

function renderVersion() {
  const metadata = state.metadata;
  const rows = [
    ["Site version", metadata.site_version],
    ["Analysis version", metadata.analysis_version],
    ["Data freeze", metadata.data_freeze_date],
    ["Manuscript source", metadata.manuscript_version],
    ["Supporting Information source", metadata.supporting_information_version],
    ["Generated UTC", metadata.generated_at_utc],
    ["Git commit", metadata.git_commit || "Pending first approved commit"],
  ];
  byId("version-metadata").innerHTML = rows.map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  byId("manuscript-title").textContent = metadata.manuscript_title;
  byId("author-list").textContent = metadata.authors.join(", ");
  byId("source-identifier").textContent = metadata.source_analysis_identifier;
  if (metadata.github_release) {
    byId("github-release").textContent = "COMPASS v1.0.1";
    byId("github-release").href = metadata.github_release;
  }
  byId("zenodo-doi").textContent = metadata.zenodo_doi || metadata.zenodo_status || "No DOI assigned.";
  byId("site-license").textContent = metadata.license || "Publicly viewable source; all rights reserved.";
}

function csvEscape(value) {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadCsv(filename, rows, columns) {
  const lines = [columns.map(csvEscape).join(",")];
  rows.forEach((row) => lines.push(columns.map((column) => csvEscape(row[column])).join(",")));
  const blob = new Blob([`${lines.join("\n")}\n`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function wireEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => showSection(button.dataset.section));
  });
  ["national-target", "panel-size"].forEach((id) => {
    byId(id).addEventListener("input", renderNational);
  });
  ["national-search", "national-class", "national-support"].forEach((id) => {
    byId(id).addEventListener("input", () => {
      state.nationalPage = 1;
      renderNational();
    });
  });
  ["catalog-search", "catalog-class", "catalog-support", "catalog-status"].forEach((id) => {
    byId(id).addEventListener("input", () => {
      state.catalogPage = 1;
      renderCatalog();
    });
  });
  byId("national-prev").addEventListener("click", () => { state.nationalPage -= 1; renderNational(); });
  byId("national-next").addEventListener("click", () => { state.nationalPage += 1; renderNational(); });
  byId("catalog-prev").addEventListener("click", () => { state.catalogPage -= 1; renderCatalog(); });
  byId("catalog-next").addEventListener("click", () => { state.catalogPage += 1; renderCatalog(); });
  byId("national-page-jump").addEventListener("submit", (event) => {
    event.preventDefault();
    const pages = Math.max(1, Math.ceil(selectedNationalRows().length / state.nationalPageSize));
    const page = requestedPage("national-page-input", pages);
    if (page == null) return;
    state.nationalPage = page;
    renderNational();
  });
  byId("catalog-page-jump").addEventListener("submit", (event) => {
    event.preventDefault();
    const pages = Math.max(1, Math.ceil(filteredCatalog().length / state.catalogPageSize));
    const page = requestedPage("catalog-page-input", pages);
    if (page == null) return;
    state.catalogPage = page;
    renderCatalog();
  });
  ["national-page-input", "catalog-page-input"].forEach((id) => {
    byId(id).addEventListener("input", () => byId(id).setCustomValidity(""));
  });
  byId("state-select").addEventListener("change", renderRegional);
  byId("download-current-national").addEventListener("click", () => {
    const target = byId("national-target").value;
    const rows = selectedNationalRows().map((row) => ({
      national_rank: row.national_rank,
      scientific_name: row.scientific_name,
      class: row.class,
      family: row.family,
      protective_evidence_volume: row.evidence_support_label,
      protective_contexts: row.direct_protective_context_count,
      protective_chemicals: row.direct_protective_chemical_count,
      greedy_incremental_gain_lower5: row.selection_incremental_gain_lower5,
      evaluation_target: target,
      cumulative_expected_capture: row.cumulative_expected_capture[target],
      rank_scope: row.rank_scope,
    }));
    downloadCsv("compass_national_current_view.csv", rows, Object.keys(rows[0] || {}));
  });
  byId("download-current-catalog").addEventListener("click", () => {
    const rows = filteredCatalog();
    const columns = ["scientific_name", "national_rank", "class", "family", "protective_context_count", "protective_chemical_count", "warning_context_count", "eligible_state_count", "evidence_support_label", "rank_scope", "note"];
    downloadCsv("compass_species_catalog_current_view.csv", rows, columns);
  });
}

async function initialize() {
  try {
    const [metadata, national, coverage, catalog, regional, stateMap] = await Promise.all([
      loadJson("site_data/metadata.json"),
      loadJson("site_data/national_sequence.json"),
      loadJson("site_data/coverage.json"),
      loadJson("site_data/species_catalog.json"),
      loadJson("site_data/regional.json"),
      loadJson("site_data/state_map.json"),
    ]);
    Object.assign(state, { metadata, national, coverage, catalog, regional, stateMap });
    byId("panel-size").max = String(national.length);
    const nationalClasses = [...new Set(national.map((row) => row.class).filter(Boolean))].sort();
    const nationalSupport = [...new Set(national.map((row) => row.evidence_support).filter(Boolean))].sort();
    const catalogClasses = [...new Set(catalog.map((row) => row.class).filter(Boolean))].sort();
    const catalogSupport = [...new Set(catalog.map((row) => row.evidence_support).filter(Boolean))].sort();
    populateSelect(byId("national-class"), nationalClasses, "All classes");
    populateSelect(byId("national-support"), nationalSupport, "All evidence volumes", supportLabel);
    populateSelect(byId("catalog-class"), catalogClasses, "All classes");
    populateSelect(byId("catalog-support"), catalogSupport, "All evidence volumes", supportLabel);
    populateStateSelect();
    wireEvents();
    renderOverview();
    renderNational();
    renderCoverageChart();
    renderCatalog();
    renderStateMap();
    renderRegional();
    renderVersion();
  } catch (error) {
    const panel = byId("load-error");
    panel.classList.remove("hidden");
    panel.textContent = `The frozen site data could not be loaded. Serve the built folder through a local web server and verify the site_data directory. ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", initialize);
