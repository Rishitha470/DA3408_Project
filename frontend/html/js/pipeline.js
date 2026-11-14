/* Pipeline console: renders GET /api/v1/pipeline and refreshes every 15 seconds. */
(function () {
  "use strict";
  const API = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v, d = 3) => (typeof v === "number" ? Number(v.toFixed(d)).toLocaleString() : esc(v ?? "-"));
  const when = (iso) => (iso ? new Date(iso).toLocaleString() : "-");

  function renderDag(dag) {
    const order = dag.stages.map((s) => s.name);
    $("dag").innerHTML = order.map((name, i) => {
      const st = dag.stages.find((s) => s.name === name), r = st.last_run;
      const cls = r ? r.status : "never";
      const body = r
        ? `<span class="pill ${esc(r.status)}">${esc(r.status)}</span>
           <div class="small">${fmt(r.duration_s, 1)} s${r.rows_per_s ? ` | ${fmt(r.rows_per_s, 0)} rows/s` : ""}</div>
           <div class="small">${when(r.finished_at)}</div>`
        : `<div class="small">not run yet</div>`;
      return (i ? `<div class="dag-arrow" aria-hidden="true"></div>` : "") +
        `<div class="dag-node ${cls}"><div class="name">${esc(name)}</div>${body}</div>`;
    }).join("");
  }

  function renderSummary(s, other) {
    $("run-summary").innerHTML = `
      <div class="kpi"><span class="v">${s.total}</span><span class="k">recorded runs</span></div>
      <div class="kpi"><span class="v">${s.success}</span><span class="k">successful</span></div>
      <div class="kpi"><span class="v">${s.failed}</span><span class="k">failed</span></div>`;
    const jobs = Object.entries(other || {});
    $("other-jobs").innerHTML = jobs.length ? "Other jobs: " + jobs.map(([n, r]) =>
      `${esc(n)} <span class="pill ${esc(r.status)}">${esc(r.status)}</span> (${when(r.finished_at)})`).join(", ") : "";
  }

  function renderData(d) {
    if (!d) { $("data").innerHTML = "<p>No data report yet. Run <code>dvc repro</code>.</p>"; return; }
    const checks = Object.entries(d.quality_checks || {}).map(([k, ok]) =>
      `<tr><td>${esc(k.replaceAll("_", " "))}</td><td><span class="pill ${ok ? "passed" : "failed"}">${ok ? "passed" : "failed"}</span></td></tr>`).join("");
    $("data").innerHTML = `<div class="kpis">
        <div class="kpi"><span class="v">${fmt(d.rows_in, 0)}</span><span class="k">rows in</span></div>
        <div class="kpi"><span class="v">${fmt(d.rows_out, 0)}</span><span class="k">rows kept</span></div>
        <div class="kpi"><span class="v">${fmt(d.ray_processing_rows_per_s, 0)}</span><span class="k">rows/s (Ray)</span></div>
        <div class="kpi"><span class="v">${fmt(d.positive_rate * 100, 1)}%</span><span class="k">with diabetes</span></div></div>
      <p class="small">Invalid rows dropped: ${fmt(d.invalid_rows_dropped, 0)} | duplicates: ${fmt(d.duplicate_rows, 0)}
        | split: ${fmt(d.split_rows.train, 0)} train / ${fmt(d.split_rows.val, 0)} validation / ${fmt(d.split_rows.test, 0)} test
        | ${d.ray.num_cpus} CPUs, ${d.ray.num_blocks} blocks, total ${fmt(d.timings_s.total_s, 1)} s</p>
      <table><thead><tr><th>Quality check</th><th>Result</th></tr></thead><tbody>${checks}</tbody></table>`;
  }

  function renderTest(m, acc) {
    if (!m) { $("test").innerHTML = "<p>No evaluation yet.</p>"; return; }
    const rows = acc ? Object.entries(acc.checks).map(([k, c]) =>
      `<tr><td>${esc(k.replace("min_", ""))} at least ${c.required}</td><td>${fmt(c.actual, 3)}</td>
       <td><span class="pill ${c.passed ? "passed" : "failed"}">${c.passed ? "passed" : "failed"}</span></td></tr>`).join("") : "";
    $("test").innerHTML = `<div class="kpis">
        <div class="kpi"><span class="v">${fmt(m.roc_auc)}</span><span class="k">ROC-AUC</span></div>
        <div class="kpi"><span class="v">${fmt(m.recall)}</span><span class="k">recall</span></div>
        <div class="kpi"><span class="v">${fmt(m.precision)}</span><span class="k">precision</span></div>
        <div class="kpi"><span class="v">${fmt(m.f1)}</span><span class="k">F1</span></div></div>
      <p class="small">Model ${esc(m.model_version)}${m.registry ? ` | registry: ${esc(m.registry.name)} v${esc(m.registry.version)} @${esc(m.registry.alias)}` : ""}
        | missed cases: ${fmt(m.false_negatives, 0)} of ${fmt(m.false_negatives + m.true_positives, 0)}</p>
      <table><thead><tr><th>Acceptance criterion</th><th>Actual</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderModels(c) {
    if (!c) { $("models").innerHTML = "<p>No training run yet.</p>"; return; }
    const cols = ["roc_auc", "recall", "precision", "f1", "accuracy", "threshold", "train_time_s", "latency_p95_ms"];
    $("models").innerHTML = `<table><thead><tr><th>Model</th>${cols.map((k) => `<th>${k.replaceAll("_", " ")}</th>`).join("")}</tr></thead>
      <tbody>${c.candidates.map((r) => `<tr class="${r.name === c.best ? "best" : ""}"><td>${esc(r.name)}</td>
        ${cols.map((k) => `<td>${fmt(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  function renderDrift(d) {
    if (!d || !d.features || !Object.keys(d.features).length) {
      $("drift").innerHTML = `<p>Not enough recent predictions yet (${d ? d.n_samples || 0 : 0} so far; at least
        ${d && d.min_samples ? d.min_samples : 100} needed).</p>`;
      return;
    }
    const rows = Object.entries(d.features).sort((a, b) => b[1].psi - a[1].psi).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${fmt(v.psi, 3)}</td><td><span class="pill ${esc(v.level)}">${esc(v.level)}</span></td></tr>`).join("");
    $("drift").innerHTML = `<p>${d.drift_detected
        ? `<span class="pill failed">drift detected</span> in ${d.drifted_features.map(esc).join(", ")}. Retraining is recommended.`
        : `<span class="pill passed">no significant drift</span>`} Based on the last ${fmt(d.n_samples, 0)} predictions,
        compared with the training data (PSI: below 0.1 stable, 0.1-0.2 moderate, 0.2 or more significant).</p>
      <div class="table-wrap"><table><thead><tr><th>Feature</th><th>PSI</th><th>Level</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function renderRuns(runs) {
    if (!runs.length) { $("runs").innerHTML = "<p>No runs recorded yet.</p>"; return; }
    $("runs").innerHTML = `<table><thead><tr><th>Stage</th><th>Status</th><th>Finished</th><th>Duration (s)</th>
      <th>Rows</th><th>Rows/s</th><th>Error</th></tr></thead><tbody>${runs.map((r) => `<tr>
      <td>${esc(r.stage)}</td><td><span class="pill ${esc(r.status)}">${esc(r.status)}</span></td><td>${when(r.finished_at)}</td>
      <td>${fmt(r.duration_s, 2)}</td><td>${fmt(r.rows_out, 0)}</td><td>${fmt(r.rows_per_s, 0)}</td>
      <td style="white-space:normal">${esc(r.error || "")}</td></tr>`).join("")}</tbody></table>`;
  }

  async function load() {
    try {
      const res = await fetch(`${API}/api/v1/pipeline`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const p = await res.json();
      renderDag(p.dag); renderSummary(p.summary, p.other_jobs); renderData(p.data_quality);
      renderTest(p.test_metrics, p.acceptance); renderModels(p.model_comparison); renderDrift(p.drift);
      renderRuns(p.runs);
      $("tools").innerHTML = Object.entries(p.tools || {}).map(([n, u]) =>
        `<a href="${esc(u)}" target="_blank" rel="noopener">Open ${esc(n)}</a>`).join("");
      $("error").hidden = true;
      $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}. Refreshes every 15 seconds.`;
    } catch (err) {
      $("error").textContent = `Could not load pipeline data from ${API} (${err.message}). Is the backend running?`;
      $("error").hidden = false;
    }
  }
  load();
  setInterval(load, 15000);
})();
