/* Risk check wizard: collects the 21 answers, calls POST /api/v1/predict, shows the result. */
(function () {
  "use strict";
  const API = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
  const YES_NO = [["Yes", 1], ["No", 0]];

  const AGE_GROUPS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80 or older"].map((l, i) => [l, i + 1]);
  const EDUCATION = ["Never attended school / kindergarten only", "Primary school (grades 1-8)",
    "Some high school (grades 9-11)", "Finished high school", "Some college or diploma",
    "College degree or higher"].map((l, i) => [l, i + 1]);
  const INCOME = ["Less than $10,000", "$10,000 - $15,000", "$15,000 - $20,000", "$20,000 - $25,000",
    "$25,000 - $35,000", "$35,000 - $50,000", "$50,000 - $75,000", "$75,000 or more"].map((l, i) => [l, i + 1]);
  const GEN_HEALTH = [["Excellent", 1], ["Very good", 2], ["Good", 3], ["Fair", 4], ["Poor", 5]];

  // Each step is a list of questions. type: choice (buttons), select, days (0-30), bmi.
  const STEPS = [
    { title: "About you", questions: [
      { key: "Age", type: "select", label: "How old are you?", options: AGE_GROUPS },
      { key: "Sex", type: "choice", label: "Sex", hint: "As recorded in the original health survey.",
        options: [["Female", 0], ["Male", 1]] } ] },
    { title: "Your body", questions: [
      { key: "BMI", type: "bmi", label: "Height and weight",
        hint: "We use these to work out your body mass index (BMI). Nothing else is stored." } ] },
    { title: "Health conditions", questions: [
      { key: "HighBP", type: "choice", label: "Has a doctor told you that you have high blood pressure?", options: YES_NO },
      { key: "HighChol", type: "choice", label: "Has a doctor told you that you have high cholesterol?", options: YES_NO },
      { key: "CholCheck", type: "choice", label: "Have you had your cholesterol checked in the last 5 years?", options: YES_NO },
      { key: "Stroke", type: "choice", label: "Have you ever had a stroke?", options: YES_NO },
      { key: "HeartDiseaseorAttack", type: "choice", label: "Have you had heart disease or a heart attack?", options: YES_NO },
      { key: "DiffWalk", type: "choice", label: "Do you have serious difficulty walking or climbing stairs?", options: YES_NO } ] },
    { title: "Daily habits", questions: [
      { key: "PhysActivity", type: "choice", label: "Did you do any exercise or sport in the last 30 days?",
        hint: "Not counting your job.", options: YES_NO },
      { key: "Fruits", type: "choice", label: "Do you eat fruit at least once a day?", options: YES_NO },
      { key: "Veggies", type: "choice", label: "Do you eat vegetables at least once a day?", options: YES_NO },
      { key: "Smoker", type: "choice", label: "Have you smoked at least 100 cigarettes in your life?",
        hint: "That is about 5 packs.", options: YES_NO },
      { key: "HvyAlcoholConsump", type: "choice", label: "Do you drink heavily?",
        hint: "Men: more than 14 drinks a week. Women: more than 7 drinks a week.", options: YES_NO } ] },
    { title: "How you have been feeling", questions: [
      { key: "GenHlth", type: "choice", label: "In general, how would you rate your health?", options: GEN_HEALTH },
      { key: "PhysHlth", type: "days", label: "In the last 30 days, on how many days was your physical health not good?",
        hint: "Enter 0 to 30. Include illness and injury." },
      { key: "MentHlth", type: "days", label: "In the last 30 days, on how many days was your mental health not good?",
        hint: "Enter 0 to 30. Include stress, low mood and worry." } ] },
    { title: "Health care and background", questions: [
      { key: "AnyHealthcare", type: "choice", label: "Do you have any health insurance or health coverage?", options: YES_NO },
      { key: "NoDocbcCost", type: "choice", label: "In the last year, did you skip seeing a doctor because of the cost?", options: YES_NO },
      { key: "Education", type: "select", label: "Highest level of education", options: EDUCATION },
      { key: "Income", type: "select", label: "Yearly household income",
        hint: "In US dollars, as in the survey the model learned from. Pick the closest.", options: INCOME } ] },
  ];

  const ADVICE = {
    HighBP: "Keeping blood pressure under control (with your doctor's help) lowers your risk.",
    HighChol: "Bringing cholesterol down through diet, activity or medicine lowers your risk.",
    BMI: "Moving towards a healthy weight is one of the biggest ways to lower risk.",
    PhysActivity: "Regular activity, even brisk walking, helps your body use sugar better.",
    Fruits: "Eating fruit every day is linked with lower risk.",
    Veggies: "Eating vegetables every day is linked with lower risk.",
    HvyAlcoholConsump: "Cutting down on alcohol lowers your risk.",
    GenHlth: "Improving your overall health makes a difference.",
    PhysHlth: "Fewer days of poor physical health go with lower risk.",
    MentHlth: "Looking after your mental health is linked with lower risk.",
  };

  const answers = {};
  let step = 0;
  const $ = (id) => document.getElementById(id);

  function renderSteps() {
    $("steps").innerHTML = STEPS.map((_, i) =>
      `<li class="${i < step ? "done" : i === step ? "current" : ""}"></li>`).join("");
    $("step-count").textContent = `Step ${step + 1} of ${STEPS.length}: ${STEPS[step].title}`;
    $("back").style.visibility = step === 0 ? "hidden" : "visible";
    $("next").textContent = step === STEPS.length - 1 ? "See my result" : "Next";
  }

  function questionHtml(q) {
    const hint = q.hint ? `<p class="hint">${q.hint}</p>` : `<p class="hint"></p>`;
    if (q.type === "choice") {
      const btns = q.options.map(([label, value]) =>
        `<button type="button" data-key="${q.key}" data-value="${value}"
           aria-pressed="${answers[q.key] === value}">${label}</button>`).join("");
      return `<div class="question" data-q="${q.key}" role="group" aria-labelledby="l-${q.key}">
        <span class="q-label" id="l-${q.key}">${q.label}</span>${hint}<div class="choices">${btns}</div></div>`;
    }
    if (q.type === "select") {
      const opts = q.options.map(([label, value]) =>
        `<option value="${value}" ${answers[q.key] === value ? "selected" : ""}>${label}</option>`).join("");
      return `<div class="question" data-q="${q.key}"><label for="f-${q.key}">${q.label}</label>${hint}
        <select id="f-${q.key}" data-key="${q.key}"><option value="">Choose one</option>${opts}</select></div>`;
    }
    if (q.type === "days") {
      const v = answers[q.key] ?? "";
      return `<div class="question" data-q="${q.key}"><label for="f-${q.key}">${q.label}</label>${hint}
        <input type="number" id="f-${q.key}" data-key="${q.key}" min="0" max="30" step="1" inputmode="numeric" value="${v}"></div>`;
    }
    // bmi: height + weight, or BMI directly
    const h = answers._height ?? "", w = answers._weight ?? "";
    return `<div class="question" data-q="BMI"><span class="q-label">${q.label}</span>${hint}
      <div class="inline-inputs">
        <label>Height (cm)<input type="number" id="f-height" min="100" max="230" step="1" inputmode="numeric" value="${h}"></label>
        <label>Weight (kg)<input type="number" id="f-weight" min="25" max="300" step="0.1" inputmode="decimal" value="${w}"></label>
      </div>
      <p class="bmi-readout" id="bmi-readout"></p>
      <button type="button" class="link-btn" id="bmi-direct">I know my BMI</button>
      <div id="bmi-direct-box" hidden><label for="f-bmi-direct">BMI</label>
        <input type="number" id="f-bmi-direct" min="12" max="98" step="0.1" inputmode="decimal"></div></div>`;
  }

  function bmiCategory(b) {
    return b < 18.5 ? "underweight" : b < 25 ? "a healthy weight" : b < 30 ? "overweight" : "obese";
  }

  function updateBmi() {
    const h = parseFloat($("f-height").value), w = parseFloat($("f-weight").value);
    const direct = parseFloat($("f-bmi-direct").value);
    answers._height = $("f-height").value; answers._weight = $("f-weight").value;
    let bmi = null;
    if (!$("bmi-direct-box").hidden && !isNaN(direct)) bmi = direct;
    else if (h >= 100 && h <= 230 && w >= 25 && w <= 300) bmi = w / Math.pow(h / 100, 2);
    if (bmi !== null && bmi >= 12 && bmi <= 98) {
      answers.BMI = Math.round(bmi * 10) / 10;
      $("bmi-readout").innerHTML = `Your BMI is <strong>${answers.BMI}</strong>, which counts as ${bmiCategory(answers.BMI)}.`;
    } else {
      delete answers.BMI;
      $("bmi-readout").textContent = (h || w || direct) ? "Check the numbers: height 100-230 cm, weight 25-300 kg, BMI 12-98." : "";
    }
  }

  function renderStep() {
    renderSteps();
    $("step-body").innerHTML = `<h2>${STEPS[step].title}</h2>` + STEPS[step].questions.map(questionHtml).join("");
    $("form-error").hidden = true;
    if (STEPS[step].questions[0].type === "bmi") {
      ["f-height", "f-weight", "f-bmi-direct"].forEach((id) => $(id).addEventListener("input", updateBmi));
      $("bmi-direct").addEventListener("click", () => { $("bmi-direct-box").hidden = false; $("f-bmi-direct").focus(); });
      updateBmi();
    }
  }

  // event delegation for choice buttons, selects and number inputs
  $("step-body").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-key]");
    if (!b) return;
    answers[b.dataset.key] = Number(b.dataset.value);
    b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    b.closest(".question").classList.remove("missing");
  });
  function onFieldInput(e) {
    const el = e.target;
    if (!el.dataset.key) return;
    const v = el.value === "" ? null : Number(el.value);
    const valid = v !== null && Number.isInteger(v) && (el.type !== "number" || (v >= 0 && v <= 30));
    if (valid) { answers[el.dataset.key] = v; el.closest(".question").classList.remove("missing"); }
    else delete answers[el.dataset.key];
  }
  $("step-body").addEventListener("change", onFieldInput);
  $("step-body").addEventListener("input", onFieldInput);

  function stepComplete() {
    let ok = true;
    for (const q of STEPS[step].questions) {
      const box = document.querySelector(`[data-q="${q.key}"]`);
      const missing = answers[q.key] === undefined;
      box.classList.toggle("missing", missing);
      if (missing && ok) { ok = false; (box.querySelector("button, select, input") || box).focus(); }
    }
    return ok;
  }

  function payload() {
    const out = {};
    STEPS.forEach((s) => s.questions.forEach((q) => { out[q.key] = answers[q.key]; }));
    return out;
  }

  async function submit() {
    const btn = $("next");
    btn.disabled = true; btn.textContent = "Checking...";
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    try {
      const res = await fetch(`${API}/api/v1/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()), signal: ctrl.signal });
      const body = await res.json().catch(() => ({}));
      if (res.ok) return showResult(body);
      if (res.status === 422) showError("Some answers could not be used. Go back and check them: " +
        (body.details || []).map((d) => d.field).join(", "));
      else if (res.status === 503) showError("The risk model is starting up or busy. Wait a minute and press the button again.");
      else showError("Something went wrong on our side. Please try again. Reference: " + (body.request_id || "n/a"));
    } catch (err) {
      showError("We could not reach the risk service. Check your internet connection, then press the button again.");
    } finally {
      clearTimeout(timer); btn.disabled = false; renderSteps();
    }
  }

  function showError(msg) { const box = $("form-error"); box.textContent = msg; box.hidden = false; }

  function showResult(r) {
    const words = { low: "Low risk", moderate: "Moderate risk", high: "High risk" };
    const explain = {
      low: "Your answers look like those of people who mostly did not have diabetes. Keep up healthy habits and have regular check-ups.",
      moderate: "Some of your answers are linked with diabetes. It is worth talking to a doctor about a blood sugar test at your next visit.",
      high: "Your answers are similar to those of many people who have diabetes. Please see a doctor and ask for a blood sugar (HbA1c) test.",
    };
    // dial zones: low < 0.6*t <= moderate < t <= high; the marker shows this person's score
    const t = r.threshold, lowEnd = 0.6 * t * 100, modEnd = t * 100, pos = Math.min(99.5, Math.max(0.5, r.probability * 100));
    const factors = r.top_factors.length
      ? `<h3>What would lower your risk most</h3><ul class="factors">${r.top_factors.map((f) =>
          `<li><strong>${f.label.replace(/ \(.*\)$/, "")}</strong>${ADVICE[f.feature] || ""}
           <span class="impact">Changing this would lower your score by about ${Math.round(f.impact * 100)} points.</span></li>`).join("")}</ul>`
      : `<h3>What would lower your risk most</h3><p>None of the habits we asked about stands out. Keep doing what you are doing.</p>`;
    const el = $("result");
    el.innerHTML = `<div class="panel">
      <div class="result-head"><p class="result-level ${r.risk_level}">${words[r.risk_level]}</p></div>
      <p class="result-sub">${explain[r.risk_level]}</p>
      <div class="dial" role="img" aria-label="Risk score ${Math.round(r.probability * 100)} out of 100, ${words[r.risk_level]}">
        <div class="dial-track"><div class="z-low" style="width:${lowEnd}%"></div>
          <div class="z-moderate" style="width:${modEnd - lowEnd}%"></div><div class="z-high" style="flex:1"></div></div>
        <div class="dial-marker" style="left:0%"><span>Your score: ${Math.round(r.probability * 100)}</span></div>
        <div class="dial-labels"><span>0</span><span>100</span></div>
      </div>
      ${factors}
      <p class="disclaimer">${r.disclaimer} The score compares your answers with a large health survey;
        it is not the chance that you have diabetes.</p>
      <div class="actions"><button type="button" class="btn btn-ghost" id="edit">Change my answers</button>
        <button type="button" class="btn btn-primary" id="restart">Start again</button></div>
      <p class="meta">Model ${r.model_version} | reference ${r.request_id}</p></div>`;
    $("wizard").hidden = true; $("intro").hidden = true; el.hidden = false; el.focus();
    requestAnimationFrame(() => { el.querySelector(".dial-marker").style.left = pos + "%"; });
    $("edit").onclick = () => { el.hidden = true; $("wizard").hidden = false; $("intro").hidden = false; step = 0; renderStep(); };
    $("restart").onclick = () => { Object.keys(answers).forEach((k) => delete answers[k]); $("edit").onclick(); };
  }

  $("back").addEventListener("click", () => { if (step > 0) { step--; renderStep(); } });
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!stepComplete()) return;
    if (step < STEPS.length - 1) { step++; renderStep(); window.scrollTo({ top: 0, behavior: "smooth" }); }
    else submit();
  });
  renderStep();
})();
