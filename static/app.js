/* ============================================================
   Phishing URL Detector — Frontend Logic
   ============================================================ */

const MAX_BATCH_URLS = 1000;
const URL_PATTERN = /(?:https?:\/\/|www\.)[^\s"'<>;,]+|(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?::\d+)?(?:\/[^\s"'<>;,]*)?|\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:\/[^\s"'<>;,]*)?/gi;

function extractUrls(text) {
  return (text.match(URL_PATTERN) || []).map(url => url.replace(/[),.;!?}\]]+$/, ""));
}

async function importUrlFiles(fileInput) {
  const files = Array.from(fileInput.files);
  const input = document.getElementById("urlInput");
  const status = document.getElementById("fileStatus");
  const supportedFiles = files.filter(file => /\.(txt|csv|json|log|md)$/i.test(file.name));
  fileInput.value = "";

  if (!supportedFiles.length) {
    status.textContent = "Choose text, CSV, or JSON files containing URLs.";
    return;
  }

  try {
    const importedUrls = [];
    for (const file of supportedFiles) {
      importedUrls.push(...extractUrls(await file.text()));
      if (importedUrls.length + input.value.split(/\r?\n/).filter(line => line.trim()).length > MAX_BATCH_URLS) {
        status.textContent = `The import exceeds the ${MAX_BATCH_URLS.toLocaleString()} URL limit. Nothing was added.`;
        return;
      }
    }

    if (!importedUrls.length) {
      status.textContent = "No URLs found in the selected files.";
      return;
    }

    const existingUrls = input.value.split(/\r?\n/).map(url => url.trim()).filter(Boolean);
    input.value = [...existingUrls, ...importedUrls].join("\n");
    status.textContent = `Added ${importedUrls.length.toLocaleString()} URLs from ${supportedFiles.length} file${supportedFiles.length === 1 ? "" : "s"}.`;
  } catch (error) {
    status.textContent = `Could not read the selected files: ${error.message}`;
  }
}

function classifyRisk(pct) {
  if (pct < 35) return "SAFE";
  if (pct < 70) return "SUSPICIOUS";
  return "PHISHING";
}

function verdictIcon(status) {
  return { SAFE: "✅", SUSPICIOUS: "⚠️", PHISHING: "🚨" }[status] ?? "❓";
}

function setBar(barId, pct) {
  const el = document.getElementById(barId);
  // Use setTimeout so CSS transition triggers after element is visible
  setTimeout(() => { el.style.width = Math.min(pct, 100) + "%"; }, 50);
}

function populateReasons(listId, reasons, status) {
  const ul = document.getElementById(listId);
  ul.innerHTML = "";
  reasons.forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    li.className = "model-reason-item reason-" + status;
    ul.appendChild(li);
  });
}

async function analyzeUrl() {
  const input  = document.getElementById("urlInput");
  const btn    = document.getElementById("analyzeBtn");
  const btnTxt = document.getElementById("btnText");
  const spin   = document.getElementById("btnSpinner");
  const panel  = document.getElementById("resultPanel");
  const status = document.getElementById("analysisStatus");
  const emptyResults = document.getElementById("emptyResults");

  const urls = input.value.split(/\r?\n/).map(url => url.trim()).filter(Boolean);
  if (!urls.length) {
    status.textContent = "Enter at least one URL or domain to check.";
    input.focus();
    return;
  }
  if (urls.length > MAX_BATCH_URLS) {
    status.textContent = `Analyze up to ${MAX_BATCH_URLS.toLocaleString()} URLs per request.`;
    input.focus();
    return;
  }

  status.textContent = "";
  btn.disabled = true;
  btnTxt.textContent = "Checking links...";
  spin.classList.remove("hidden");
  panel.classList.add("hidden");
  document.getElementById("batchPanel").classList.add("hidden");
  emptyResults.classList.add("hidden");

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(urls.length === 1 ? { url: urls[0] } : { urls })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) throw new Error(data.error || "The analysis service is unavailable. Please try again.");
    if (data.results) renderBatchResults(data);
    else renderResult(data);
  } catch (err) {
    status.textContent = err.message || "The analysis could not be completed. Please try again.";
    emptyResults.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btnTxt.textContent = "Check link";
    spin.classList.add("hidden");
  }
}

function renderBatchResults(data) {
  const panel = document.getElementById("batchPanel");
  const tbody = document.getElementById("batchResults");
  tbody.replaceChildren();
  document.getElementById("batchHeading").textContent = `Results for ${data.count.toLocaleString()} URLs`;
  const rows = document.createDocumentFragment();

  data.results.forEach(result => {
    const row = document.createElement("tr");
    [result.url, result.status, `${result.risk_score}%`,
      result.lr_prob === null ? "Blacklist" : `${result.lr_status} · ${result.lr_prob}%`,
      result.rf_prob === null ? "Blacklist" : `${result.rf_status} · ${result.rf_prob}%`
    ].forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 1) cell.className = "table-verdict verdict-text-" + result.status;
      row.appendChild(cell);
    });

    const reasonCell = document.createElement("td");
    const details = document.createElement("details");
    details.className = "batch-reasons";
    const summary = document.createElement("summary");
    summary.textContent = "View reasons";
    details.appendChild(summary);

    [
      ["Detection insights", result.reasons],
      ["Logistic Regression", result.lr_reasons],
      ["Random Forest", result.rf_reasons]
    ].forEach(([headingText, reasons]) => {
      if (!reasons || !reasons.length) return;
      const heading = document.createElement("h4");
      heading.textContent = headingText;
      details.appendChild(heading);
      const list = document.createElement("ul");
      reasons.forEach(reason => {
        const item = document.createElement("li");
        item.textContent = reason;
        list.appendChild(item);
      });
      details.appendChild(list);
    });

    reasonCell.appendChild(details);
    row.appendChild(reasonCell);
    rows.appendChild(row);
  });
  tbody.appendChild(rows);

  document.getElementById("emptyResults").classList.add("hidden");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadModelMetrics() {
  const body = document.getElementById("metricsBody");
  const method = document.getElementById("metricsMethod");
  const note = document.getElementById("metricsNote");
  try {
    const response = await fetch("/metrics");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to load model metrics.");
    Object.entries(data.models).forEach(([name, metrics]) => {
      const row = document.createElement("tr");
      [name, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score]
        .forEach((value, index) => {
          const cell = document.createElement("td");
          cell.textContent = index === 0 ? value : `${value}%`;
          row.appendChild(cell);
        });
      body.appendChild(row);
    });
    method.textContent = data.evaluation;
    note.textContent = `Scores use phishing as the positive class and are based on ${data.sample_count} synthetic training URLs. Treat this small demo dataset as illustrative, not a measure of production performance.`;
  } catch (error) {
    method.textContent = "Evaluation unavailable";
    note.textContent = error.message;
  }
}

function renderResult(data) {
  const panel = document.getElementById("resultPanel");

  /* --- Overall verdict --- */
  const badge = document.getElementById("verdictBadge");
  badge.className = "verdict-badge verdict-" + data.status;
  document.getElementById("verdictIcon").textContent = verdictIcon(data.status);
  document.getElementById("verdictText").textContent = data.status;

  /* --- Combined score --- */
  document.getElementById("combinedPct").textContent = data.risk_score + "%";
  setBar("combinedBar", data.risk_score);

  /* --- Blacklist path: no per-model data --- */
  const modelsGrid = document.querySelector(".models-grid");
  if (data.lr_prob == null) {
    modelsGrid.style.display = "none";
  } else {
    modelsGrid.style.display = "grid";

    /* LR model */
    const lrStatus = classifyRisk(data.lr_prob);
    document.getElementById("lrPct").textContent          = data.lr_prob + "%";
    document.getElementById("lrVerdict").textContent      = lrStatus;
    document.getElementById("lrVerdict").className        = "model-verdict-badge mvb-" + lrStatus;
    setBar("lrBar", data.lr_prob);
    populateReasons("lrReasons", data.lr_reasons || [], lrStatus);

    /* RF model */
    const rfStatus = classifyRisk(data.rf_prob);
    document.getElementById("rfPct").textContent          = data.rf_prob + "%";
    document.getElementById("rfVerdict").textContent      = rfStatus;
    document.getElementById("rfVerdict").className        = "model-verdict-badge mvb-" + rfStatus;
    setBar("rfBar", data.rf_prob);
    populateReasons("rfReasons", data.rf_reasons || [], rfStatus);
  }

  /* --- Reasons --- */
  const list = document.getElementById("reasonsList");
  list.innerHTML = "";
  (data.reasons || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    list.appendChild(li);
  });

  document.getElementById("emptyResults").classList.add("hidden");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* Allow pressing Enter in the input */
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("urlInput").addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) analyzeUrl();
  });
  document.getElementById("chooseFilesBtn").addEventListener("click", () => document.getElementById("urlFiles").click());
  document.getElementById("chooseFolderBtn").addEventListener("click", () => document.getElementById("urlFolderFiles").click());
  document.getElementById("urlFiles").addEventListener("change", event => importUrlFiles(event.currentTarget));
  document.getElementById("urlFolderFiles").addEventListener("change", event => importUrlFiles(event.currentTarget));
  loadModelMetrics();
});
