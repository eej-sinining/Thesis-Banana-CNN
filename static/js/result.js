const mobileMenuButton = document.getElementById("mobile-menu-button");
const mobileMenu = document.getElementById("mobile-menu");

// Mobile menu toggle
if (mobileMenuButton && mobileMenu) {
  mobileMenuButton.addEventListener("click", () => {
    mobileMenu.classList.toggle("hidden");
  });
}

function toPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderMetricTableRows(perClassMetrics) {
  return perClassMetrics
    .map(
      (metric) => `
      <tr class="border-b border-gray-700">
        <td class="px-3 py-2 text-gray-300">${metric.class_name}</td>
        <td class="px-3 py-2 text-gray-400">${toPercent(metric.precision)}</td>
        <td class="px-3 py-2 text-gray-400">${toPercent(metric.recall)}</td>
        <td class="px-3 py-2 text-gray-400">${toPercent(metric.f1_score)}</td>
        <td class="px-3 py-2 text-gray-400">${toPercent(metric.specificity)}</td>
        <td class="px-3 py-2 text-gray-400">${metric.support}</td>
      </tr>
    `,
    )
    .join("");
}

function renderConfusionMatrix(classOrder, matrix, toneClass) {
  const headerCells = classOrder
    .map((label) => `<th class="px-3 py-2 text-xs font-semibold text-gray-400">${label}</th>`)
    .join("");
  const rows = matrix
    .map((row, idx) => {
      const values = row
        .map((value) => `<td class="px-3 py-2 text-center text-sm text-gray-200">${value}</td>`)
        .join("");
      return `
        <tr class="border-b border-gray-700">
          <th class="px-3 py-2 text-xs font-semibold ${toneClass}">${classOrder[idx]}</th>
          ${values}
        </tr>
      `;
    })
    .join("");

  return `
    <div class="overflow-x-auto rounded-lg border border-gray-700">
      <table class="min-w-full text-left">
        <thead class="bg-gray-900">
          <tr>
            <th class="px-3 py-2 text-xs font-semibold text-gray-500">True \\ Pred</th>
            ${headerCells}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function buildModelReportCard(modelKey, report) {
  const toneClass = modelKey === "enhanced" ? "text-green-400" : "text-gray-300";
  const metrics = report.metrics;
  const annotationPath = `/static/${report.annotation.relative_path}`;

  return `
    <div class="bg-gray-800 bg-opacity-60 border ${
      modelKey === "enhanced" ? "border-green-500" : "border-gray-700"
    } rounded-2xl p-6">
      <div class="mb-4">
        <h3 class="text-lg font-semibold ${toneClass}">${report.model.name}</h3>
        <p class="text-xs text-gray-500">Version ${report.model.version} • ${report.model.type}</p>
      </div>

      <div class="grid grid-cols-2 gap-3 mb-5">
        <div class="bg-gray-900 rounded-lg p-3">
          <p class="text-xs text-gray-500 mb-1">Precision (Weighted)</p>
          <p class="text-base font-semibold ${toneClass}">${toPercent(metrics.precision_weighted)}</p>
        </div>
        <div class="bg-gray-900 rounded-lg p-3">
          <p class="text-xs text-gray-500 mb-1">Recall (Weighted)</p>
          <p class="text-base font-semibold ${toneClass}">${toPercent(metrics.recall_weighted)}</p>
        </div>
        <div class="bg-gray-900 rounded-lg p-3">
          <p class="text-xs text-gray-500 mb-1">F1-Score (Weighted)</p>
          <p class="text-base font-semibold ${toneClass}">${toPercent(metrics.f1_score_weighted)}</p>
        </div>
        <div class="bg-gray-900 rounded-lg p-3">
          <p class="text-xs text-gray-500 mb-1">Accuracy</p>
          <p class="text-base font-semibold ${toneClass}">${toPercent(metrics.accuracy)}</p>
        </div>
      </div>

      <div class="mb-5">
        <h4 class="text-sm font-semibold text-white mb-2">Per-Class Metrics (includes specificity)</h4>
        <div class="overflow-x-auto rounded-lg border border-gray-700">
          <table class="min-w-full text-left">
            <thead class="bg-gray-900">
              <tr>
                <th class="px-3 py-2 text-xs font-semibold text-gray-500">Class</th>
                <th class="px-3 py-2 text-xs font-semibold text-gray-500">Precision</th>
                <th class="px-3 py-2 text-xs font-semibold text-gray-500">Recall</th>
                <th class="px-3 py-2 text-xs font-semibold text-gray-500">F1</th>
                <th class="px-3 py-2 text-xs font-semibold text-gray-500">Specificity</th>
                <th class="px-3 py-2 text-xs font-semibold text-gray-500">Support</th>
              </tr>
            </thead>
            <tbody>${renderMetricTableRows(metrics.per_class)}</tbody>
          </table>
        </div>
      </div>

      <div class="mb-5">
        <h4 class="text-sm font-semibold text-white mb-2">Confusion Matrix</h4>
        ${renderConfusionMatrix(metrics.class_order, metrics.confusion_matrix, toneClass)}
      </div>

      <div>
        <h4 class="text-sm font-semibold text-white mb-2">Annotation (Bounding Box)</h4>
        <img src="${annotationPath}" alt="${report.model.name} annotation" class="w-full max-h-72 object-contain rounded-lg border border-gray-700 bg-black">
      </div>
    </div>
  `;
}

function appendReportsSection(reportData) {
  const contentRoot = document.querySelector("main .max-w-5xl");
  if (!contentRoot) return;

  const section = document.createElement("section");
  section.className = "max-w-5xl mx-auto mt-10";
  section.innerHTML = `
    <div class="bg-gray-900 bg-opacity-50 border border-gray-700 rounded-2xl p-6">
      <div class="mb-6">
        <h2 class="text-2xl font-bold text-white mb-1">Additional Model Reports</h2>
        <p class="text-xs text-gray-500">Baseline and enhanced model evaluation summary with confusion matrices and annotation outputs.</p>
      </div>
      <div class="grid md:grid-cols-2 gap-6">
        ${buildModelReportCard("baseline", reportData.baseline)}
        ${buildModelReportCard("enhanced", reportData.enhanced)}
      </div>
    </div>
  `;
  contentRoot.appendChild(section);
}

function appendReportStatus(message, isError = false) {
  const contentRoot = document.querySelector("main .max-w-5xl");
  if (!contentRoot) return null;

  const wrap = document.createElement("section");
  wrap.className = "max-w-5xl mx-auto mt-10";
  wrap.innerHTML = `
    <div class="rounded-xl border ${isError ? "border-red-500 bg-red-950 bg-opacity-40" : "border-gray-700 bg-gray-900 bg-opacity-50"} px-4 py-3">
      <p class="text-sm ${isError ? "text-red-300" : "text-gray-300"}">${message}</p>
    </div>
  `;
  contentRoot.appendChild(wrap);
  return wrap;
}

async function loadAdditionalReports() {
  const analyzedImage = document.querySelector('img[alt="Analyzed banana leaf"]');
  if (!analyzedImage) return;
  const loadingNode = appendReportStatus("Loading additional reports...");

  const src = analyzedImage.getAttribute("src") || "";
  const fileName = src.split("/").pop();
  if (!fileName) {
    if (loadingNode) loadingNode.remove();
    appendReportStatus("Unable to resolve image filename for report generation.", true);
    return;
  }

  try {
    const response = await fetch(`/api/reports/${encodeURIComponent(fileName)}`);
    if (!response.ok) {
      throw new Error(`API request failed (${response.status})`);
    }

    const payload = await response.json();
    if (!payload.success) {
      throw new Error(payload.error || "Report generation failed");
    }

    if (loadingNode) loadingNode.remove();
    appendReportsSection(payload);
  } catch (error) {
    if (loadingNode) loadingNode.remove();
    appendReportStatus(
      "Report loading failed. Restart the Flask server and refresh the page, then try prediction again.",
      true,
    );
    console.error("Unable to load additional reports", error);
  }
}

// Reports are rendered server-side in templates/result.html.
