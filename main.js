let parsedData = null;
let chartInstance = null;

document.getElementById("fileInput").addEventListener("change", handleFile);

function handleFile(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => {
      parsedData = results.data;

      const headers = results.meta.fields || [];
      if (headers.length < 2) {
        alert("Your CSV must have at least 2 columns (features + target).");
        return;
      }

      // Populate target dropdown
      const targetSelect = document.getElementById("targetSelect");
      targetSelect.innerHTML = "";
      headers.forEach(h => {
        const opt = document.createElement("option");
        opt.value = h;
        opt.textContent = h;
        targetSelect.appendChild(opt);
      });

      targetSelect.disabled = false;
      document.getElementById("runBtn").disabled = false;
    },
    error: (err) => alert("Failed to parse CSV: " + err.message)
  });
}

function runSimulation() {
  if (!parsedData || parsedData.length === 0) {
    alert("Upload a CSV first.");
    return;
  }

  const targetCol = document.getElementById("targetSelect").value;

  // Keep only rows where ALL needed values are numeric
  const headers = Object.keys(parsedData[0]);
  const featureCols = headers.filter(h => h !== targetCol);

  // Build X (2D) and y (1D)
  const X = [];
  const y = [];

  for (const row of parsedData) {
    const yVal = row[targetCol];
    if (!Number.isFinite(yVal)) continue;

    const xRow = [];
    let ok = true;
    for (const f of featureCols) {
      const v = row[f];
      if (!Number.isFinite(v)) { ok = false; break; }
      xRow.push(v);
    }
    if (!ok) continue;

    X.push(xRow);
    y.push(yVal);
  }

  if (X.length < 20) {
    alert("Not enough usable numeric rows after cleaning. Check missing/text values.");
    return;
  }

  const { X_train, X_test, y_train, y_test } = trainTestSplit2D(X, y, 0.2, true);

  const results = {};

  // Multivariate Linear Regression (OLS)
  const linModel = multivariateLinearRegression(X_train, y_train);
  const yPredLin = X_test.map(x => linModel.predict(x));
  results["Linear Regression (OLS)"] = mse(y_test, yPredLin);

  // KNN Regression
  const knnModel = knnRegressionMulti(X_train, y_train, 7);
  const yPredKnn = X_test.map(x => knnModel.predict(x));
  results["KNN (k=7)"] = mse(y_test, yPredKnn);

  displayResults(results, targetCol);
}

function displayResults(modelMSEs, targetCol) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = `<div><b>Target:</b> ${targetCol}</div><br/>`;

  let bestModel = null;
  let bestScore = Infinity;

  for (const [name, score] of Object.entries(modelMSEs)) {
    resultsDiv.innerHTML += `${name}: <b>${score.toFixed(4)}</b><br/>`;
    if (score < bestScore) {
      bestScore = score;
      bestModel = name;
    }
  }

  resultsDiv.innerHTML += `<br><b>🏆 Best Model:</b> ${bestModel} (MSE = ${bestScore.toFixed(4)})`;

  const ctx = document.getElementById("chart");
  if (chartInstance) chartInstance.destroy();

  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: Object.keys(modelMSEs),
      datasets: [{ label: "Test MSE", data: Object.values(modelMSEs) }]
    }
  });
}
