function runSimulation() {
    const fileInput = document.getElementById('fileInput');
    const reader = new FileReader();

    reader.onload = function(event) {
        const lines = event.target.result.split('\n');
        const X = [];
        const y = [];

        for (let line of lines) {
            const parts = line.split(',');
            if (parts.length === 2) {
                X.push(parseFloat(parts[0]));
                y.push(parseFloat(parts[1]));
            }
        }

        const {X_train, X_test, y_train, y_test} = trainTestSplit(X, y);

        const models = {};

        const linModel = linearRegression(X_train, y_train);
        const linPred = X_test.map(x=>linModel(x));
        models["Linear Regression"] = mse(y_test, linPred);

        const knnModel = knnRegression(X_train, y_train, 5);
        const knnPred = X_test.map(x=>knnModel(x));
        models["KNN"] = mse(y_test, knnPred);

        displayResults(models);
    };

    reader.readAsText(fileInput.files[0]);
}

function displayResults(models) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    let bestModel = null;
    let bestScore = Infinity;

    for (let model in models) {
        resultsDiv.innerHTML += `${model}: MSE = ${models[model]} <br>`;
        if (models[model] < bestScore) {
            bestScore = models[model];
            bestModel = model;
        }
    }

    resultsDiv.innerHTML += `<br><b>Best Model: ${bestModel}</b>`;

    const ctx = document.getElementById('chart');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(models),
            datasets: [{
                label: 'MSE',
                data: Object.values(models)
            }]
        }
    });
}
