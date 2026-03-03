function linearRegression(X, y) {
    const n = X.length;
    const meanX = X.reduce((a,b)=>a+b)/n;
    const meanY = y.reduce((a,b)=>a+b)/n;

    let num = 0;
    let den = 0;

    for (let i = 0; i < n; i++) {
        num += (X[i] - meanX) * (y[i] - meanY);
        den += (X[i] - meanX) ** 2;
    }

    const slope = num / den;
    const intercept = meanY - slope * meanX;

    return x => intercept + slope * x;
}
