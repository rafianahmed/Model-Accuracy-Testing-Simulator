function trainTestSplit2D(X, y, testRatio = 0.2, shuffle = true) {
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);

  if (shuffle) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
  }

  const testSize = Math.floor(n * testRatio);
  const testIdx = idx.slice(0, testSize);
  const trainIdx = idx.slice(testSize);

  const X_train = trainIdx.map(i => X[i]);
  const y_train = trainIdx.map(i => y[i]);
  const X_test = testIdx.map(i => X[i]);
  const y_test = testIdx.map(i => y[i]);

  return { X_train, X_test, y_train, y_test };
}

// OLS: beta = (X^T X)^-1 X^T y, with intercept
function multivariateLinearRegression(X, y) {
  // Add intercept column of 1s
  const Xb = X.map(row => [1, ...row]);

  const Xmat = math.matrix(Xb);
  const yvec = math.matrix(y);

  const Xt = math.transpose(Xmat);
  const XtX = math.multiply(Xt, Xmat);

  // Small ridge term for numerical stability
  const lambda = 1e-8;
  const I = math.identity(Xb[0].length);
  const XtX_reg = math.add(XtX, math.multiply(lambda, I));

  const XtY = math.multiply(Xt, yvec);
  const beta = math.multiply(math.inv(XtX_reg), XtY); // (p+1) x 1

  return {
    predict: (x) => {
      const xb = [1, ...x];
      return math.dot(xb, beta);
    }
  };
}

// Multi-dimensional KNN regression
function knnRegressionMulti(X, y, k = 5) {
  function dist(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return Math.sqrt(s);
  }

  return {
    predict: (x) => {
      const arr = X.map((xi, i) => ({ d: dist(xi, x), v: y[i] }));
      arr.sort((p, q) => p.d - q.d);
      const nn = arr.slice(0, k);
      return nn.reduce((sum, o) => sum + o.v, 0) / k;
    }
  };
}
