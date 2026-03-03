function trainTestSplit(X, y, testRatio = 0.2) {
    const n = X.length;
    const testSize = Math.floor(n * testRatio);

    const X_train = X.slice(0, n - testSize);
    const X_test = X.slice(n - testSize);

    const y_train = y.slice(0, n - testSize);
    const y_test = y.slice(n - testSize);

    return { X_train, X_test, y_train, y_test };
}
