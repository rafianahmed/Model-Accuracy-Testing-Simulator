function mse(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
        sum += Math.pow(yTrue[i] - yPred[i], 2);
    }
    return sum / yTrue.length;
}
