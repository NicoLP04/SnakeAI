#include "losses.h"

double mse(const Matrix& predictions, const Matrix& targets) {
    Matrix diff = predictions - targets;
    return (diff * diff).sum() / predictions.getHeight();
}

Matrix msePrime(const Matrix& predictions, const Matrix& targets) {
    return ((predictions - targets) * 2) * (1.0 / predictions.getHeight());
}

double crossEntropy(const Matrix& predictions, const Matrix& targets) {
    Matrix logPredictions = predictions.applyFunction(log);
    Matrix diff = logPredictions * targets;
    return -diff.sum() / predictions.getHeight();
}

Matrix crossEntropyPrime(const Matrix& predictions, const Matrix& targets) {
    return (predictions - targets) * (1.0 / predictions.getHeight());
}
