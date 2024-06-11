#include "losses.h"

double mse(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    double sum = 0;
    for (int i = 0; i < yTrue.size(); i++) {
        sum += pow(yTrue[i] - yPred[i], 2);
    }

    return sum / yTrue.size();
}

std::vector<double> msePrime(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    std::vector<double> gradient(yTrue.size());
    for (int i = 0; i < yTrue.size(); i++) {
        gradient[i] = 2 * (yPred[i] - yTrue[i]) / yTrue.size();
    }

    return gradient;
}

double binaryCrossEntropy(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    double epsilon = 0.00001;
    double sum = 0;
    for (int i = 0; i < yTrue.size(); i++) {
        double value1 = std::max(yPred[i], epsilon);
        double value2 = std::max(1 - yPred[i], epsilon);
        sum += yTrue[i] * log(value1) + (1 - yTrue[i]) * log(value2);
    }

    return -sum / yTrue.size();
}

std::vector<double> binaryCrossEntropyPrime(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    double epsilon = 0.00001;
    std::vector<double> gradient(yTrue.size());
    for (int i = 0; i < yTrue.size(); i++) {
        double value1 = std::max(yPred[i], epsilon);
        gradient[i] = ((1 - yTrue[i]) / (1 - yPred[i]) - yTrue[i] / value1) / yTrue.size();
    }

    return gradient;
}
