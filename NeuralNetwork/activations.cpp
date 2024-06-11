#include "activations.h"

// Linear activation function
std::vector<double> linear(const std::vector<double>& output) { return output; }
std::vector<double> linearPrime(const std::vector<double>& output, const std::vector<double>& outputGradient) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        res[i] = outputGradient[i];
    }
    return res;
}

// Sigmoid activation function
std::vector<double> sigmoid(const std::vector<double>& output) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        res[i] = 1 / (1 + exp(-output[i]));
    }
    return res;
}
std::vector<double> sigmoidPrime(const std::vector<double>& output, const std::vector<double>& outputGradient) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        double sig = 1 / (1 + exp(-output[i]));
        res[i] = sig * (1 - sig) * outputGradient[i];
    }
    return res;
}

// ReLU activation function
std::vector<double> relu(const std::vector<double>& output) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        res[i] = output[i] > 0 ? output[i] : 0;
    }
    return res;
}
std::vector<double> reluPrime(const std::vector<double>& output, const std::vector<double>& outputGradient) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        res[i] = (output[i] > 0 ? 1 : 0) * outputGradient[i];
    }
    return res;
}

// Tanh activation function
std::vector<double> tanhh(const std::vector<double>& output) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        res[i] = tanh(output[i]);
    }
    return res;
}
std::vector<double> tanhPrime(const std::vector<double>& output, const std::vector<double>& outputGradient) {
    std::vector<double> res(output.size());
    for (int i = 0; i < output.size(); i++) {
        res[i] = (1 - pow(tanh(output[i]), 2)) * outputGradient[i];
    }
    return res;
}

// Softmax activation function
std::vector<double> softmax(const std::vector<double>& output) {
    double max = *std::max_element(output.begin(), output.end());
    std::vector<double> res(output.size());
    double sum = 0;
    for (int i = 0; i < output.size(); i++) {
        res[i] = exp(output[i] - max);
        sum += res[i];
    }
    for (int i = 0; i < output.size(); i++) {
        res[i] /= sum;
    }
    return res;
}

// Softmax derivative (Jacobean matrix)
std::vector<double> softmaxPrime(const std::vector<double>& output, const std::vector<double>& outputGradient) {
    std::vector<double> s = softmax(output);
    std::vector<std::vector<double>> jacobian(output.size(), std::vector<double>(output.size()));

    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output.size(); j++) {
            if (i == j) {
                jacobian[i][j] = s[i] * (1 - s[i]);
            } else {
                jacobian[i][j] = -s[i] * s[j];
            }
        }
    }

    std::vector<double> result(output.size(), 0.0);
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output.size(); j++) {
            result[i] += jacobian[i][j] * outputGradient[j];
        }
    }

    return result;
}
