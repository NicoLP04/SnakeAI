#pragma once

#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

double mse(std::vector<double> yTrue, std::vector<double> yPred);
std::vector<double> msePrime(std::vector<double> yTrue, std::vector<double> yPred);

double binaryCrossEntropy(std::vector<double> yTrue, std::vector<double> yPred);
std::vector<double> binaryCrossEntropyPrime(std::vector<double> yTrue, std::vector<double> yPred);

// enum of loss functions
enum LossFunction { MSE, BINARY_CROSS_ENTROPY };

// map
static std::map<LossFunction, double (*)(std::vector<double>, std::vector<double>)> lossFunctions = {
    {MSE, mse},
    {BINARY_CROSS_ENTROPY, binaryCrossEntropy},
};
static std::map<LossFunction, std::vector<double> (*)(std::vector<double>, std::vector<double>)> lossFunctionPrimes = {
    {MSE, msePrime},
    {BINARY_CROSS_ENTROPY, binaryCrossEntropyPrime},
};
