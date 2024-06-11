#pragma once

#include <random>
#include <stdexcept>
#include <vector>

#include "activations.h"

class Layer {
   public:
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate,
                                         double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) = 0;
};

class Dense : public Layer {
   public:
    Dense(int inputSize, int outputSize, ActivationFunction activationFunction);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate, double beta1 = 0.9,
                                 double beta2 = 0.999, double epsilon = 1e-8) override;

   protected:
    ActivationFunction mActivationFunction;
    std::vector<double> mWeights;
    std::vector<double> mBiases;
    std::vector<double> mInput;
    std::vector<double> mOutput;
    std::vector<double> mFirstMoment;
    std::vector<double> mSecondMoment;
    double mIteration;
    int mInputSize;
    int mOutputSize;
};
