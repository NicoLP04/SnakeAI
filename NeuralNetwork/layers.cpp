#include "layers.h"

#include "activations.h"

Dense::Dense(int inputSize, int outputSize, ActivationFunction activationFunction)
    : mInputSize(inputSize), mOutputSize(outputSize) {
    mOutput.resize(outputSize);
    mWeights.resize(inputSize * outputSize);
    mBiases.resize(outputSize);
    mActivationFunction = activationFunction;
    mFirstMoment.resize(inputSize * outputSize + outputSize);
    mSecondMoment.resize(inputSize * outputSize + outputSize);
    mIteration = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);
    // std::uniform_real_distribution<double> d(-2, 2);

    for (double& weight : mWeights) weight = d(gen);
    for (double& bias : mBiases) bias = d(gen);
}

std::vector<double> Dense::forward(const std::vector<double>& input) {
    if (input.size() != mInputSize) {
        throw std::invalid_argument("Input size must match layer input size.");
    }

    mInput = input;

    for (int i = 0; i < mOutputSize; i++) {
        mOutput[i] = mBiases[i];
        for (int j = 0; j < mInputSize; j++) {
            mOutput[i] += mWeights[i * mInputSize + j] * input[j];
        }
    }

    return activationFunctions[mActivationFunction](mOutput);
}

std::vector<double> Dense::backward(const std::vector<double>& outputGradient, double learningRate, double beta1,
                                    double beta2, double epsilon) {
    if (outputGradient.size() != mOutputSize) {
        throw std::invalid_argument("Output gradient size must match layer output size.");
    }

    // Compute activation prime
    std::vector<double> activatedGradient = activationFunctionPrimes[mActivationFunction](mOutput, outputGradient);
    std::vector<double> inputGradient(mInputSize, 0.0);

    // Compute input gradient and weight gradients
    // for (int i = 0; i < mOutputSize; ++i) {
    //     for (int j = 0; j < mInputSize; ++j) {
    //         inputGradient[j] += mWeights[i * mInputSize + j] * activatedGradient[i];
    //         mWeights[i * mInputSize + j] -= learningRate * activatedGradient[i] * mInput[j];
    //     }
    //     mBiases[i] -= learningRate * activatedGradient[i];
    // }

    // Update moments and gradients using Adam
    mIteration++;
    for (int i = 0; i < mOutputSize; ++i) {
        for (int j = 0; j < mInputSize; ++j) {
            inputGradient[j] += mWeights[i * mInputSize + j] * activatedGradient[i];

            // Update first / second moment estimate
            mFirstMoment[i * mInputSize + j] =
                beta1 * mFirstMoment[i * mInputSize + j] + (1 - beta1) * activatedGradient[i] * mInput[j];
            mSecondMoment[i * mInputSize + j] =
                beta2 * mSecondMoment[i * mInputSize + j] + (1 - beta2) * pow(activatedGradient[i] * mInput[j], 2);

            // Compute bias-corrected first / second moment estimate
            double mHat = mFirstMoment[i * mInputSize + j] / (1 - pow(beta1, mIteration));
            double vHat = mSecondMoment[i * mInputSize + j] / (1 - pow(beta2, mIteration));

            // Update weights
            mWeights[i * mInputSize + j] -= learningRate * mHat / (sqrt(vHat) + epsilon);
        }

        // Update first / second moment estimate
        mFirstMoment[mOutputSize * mInputSize + i] =
            beta1 * mFirstMoment[mOutputSize * mInputSize + i] + (1 - beta1) * activatedGradient[i];
        mSecondMoment[mOutputSize * mInputSize + i] =
            beta2 * mSecondMoment[mOutputSize * mInputSize + i] + (1 - beta2) * pow(activatedGradient[i], 2);

        // Compute bias-corrected first / second moment estimate for biases
        double mHatBias = mFirstMoment[mOutputSize * mInputSize + i] / (1 - pow(beta1, mIteration));
        double vHatBias = mSecondMoment[mOutputSize * mInputSize + i] / (1 - pow(beta2, mIteration));

        // Update biases
        mBiases[i] -= learningRate * mHatBias / (sqrt(vHatBias) + epsilon);
    }

    return inputGradient;
}
