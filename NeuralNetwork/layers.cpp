#include "layers.h"

// used to init random mWeights and biases
double random(double x) {
    // Create a random number generator engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a uniform distribution between -0.5 and 0.5
    std::uniform_real_distribution<double> dis(0, 1);

    // Generate and return a random value
    return dis(gen);
}

Dense::Dense(int input_size, int output_size) {
    mWeights = Matrix(output_size, input_size);
    mBiases = Matrix(output_size, 1);

    // assign random values
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            mWeights.set(i, j, random(0));
        }
        mBiases.set(i, 0, random(0));
    }
}

Matrix Dense::forward(const Matrix& input) {
    mInput = input.transpose();
    return mWeights * mInput + mBiases;
}

Matrix Dense::backward(const Matrix& outputGradient,
                       const double learningRate) {
    Matrix weightsGradient = outputGradient * mInput.transpose();
    Matrix inputGradient = mWeights.transpose() * outputGradient;

    mWeights = mWeights - weightsGradient * learningRate;
    mBiases = mBiases - outputGradient * learningRate;

    return inputGradient;
}

Activation::Activation(double (*activation)(double),
                       double (*activationPrime)(double)) {
    mActivation = activation;
    mActivationPrime = activationPrime;
}

Matrix Activation::forward(const Matrix& input) {
    mInput = input.transpose();
    return mInput.applyFunction(mActivation);
}

Matrix Activation::backward(const Matrix& outputGradient,
                            const double learningRate) {
    return outputGradient.multiplyElementWise(
        mInput.transpose().applyFunction(mActivationPrime));
}

Sigmoid::Sigmoid() : Activation(sigmoid, sigmoidPrime) {}
double Sigmoid::sigmoid(double x) { return 1 / (1 + exp(-x)); }
double Sigmoid::sigmoidPrime(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

Relu::Relu() : Activation(relu, reluPrime) {}
double Relu::relu(double x) { return x > 0 ? x : 0; }
double Relu::reluPrime(double x) { return x > 0 ? 1 : 0; }

Softmax::Softmax() : Activation(softmax, softmaxPrime) {}
double Softmax::softmax(double x) { return exp(x); }
double Softmax::softmaxPrime(double x) { return softmax(x) * (1 - softmax(x)); }

Tanh::Tanh() : Activation(tanh, tanhPrime) {}
double Tanh::tanh(double x) { return std::tanh(x); }
double Tanh::tanhPrime(double x) { return 1 - std::pow(tanh(x), 2); }
