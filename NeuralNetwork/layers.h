#pragma once

#include <cmath>
#include <random>

#include "matrix.h"

class Layer {
   public:
    Layer() = default;
    ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& outputGradient,
                            const double learningRate) = 0;

    Matrix mInput;
    Matrix mOutput;
};

class Dense : public Layer {
   public:
    Dense(int input_size, int output_size);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& outputGradient,
                    const double learningRate) override;

   private:
    Matrix mWeights;
    Matrix mBiases;
};

class Activation : public Layer {
   public:
    Activation(double (*activation)(double), double (*activationPrime)(double));
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& outputGradient,
                    const double learningRate) override;

   private:
    double (*mActivation)(double);
    double (*mActivationPrime)(double);
};

class Sigmoid : public Activation {
   public:
    Sigmoid();
    static double sigmoid(double x);
    static double sigmoidPrime(double x);
};

class Relu : public Activation {
   public:
    Relu();
    static double relu(double x);
    static double reluPrime(double x);
};

class Softmax : public Activation {
   public:
    Softmax();
    static double softmax(double x);
    static double softmaxPrime(double x);
};

class Tanh : public Activation {
   public:
    Tanh();
    static double tanh(double x);
    static double tanhPrime(double x);
};
