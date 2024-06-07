#pragma once

#include "layers.h"
#include "losses.h"
#include "matrix.h"

class NeuralNetwork {
   public:
    NeuralNetwork();
    NeuralNetwork& addLayer(Layer* layer);
    Matrix predict(const Matrix& input);
    void train(double (*lossFunction)(const Matrix&, const Matrix&),
               Matrix (*lossFunctionPrime)(const Matrix&, const Matrix&),
               const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               const int epochs, const double learningRate,
               bool verbose = true);

   private:
    std::vector<Layer*> mLayers;
};
