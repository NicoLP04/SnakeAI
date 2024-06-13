#pragma once

#include "layers.h"
#include "losses.h"

class NeuralNetwork {
   public:
    NeuralNetwork(LossFunction lossFunction);
    NeuralNetwork &addLayer(Layer *layer);
    std::vector<double> forward(std::vector<double> input);
    void backward(std::vector<double> outputGradient, double learningRate);

    LossFunction mLossFunction;

   private:
    std::vector<Layer *> mLayers;
};
