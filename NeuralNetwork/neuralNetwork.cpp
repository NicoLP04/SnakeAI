#include "neuralNetwork.h"

NeuralNetwork::NeuralNetwork(LossFunction lossFunction) { mLossFunction = lossFunction; }

NeuralNetwork &NeuralNetwork::addLayer(Layer *layer) {
    mLayers.push_back(layer);
    return *this;
}

std::vector<double> NeuralNetwork::forward(std::vector<double> input) {
    for (Layer *layer : mLayers) {
        input = layer->forward(input);
    }
    return input;
}

void NeuralNetwork::backward(std::vector<double> outputGradient, double learningRate) {
    for (int i = mLayers.size() - 1; i >= 0; i--) {
        outputGradient = mLayers[i]->backward(outputGradient, learningRate);
    }
}
