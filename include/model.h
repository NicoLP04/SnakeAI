#pragma once

#include <iostream>

#include "neuralNetwork.h"

class Model {
   public:
    Model(double learningRate, double gamma);
    std::vector<double> predict(std::vector<double> state);
    void trainStep(std::vector<int> state, std::vector<int> nextState,
                   std::vector<int> action, int reward, bool done);

   private:
    NeuralNetwork mNN;
    double learningRate;
    double gamma;
};
