#include "neuralNetwork.h"

#include <time.h>

#include <algorithm>
#include <iomanip>
#include <iostream>

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork& NeuralNetwork::addLayer(Layer* layer) {
    mLayers.push_back(layer);
    return *this;
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    Matrix output = input;
    for (Layer* layer : mLayers) output = layer->forward(output);
    return output;
}

void NeuralNetwork::train(double (*lossFunction)(const Matrix&, const Matrix&),
                          Matrix (*lossFunctionPrime)(const Matrix&,
                                                      const Matrix&),
                          const std::vector<std::vector<double>>& inputsVec,
                          const std::vector<std::vector<double>>& targetsVec,
                          const int epochs, const double learningRate,
                          bool verbose) {
    std::vector<int> indices(inputsVec.size());
    for (int i = 0; i < inputsVec.size(); i++) indices[i] = i;

    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;
    for (int i = 0; i < inputsVec.size(); i++) {
        inputs.push_back(
            Matrix(std::vector<std::vector<double>>{inputsVec[i]}));
        targets.push_back(
            Matrix(std::vector<std::vector<double>>{targetsVec[i]}));
    }

    // get the time
    clock_t start = clock();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(),
                     std::default_random_engine());
        int correct = 0;
        double error = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int index = indices[i];
            Matrix output = predict(inputs[index]);
            Matrix target = targets[index];
            error += lossFunction(output, target);
            Matrix outputGradient = lossFunctionPrime(output, target);
            for (int j = mLayers.size() - 1; j >= 0; j--)
                outputGradient =
                    mLayers[j]->backward(outputGradient, learningRate);
            if (std::round(output.get(0, 0) * 100) ==
                targets[index].get(0, 0) * 100)
                correct++;
        }
        if (verbose) {
            int barWidth = 70;

            std::cout << "[";
            int pos = barWidth * epoch / epochs;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos)
                    std::cout << "=";
                else if (i == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            std::cout << "] " << int(epoch * 100.0 / epochs) << std::fixed
                      << " %" << " - Time: " << elapsed
                      << "s - Error: " << error << " - Accuracy: "
                      << (int)(correct * 100.0) / inputs.size() << "% \r";
            std::cout.flush();
        }
    }
    if (verbose) std::cout << std::endl;
}
