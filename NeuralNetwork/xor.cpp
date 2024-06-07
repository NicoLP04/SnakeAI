#include <iostream>

#include "matrix.h"
#include "neuralNetwork.h"

int main() {
    NeuralNetwork nn;
    nn.addLayer(new Dense(2, 3))
        .addLayer(new Relu())
        .addLayer(new Dense(3, 1))
        .addLayer(new Sigmoid());

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    nn.train(mse, msePrime, inputs, targets, 10000, 0.1);
    for (const std::vector<double>& input : inputs) {
        Matrix inputMatrix = Matrix{std::vector<std::vector<double>>{input}};
        Matrix output = nn.predict(inputMatrix);
        std::cout << inputMatrix << " -> " << output << std::endl;
    }
    return 0;
}
