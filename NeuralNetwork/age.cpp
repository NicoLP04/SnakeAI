#include <iostream>

#include "layers.h"
#include "matrix.h"
#include "neuralNetwork.h"

int main() {
    NeuralNetwork nn;
    nn.addLayer(new Dense(1, 4))
        .addLayer(new Sigmoid())
        .addLayer(new Dense(4, 1))
        .addLayer(new Relu());

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
    for (double i = 0; i < 100; i++) {
        inputs.push_back({(double)i / 100});
        targets.push_back({(double)(i) / 100});
    }

    // nn.train(crossEntropy, crossEntropyPrime, inputs, targets, 3001, 0.1);
    nn.train(mse, msePrime, inputs, targets, 5001, 0.2);
    int correct = 0;
    for (int input = 0; input < 100; input++) {
        Matrix inputMatrix =
            Matrix{std::vector<std::vector<double>>{{input / 100.}}};
        Matrix output = nn.predict(inputMatrix);
        if (std::round(output.get(0, 0) * 100) == input) correct++;
    }

    std::cout << "Accuracy: " << correct << "/100" << std::endl;

    std::cout << "Enter your age: " << std::endl;
    std::string age;
    std::cin >> age;
    double ageDouble = std::stod(age);
    Matrix output =
        nn.predict(Matrix{std::vector<std::vector<double>>{{ageDouble / 100}}});
    int prediction = (int)(output.get(0, 0) * 100);
    std::cout << "I predict your age is: " << prediction << std::endl;

    return 0;
}
