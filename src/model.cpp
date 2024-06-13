#include "../include/model.h"

Model::Model(double learningRate, double gamma)
    : mNN(NeuralNetwork(MSE)), learningRate(learningRate), gamma(gamma) {
    // mNN.addLayer(new Dense(11, 64, RELU))
    //     .addLayer(new Dense(64, 64, RELU))
    //     .addLayer(new Dense(64, 3, SOFTMAX));
    mNN.addLayer(new Dense(11, 256, RELU)).addLayer(new Dense(256, 3, SOFTMAX));
}

void Model::trainStep(std::vector<int> state, std::vector<int> nextState,
                      std::vector<int> action, int reward, bool done) {
    std::vector<double> stateD(state.begin(), state.end());
    std::vector<double> nextStateD(nextState.begin(), nextState.end());
    std::vector<double> actionD(action.begin(), action.end());

    // Predict the Q-values for the current state
    std::vector<double> prediction = predict(stateD);
    std::vector<double> target = prediction;

    double QNew = reward;

    if (!done) {
        std::vector<double> nextTarget = predict(nextStateD);
        double maxNextQ =
            *std::max_element(nextTarget.begin(), nextTarget.end());

        QNew = reward + gamma * maxNextQ;
    }

    int idx = 0;
    double max = actionD[0];
    for (size_t i = 1; i < actionD.size(); i++) {
        if (actionD[i] > max) {
            max = actionD[i];
            idx = i;
        }
    }

    target[idx] = QNew;

    // double instanceError = lossFunctions[mNN.mLossFunction](target,
    // prediction);

    std::vector<double> gradient =
        lossFunctionPrimes[mNN.mLossFunction](target, prediction);

    mNN.backward(gradient, learningRate);
}

std::vector<double> Model::predict(std::vector<double> state) {
    return mNN.forward(state);
}
