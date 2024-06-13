#include "model.h"

Model::Model() : mNN(NeuralNetwork(MSE)) {
    // mNN.addLayer(new Dense(11, 120, SIGMOID))
    //     .addLayer(new Dense(120, 120, SIGMOID))
    //     .addLayer(new Dense(120, 3, SIGMOID));
    mNN.addLayer(new Dense(11, 256, RELU)).addLayer(new Dense(256, 3, SOFTMAX));
    learningRate = 0.001;
    gamma = 0.9;
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
    for (int i = 1; i < actionD.size(); i++) {
        if (actionD[i] > max) {
            max = actionD[i];
            idx = i;
        }
    }

    target[idx] = QNew;

    double instanceError = lossFunctions[mNN.mLossFunction](target, prediction);

    std::vector<double> gradient =
        lossFunctionPrimes[mNN.mLossFunction](target, prediction);

    mNN.backward(gradient, learningRate);
}

std::vector<double> Model::predict(std::vector<double> state) {
    return mNN.forward(state);
}
