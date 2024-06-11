#include <SFML/Graphics/RectangleShape.hpp>
#include <deque>
#include <tuple>
#include <vector>

#include "NeuralNetwork/neuralNetwork.h"
#include "snake.h"

const int MAX_MEMORY = 100000;
const int BATCH_SIZE = 1000;
const double LEARNING_RATE = 0.001;

class Agent {
   public:
    Agent();

    std::vector<int> getState(Game& game);

    void remember(std::vector<int> state, std::vector<int> action, int reward,
                  std::vector<int> nextState, bool done);

    void trainLongMemory();
    void trainShortMemory(std::vector<int> state, std::vector<int> action,
                          int reward, std::vector<int> nextState, bool done);

    std::vector<int> getAction(std::vector<int> state);

    int mGames;

   private:
    std::deque<std::tuple<std::vector<int>, std::vector<int>, int,
                          std::vector<int>, bool>>
        mMemory;
    double mEpsilon;
    double mGamma = 0;
    NeuralNetwork mModel;
};

Agent::Agent() {
    mGames = 0;
    mEpsilon = 0;  // control randomness
    mGamma = 0;    // discount rate
    mMemory = {};
    // TODO: Model, trainer
}

std::vector<int> Agent::getState(Game& game) {
    sf::RectangleShape& head = game.snake.getHead();
    std::vector<int> state;

    // get direction
    int dirR = game.snake.getDirection() == Direction::RIGHT;
    int dirL = game.snake.getDirection() == Direction::LEFT;
    int dirU = game.snake.getDirection() == Direction::UP;
    int dirD = game.snake.getDirection() == Direction::DOWN;

    // get position around the head
    sf::RectangleShape ptR;
    ptR.setPosition(head.getPosition().x + BLOCK_SIZE, head.getPosition().y);
    sf::RectangleShape ptL;
    ptL.setPosition(head.getPosition().x - BLOCK_SIZE, head.getPosition().y);
    sf::RectangleShape ptU;
    ptU.setPosition(head.getPosition().x, head.getPosition().y - BLOCK_SIZE);
    sf::RectangleShape ptD;
    ptD.setPosition(head.getPosition().x, head.getPosition().y + BLOCK_SIZE);

    // danger straight
    state.push_back((dirR && game.snake.checkCollision(ptR)) ||
                    (dirL && game.snake.checkCollision(ptL)) ||
                    (dirU && game.snake.checkCollision(ptU)) ||
                    (dirD && game.snake.checkCollision(ptD)));
    // danger right
    state.push_back((dirU && game.snake.checkCollision(ptR)) ||
                    (dirD && game.snake.checkCollision(ptL)) ||
                    (dirL && game.snake.checkCollision(ptU)) ||
                    (dirR && game.snake.checkCollision(ptD)));
    // danger left
    state.push_back((dirD && game.snake.checkCollision(ptR)) ||
                    (dirU && game.snake.checkCollision(ptL)) ||
                    (dirR && game.snake.checkCollision(ptU)) ||
                    (dirL && game.snake.checkCollision(ptD)));
    // move direction
    state.push_back(dirL);
    state.push_back(dirR);
    state.push_back(dirU);
    state.push_back(dirD);
    // food location
    state.push_back(game.snake.getHead().getPosition().x <
                    game.food.getRect().getPosition().x);  // food left
    state.push_back(game.snake.getHead().getPosition().x >
                    game.food.getRect().getPosition().x);  // food right
    state.push_back(game.snake.getHead().getPosition().y <
                    game.food.getRect().getPosition().y);  // food up
    state.push_back(game.snake.getHead().getPosition().y >
                    game.food.getRect().getPosition().y);  // food down

    return state;
}

void Agent::remember(std::vector<int> state, std::vector<int> action,
                     int reward, std::vector<int> nextState, bool done) {
    mMemory.push_back({state, action, reward, nextState, done});
    if (mMemory.size() > MAX_MEMORY) mMemory.pop_front();
}

void Agent::trainLongMemory() {
    if (mMemory.size() > BATCH_SIZE) {
        std::vector<std::vector<int>> miniBatch;
        int memoryLength = mMemory.size();
        for (int i = 0; i < BATCH_SIZE; i++) {
            int randomIndex = rand() % memoryLength;
            miniBatch.push_back(std::get<0>(mMemory[randomIndex]));
        }
    }
}

void Agent::trainShortMemory(std::vector<int> state, std::vector<int> action,
                             int reward, std::vector<int> nextState,
                             bool done) {
    mModel.train(state, action, reward, nextState, done);
}

std::vector<int> Agent::getAction(std::vector<int> state) {
    mEpsilon = 80 - mGames;
    std::vector<int> action = {0, 0, 0};

    if (rand() % 200 < mEpsilon) {
        int randomIndex = rand() % 3;
        action[randomIndex] = 1;
    } else {
        std::vector<double> state0(state.size());
        for (int i = 0; i < state.size(); i++) {
            state0[i] = state[i];
        }
        std::vector<double> predicted = mModel.predict(state0);
        int maxIndex =
            std::distance(predicted.begin(),
                          std::max_element(predicted.begin(), predicted.end()));
        action[maxIndex] = 1;
    }

    return action;
}

void train() {
    int record = 0;
    int totalScore = 0;

    Agent agent{};
    Game game{};

    while (true) {
        // Get old state
        std::vector<int> stateOld = agent.getState(game);

        // Get move
        std::vector<int> action = agent.getAction(stateOld);

        // Perform move and get new state
        std::tuple<int, int, int> gameData = game.playStep(action);
        int reward = std::get<0>(gameData);
        int done = std::get<1>(gameData);
        int score = std::get<2>(gameData);
        std::vector<int> stateNew = agent.getState(game);

        // Train short memory
        agent.trainShortMemory(stateOld, action, reward, stateNew, done);

        // Remember
        agent.remember(stateOld, action, reward, agent.getState(game), done);

        // Train long memory
        if (done == 1) {
            game.reset();
            agent.mGames += 1;
            agent.trainLongMemory();

            if (score > record) {
                record = score;
                // agent.model.save();
            }

            std::cout << "Game: " << agent.mGames << " Score: " << score
                      << " Record: " << record << std::endl;
        }

        // Print score
        if (std::get<0>(gameData) > record) {
            record = std::get<0>(gameData);
        }
    }
}

int main() {
    srand(time(NULL));
    train();
    return 0;
}
