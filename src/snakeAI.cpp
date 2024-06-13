#include "../include/snakeAI.h"

#include <cstddef>

Snake::Snake(Direction _currentDirection) {
    currentDirection = _currentDirection;

    sf::RectangleShape head;
    head.setSize(sf::Vector2f(BLOCK_SIZE, BLOCK_SIZE));
    head.setPosition(200, 200);
    head.setFillColor(sf::Color(255, 128, 0));

    sf::RectangleShape firstBody;
    firstBody.setSize(sf::Vector2f(BLOCK_SIZE, BLOCK_SIZE));
    firstBody.setPosition(180, 200);
    firstBody.setFillColor(sf::Color(255, 128, 0));

    body.push_back(head);
    body.push_back(firstBody);
}

int Snake::checkCollision(sf::RectangleShape& pt) {
    // Check wall collision
    if (pt.getPosition().x < 0 || pt.getPosition().x >= width ||
        pt.getPosition().y < 0 || pt.getPosition().y >= height)
        return 1;

    // Check body collision
    for (size_t i = 1; i < body.size(); i++)
        if (pt.getPosition() == body[i].getPosition()) return 1;

    return 0;
}

void Snake::move(std::vector<int> action) {
    // [straight, left, right]
    std::vector<Direction> clockwise = {Direction::RIGHT, Direction::DOWN,
                                        Direction::LEFT, Direction::UP};

    int idx = 0;
    if (currentDirection == Direction::DOWN)
        idx = 1;
    else if (currentDirection == Direction::LEFT)
        idx = 2;
    else if (currentDirection == Direction::UP)
        idx = 3;

    Direction direction;
    if (action[0] == 1)
        direction = clockwise[idx];
    else if (action[1] == 1)
        direction = clockwise[(idx + 1) % 4];
    else
        direction = clockwise[(idx + 3) % 4];

    // Move the body
    for (int i = body.size() - 1; i > 0; i--)
        body[i].setPosition(body[i - 1].getPosition());

    // Move the head
    body[0].move(directionMap.at(direction).first,
                 directionMap.at(direction).second);

    currentDirection = direction;
}

void Snake::grow() {
    sf::RectangleShape rect;
    rect.setSize(sf::Vector2f(BLOCK_SIZE, BLOCK_SIZE));
    rect.setPosition(body[body.size() - 1].getPosition());
    rect.setFillColor(sf::Color(255, 128, 0));
    body.push_back(rect);
}

void Snake::draw(sf::RenderWindow& window) {
    for (auto& rect : body) window.draw(rect);
}

Food::Food(int x, int y) {
    rect.setSize(sf::Vector2f(BLOCK_SIZE, BLOCK_SIZE));
    rect.setPosition(x, y);
    rect.setFillColor(sf::Color(255, 0, 0));
}

Game::Game(sf::RenderWindow& _window)
    : snake(Direction::RIGHT), food(400, 400), window(_window) {
    // Initialize score, font and text
    score = 0;
    font.loadFromFile("fonts/arial.ttf");
    text.setFillColor(sf::Color::White);
    text.setFont(font);
    text.setCharacterSize(50);
    text.setPosition(10, 10);
    text.setString("Score: " + std::to_string(score));
    frameIteration = 0;
}

void Game::reset() {
    snake = Snake(Direction::RIGHT);
    food = Food(400, 400);
    score = 0;
    text.setString("Score: " + std::to_string(score));
    frameIteration = 0;
}

std::tuple<int, bool, int> Game::playStep(std::vector<int> action) {
    frameIteration++;

    // 2. Move the snake
    snake.move(action);

    // 3. Check if game is over
    if (snake.checkCollision(snake.getHead()) == 1 ||
        frameIteration > 100 * snake.getSize())
        return {-10, true, score};

    int reward = 0;
    // 4. Check if snake ate the food
    if (snake.getHead().getPosition() == food.getRect().getPosition()) {
        frameIteration = 0;
        reward = 10;
        text.setString("Score: " + std::to_string(++score));
        snake.grow();

        // food = Food((rand() % (width / BLOCK_SIZE)) * BLOCK_SIZE,
        //             (rand() % (height / BLOCK_SIZE)) * BLOCK_SIZE);

        do {
            food = Food((rand() % (width / BLOCK_SIZE)) * BLOCK_SIZE,
                        (rand() % (height / BLOCK_SIZE)) * BLOCK_SIZE);
        } while (snake.checkCollision(food.getRect()) == 1);
    }

    // 5. Update all the objects
    window.clear(sf::Color(32, 32, 32));
    food.draw(window);
    snake.draw(window);
    window.draw(text);
    window.display();

    // 6. Return game state and score
    return {reward, false, score};
}
