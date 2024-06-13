#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <iostream>

// Size of each block / pixel
const int BLOCK_SIZE = 20;

// Window size
const int width = 1200;
const int height = 800;

// Speed of the snake (can't be 0)
const int speed = 25;

// Direction enum
enum Direction { UP, DOWN, LEFT, RIGHT };
const std::map<Direction, std::pair<int, int>> directionMap = {
    {Direction::UP, {0, -BLOCK_SIZE}},
    {Direction::DOWN, {0, BLOCK_SIZE}},
    {Direction::LEFT, {-BLOCK_SIZE, 0}},
    {Direction::RIGHT, {BLOCK_SIZE, 0}}};

class Snake {
   public:
    Snake(Direction _currentDirection) {
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

    int checkCollision() {
        // Check wall collision
        if (body[0].getPosition().x < 0 || body[0].getPosition().x >= width ||
            body[0].getPosition().y < 0 || body[0].getPosition().y >= height)
            return 1;

        // Check body collision
        for (int i = 1; i < body.size(); i++)
            if (body[0].getPosition() == body[i].getPosition()) return 1;

        return 0;
    }

    void move(Direction direction) {
        // Move the body
        for (int i = body.size() - 1; i > 0; i--)
            body[i].setPosition(body[i - 1].getPosition());

        // get correct direction
        if (direction == Direction::UP && currentDirection == Direction::DOWN ||
            direction == Direction::DOWN && currentDirection == Direction::UP ||
            direction == Direction::LEFT &&
                currentDirection == Direction::RIGHT ||
            direction == Direction::RIGHT &&
                currentDirection == Direction::LEFT)
            direction = currentDirection;

        // Move the head
        body[0].move(directionMap.at(direction).first,
                     directionMap.at(direction).second);

        currentDirection = direction;
    }

    void grow() {
        sf::RectangleShape rect;
        rect.setSize(sf::Vector2f(BLOCK_SIZE, BLOCK_SIZE));
        rect.setPosition(body[body.size() - 1].getPosition());
        rect.setFillColor(sf::Color(255, 128, 0));
        body.push_back(rect);
    }

    void draw(sf::RenderWindow& window) {
        for (auto& rect : body) window.draw(rect);
    }

    sf::RectangleShape& getHead() { return body[0]; }

    Direction getDirection() { return currentDirection; }

   private:
    std::vector<sf::RectangleShape> body;
    Direction currentDirection;
};

class Food {
   public:
    Food(int x, int y) {
        rect.setSize(sf::Vector2f(BLOCK_SIZE, BLOCK_SIZE));
        rect.setPosition(x, y);
        rect.setFillColor(sf::Color(255, 0, 0));
    }

    sf::RectangleShape& getRect() { return rect; }

    void draw(sf::RenderWindow& window) { window.draw(rect); }

   private:
    sf::RectangleShape rect;
};

class Game {
   public:
    Game(sf::RenderWindow& _window)
        : snake(Direction::RIGHT), food(400, 400), window(_window) {
        // Initialize score, font and text
        score = 0;
        font.loadFromFile("arial.ttf");
        text.setFillColor(sf::Color::White);
        text.setFont(font);
        text.setCharacterSize(24);
        text.setPosition(10, 10);
        text.setString("Score: " + std::to_string(score));
    }

    std::tuple<int, int> playStep() {
        // 1. Get the direction
        Direction direction = snake.getDirection();
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
            direction = Direction::UP;
        } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
            direction = Direction::DOWN;
        } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
            direction = Direction::LEFT;
        } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            direction = Direction::RIGHT;
        }

        // 2. Move the snake
        snake.move(direction);

        // 3. Check if game is over
        if (snake.checkCollision() == 1) return {1, score};

        // 4. Check if snake ate the food
        if (snake.getHead().getPosition() == food.getRect().getPosition()) {
            text.setString("Score: " + std::to_string(++score));
            snake.grow();
            food = Food((rand() % (width / BLOCK_SIZE)) * BLOCK_SIZE,
                        (rand() % (height / BLOCK_SIZE)) * BLOCK_SIZE);
        }

        // 5. Update all the objects
        window.clear(sf::Color(32, 32, 32));
        food.draw(window);
        snake.draw(window);
        window.draw(text);
        window.display();

        // 6. Return game state and score
        return {0, score};
    }

   private:
    Snake snake;
    Food food;
    int score;
    sf::Font font;
    sf::Text text;
    sf::RenderWindow& window;
};

void game() {
    // Create window
    sf::RenderWindow window =
        sf::RenderWindow(sf::VideoMode(width, height), "Snake",
                         sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(120);
    sf::Clock clock;

    // Create game object
    Game game(window);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                window.close();
        }

        // Check if it's time to play a step
        if (clock.getElapsedTime().asSeconds() < 1. / speed) continue;
        clock.restart();

        // Play a step
        auto tuple = game.playStep();
        if (std::get<0>(tuple) == 1) {
            std::cout << "Game Over! Score: " << std::get<1>(tuple)
                      << std::endl;
            window.close();
        }
    }
}

int main() {
    srand(time(NULL));
    game();
    return 0;
}
