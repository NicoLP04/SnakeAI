#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <iostream>
//
// Size of each block / pixel
const int BLOCK_SIZE = 200;

// Window size
const int width = 1200;
const int height = 1200;

// Direction enum
enum Direction { UP, DOWN, LEFT, RIGHT };
const std::map<Direction, std::pair<int, int>> directionMap = {
    {Direction::UP, {0, -BLOCK_SIZE}},
    {Direction::DOWN, {0, BLOCK_SIZE}},
    {Direction::LEFT, {-BLOCK_SIZE, 0}},
    {Direction::RIGHT, {BLOCK_SIZE, 0}}};

class Snake {
   public:
    Snake(Direction _currentDirection);

    int checkCollision(sf::RectangleShape& pt);
    void move(std::vector<int> action);
    void grow();
    void draw(sf::RenderWindow& window);

    sf::RectangleShape& getHead() { return body[0]; }
    Direction getDirection() { return currentDirection; }
    int getSize() { return body.size(); }

   private:
    std::vector<sf::RectangleShape> body;
    Direction currentDirection;
};

class Food {
   public:
    Food(int x, int y);
    sf::RectangleShape& getRect() { return rect; }
    void changePosition(int x, int y);
    void draw(sf::RenderWindow& window) { window.draw(rect); }

   private:
    sf::RectangleShape rect;
};

class Game {
   public:
    Game(sf::RenderWindow& _window);

    void reset();
    std::tuple<int, bool, int> playStep(std::vector<int> action);

    Snake snake;
    Food food;

   private:
    int score;
    sf::Font font;
    sf::Text text;
    sf::RenderWindow& window;
    int frameIteration;
};
