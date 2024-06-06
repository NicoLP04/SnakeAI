all: snakeAI.cpp
	g++ snakeAI.cpp -o snakeAI -lsfml-system -lsfml-window -lsfml-graphics
