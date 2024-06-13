all: snakeAI.cpp model.cpp agent.cpp
	g++ snakeAI.cpp model.cpp agent.cpp -o snakeAI -g -lsfml-system -lsfml-window -lsfml-graphics
