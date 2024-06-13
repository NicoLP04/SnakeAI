# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++20 -Wall -Wextra

# Libraries
LIBS = -lsfml-system -lsfml-window -lsfml-graphics

# Source files
SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Executable
TARGET = snakeAI

# Default rule
all: $(TARGET)

# Rule to compile all source files into the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean

