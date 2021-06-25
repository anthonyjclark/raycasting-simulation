INCLUDES = -I../extern/glad/include -I../extern/lodepng
CXXFLAGS = -W -Wall -Wextra -std=c++17

OS_NAME := $(shell uname)

ifeq ($(OS_NAME), Darwin)
	LIBS = -lglfw
endif
ifeq ($(OS_NAME), Linux)
	LIBS = $(shell pkg-config --static --libs glfw3)
endif


MAIN = Game
GLAD = ../extern/glad/src/glad.cpp
LODE = ../extern/lodepng/lodepng.cpp

.PHONY: release
release: $(MAIN)
$(MAIN): $(MAIN).cpp ../RaycastWorld/RaycastWorld.h DisplayArray.h
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LODE) $(LIBS) $(MAIN).cpp -o $@

.PHONY: debug
debug: $(MAIN)-Debug
$(MAIN)-Debug: $(MAIN).cpp
	$(CXX) -g  $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LODE) $(LIBS) $(MAIN).cpp -o $@

clean:
	$(RM) -r $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN)
