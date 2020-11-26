INCLUDES = -I../extern/glad/include
LIBS = -lglfw
CXXFLAGS = -W -Wall -Wextra -std=c++17

MAIN = Game
GLAD = ../extern/glad/src/glad.c

release: $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LIBS) $(MAIN).cpp -o $(MAIN)

debug: $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LIBS) $(MAIN).cpp -o $(MAIN)-Debug

clean:
	$(RM) $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN)
