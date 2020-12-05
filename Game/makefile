INCLUDES = -I../extern/glad/include -I../extern/lodepng
LIBS = -lglfw
CXXFLAGS = -W -Wall -Wextra -std=c++17

MAIN = Game
GLAD = ../extern/glad/src/glad.cpp
LODE = ../extern/lodepng/lodepng.cpp

release: $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LODE) $(LIBS) $(MAIN).cpp -o $(MAIN)

debug: $(MAIN).cpp
	$(CXX) -g  $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LODE) $(LIBS) $(MAIN).cpp -o $(MAIN)-Debug

clean:
	$(RM) -r $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN)
