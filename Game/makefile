INCLUDES = -I../extern/glad/include -I../extern/lodepng
LIBS = -lglfw
CXXFLAGS = -W -Wall -Wextra -std=c++17

MAIN = Game
GLAD = ../extern/glad/src/glad.cpp
LODE = ../extern/lodepng/lodepng.cpp

.PHONY: release
release: $(MAIN)
$(MAIN): $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LODE) $(LIBS) $(MAIN).cpp -o $@

.PHONY: debug
debug: $(MAIN)-Debug
$(MAIN)-Debug: $(MAIN).cpp
	$(CXX) -g  $(CXXFLAGS) $(INCLUDES) $(GLAD) $(LODE) $(LIBS) $(MAIN).cpp -o $@

clean:
	$(RM) -r $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN)
