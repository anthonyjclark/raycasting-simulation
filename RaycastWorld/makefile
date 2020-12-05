INCLUDES = -I../extern/lodepng
CXXFLAGS = -W -Wall -Wextra -std=c++17

MAIN = Demo
LODE = ../extern/lodepng/lodepng.cpp

release: $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(LODE) $(MAIN).cpp -o $(MAIN)

debug: $(MAIN).cpp
	$(CXX) -g  $(CXXFLAGS) $(INCLUDES) $(LODE) $(MAIN).cpp -o $(MAIN)-Debug

clean:
	$(RM) $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN)
