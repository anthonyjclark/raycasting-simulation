INCLUDES = -I../extern/lodepng
CXXFLAGS = -W -Wall -Wextra -std=c++17

MAIN = Demo
LODE = ../extern/lodepng/lodepng.cpp

.PHONY: release
release: $(MAIN)
$(MAIN): $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(INCLUDES) $(LODE) $(MAIN).cpp -o $@

.PHONY: debug
debug: $(MAIN)-Debug
$(MAIN)-Debug: $(MAIN).cpp
	$(CXX) -g  $(CXXFLAGS) $(INCLUDES) $(LODE) $(MAIN).cpp -o $@

clean:
	$(RM) -r $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN) frame*.png
