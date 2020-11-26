INCLUDES = -I../extern/pybind11/include `python3-config --includes`
CXXFLAGS = -O3 -W -Wall -Wextra -std=c++17 -shared -undefined dynamic_lookup

MAIN = pycaster
EXT = `python3-config --extension-suffix`

library:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(MAIN).cpp -o $(MAIN)$(EXT)

clean:
	$(RM) $(MAIN)$(EXT)
