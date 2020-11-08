
INCLUDES = -I../extern/pybind11/include `python3-config --includes`
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -shared -undefined dynamic_lookup

SRC = pycaster
EXT = `python3-config --extension-suffix`

all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC).cpp -o $(SRC)$(EXT)
