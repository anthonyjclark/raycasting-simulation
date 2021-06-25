PY_INC_DIR := $(shell python3-config --includes)
PY_LIB_EXT := $(shell python3-config --extension-suffix)

INCLUDES = -I../extern/lodepng -I../extern/pybind11/include $(PY_INC_DIR)
CXXFLAGS = -O3 -W -Wall -Wextra -std=c++17 -shared

OS_NAME := $(shell uname)

ifeq ($(OS_NAME), Darwin)
	CXXFLAGS += -undefined dynamic_lookup
endif
ifeq ($(OS_NAME), Linux)
	CXXFLAGS += -fPIC
endif

MAIN = pycaster
LIBNAME = $(MAIN)$(PY_LIB_EXT)
LODE = ../extern/lodepng/lodepng.cpp

.PHONY: library
library: $(LIBNAME)
$(LIBNAME): ../RaycastWorld/RaycastWorld.h pycaster.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LODE) $(MAIN).cpp -o $@

clean:
	$(RM) $(LIBNAME)
