CXXFLAGS = -W -Wall -Wextra -std=c++17

MAIN = Demo

release: $(MAIN).cpp
	$(CXX) -O3 $(CXXFLAGS) $(MAIN).cpp -o $(MAIN)

debug: $(MAIN).cpp
	$(CXX) -g  $(CXXFLAGS) $(MAIN).cpp -o $(MAIN)-Debug

clean:
	$(RM) $(MAIN)-Debug.dSYM $(MAIN)-Debug $(MAIN)
