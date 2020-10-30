all:
	# g++ -std=c++17 -O3 -W -Wall -Wextra -o caster raycaster_pitch-altered.cpp quickcg.cpp
	# g++ -std=c++17 -g -W -Wall -Wextra -o caster raycaster_pitch-altered.cpp quickcg.cpp
	# g++ -std=c++17 -g -W -Wall -Wextra -o caster caster_demo2.cpp
	c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup -I./extern/pybind11/include `python3-config --includes` caster2bind.cpp -o caster2bind`python3-config --extension-suffix`
