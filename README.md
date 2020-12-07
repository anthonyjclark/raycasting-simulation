# RaycastWorld

Generate mazes and navigate through them with turn-by-turn directions.

# Requirements

`RaycastWorld/RaycastWorld.cpp` does not have any dependencies not found in this repository (`lodepng` is a submodule).

`Game/Game.cpp` depends on [GLFW3](https://www.glfw.org/).

`MazeGen/MazeGen.py` depends on NumPy, but this dependency can be removed.

# Getting Started

Here is the process to get the game up and running (all steps assume you are in the root project directory):

1. Generate a maze using `MazeGen/MazeGen.py`

```bash
MazeGen/MazeGen.py --width 8 --height 8 --out > Mazes/new_maze.txt
```

2. Compile `Game/Game`

```bash
cd Game
make
```

3. Run `Game/Game` with the new maze

```bash
cd Game
./Game ../Mazes/new_maze.txt
```

# Notes

Things to do:

- add lighting
- add levels (e.g., three levels of cast rays)
- agent height
- skybox
- https://github.com/sizmailov/pybind11-stubgen
- wall heights: for each vertical line
    + cast ray
    + fill from bottom until height
    + repeat until full screen height
- we can use the raycast distances to label pixels for getting depth
- compile to JS and work in browser
- switch to Julia? (pyjulia, juliaimages)

Resources:

- [Possible Python GUI](https://old.reddit.com/r/DearPyGui/comments/jp0upr/load_image_from_numpy/)
- [Raycasting with different height walls](https://stackoverflow.com/questions/47239797/ray-casting-with-different-height-size/)
- [Raycasting tutorial](https://lodev.org/cgtutor/)
- [Raycasting course](https://courses.pikuma.com/courses/raycasting)
- [Generating mazes](https://weblog.jamisbuck.org/2011/2/7/maze-generation-algorithm-recap)
