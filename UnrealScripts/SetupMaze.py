import unreal
from unreal import Vector

import sys

sys.path.append("../Utilities")
from MazeUtils import read_maze_file  # type: ignore

"""
Drag texture pngs to content browser
Right-click texture png and create material
Double-click material, add TextureCoordinate and adjust
Drag cube to content browser
Double-click cube, add created material
(This script does the rest)
"""

maze_filepath = "/home/ajc/Documents/Repositories/raycasting-simulation/Worlds/maze.txt"


def spawn_actor(location, meshpath):

    new_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, location=location
    )
    mesh = unreal.load_object(None, meshpath)
    mesh_component = new_actor.get_editor_property("static_mesh_component")  # type: ignore
    mesh_component.set_static_mesh(mesh)

    return new_actor


maze, maze_x_dim, maze_y_dim, _ = read_maze_file(maze_filepath)

CUBE_SCALE = 100
WALL_Z_OFFSET = CUBE_SCALE

MAZE_X_OFFSET = -(CUBE_SCALE * maze_x_dim) / 2 + CUBE_SCALE / 2
MAZE_Y_OFFSET = -(CUBE_SCALE * maze_y_dim) / 2 + CUBE_SCALE / 2

OPEN_SPACE = 0
WALL_NORM = 2
WALL_RIGHT = 3
WALL_LEFT = 4
WALL_GOAL = 5

floor = spawn_actor(Vector(0, 0, 0), "/Game/Floor.Floor")
floor.set_actor_scale3d(Vector(maze_x_dim, maze_y_dim, 1))  # type: ignore

walls = []
for yi, row in enumerate(maze):

    y = yi * CUBE_SCALE + MAZE_X_OFFSET

    # Unreal Engine uses left-handed coordinate system
    for xi, col in enumerate(reversed(row)):

        x = xi * CUBE_SCALE + MAZE_Y_OFFSET

        if col == WALL_NORM:
            walls.append(spawn_actor(Vector(x, y, WALL_Z_OFFSET), "/Game/Wall.Wall"))
        elif col == WALL_RIGHT:
            walls.append(
                spawn_actor(Vector(x, y, WALL_Z_OFFSET), "/Game/WallRight.WallRight")
            )
        elif col == WALL_LEFT:
            walls.append(
                spawn_actor(Vector(x, y, WALL_Z_OFFSET), "/Game/WallLeft.WallLeft")
            )
        elif col == WALL_GOAL:
            walls.append(
                spawn_actor(Vector(x, y, WALL_Z_OFFSET), "/Game/WallGoal.WallGoal")
            )

# Start at (maze_x_dim/2, -maze_y_dim/2)
# Goal at (-maze_x_dim/2, maze_y_dim/2)
