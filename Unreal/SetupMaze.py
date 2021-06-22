import unreal
from unreal import Vector

from os.path import realpath
from pathlib import Path
from typing import List, Tuple

"""
Drag texture pngs to content browser
Right-click texture png and create material
Double-click material, add TextureCoordinate and adjust (20 for floor)
Drag cube to content browser
Double-click cube, add created material
(This script does the rest)
"""

# ../Mazes/maze01.txt
maze_filepath = Path(realpath(__file__)).parents[1] / "Mazes/maze01.txt"


def read_maze_file(
    filepath: str,
) -> Tuple[List[List[int]], int, int, List[Tuple[int, int, str]], List[str]]:
    """Read a maze file and return values

    Args:
        filepath (str): path to a maze file

    Returns:
        Tuple[List[List[int]]: a maze
        int: maze width
        int: maze height
        List[Tuple[int, int, str]]: turn by turn directions
        List[str]]: texture names
    """
    with open(filepath, "r") as maze_file:
        num_textures = int(maze_file.readline())
        texture_names = [maze_file.readline() for _ in range(num_textures)]
        maze_x_dim, maze_y_dim = [int(dim) for dim in maze_file.readline().split()]
        maze = [
            [int(cell) for cell in maze_file.readline().split()]
            for _ in range(maze_y_dim)
        ]
        maze_directions = [
            (int(line.split()[0]), int(line.split()[1]), line.split()[2])
            for line in maze_file.readlines()
        ]

    return list(reversed(maze)), maze_x_dim, maze_y_dim, maze_directions, texture_names


def spawn_actor(location, meshpath):

    new_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, location=location
    )
    mesh = unreal.load_object(None, meshpath)
    mesh_component = new_actor.get_editor_property("static_mesh_component")  # type: ignore
    mesh_component.set_static_mesh(mesh)

    return new_actor


maze, maze_x_dim, maze_y_dim, _, _ = read_maze_file(maze_filepath)

CUBE_SCALE = 100
WALL_Z_OFFSET = CUBE_SCALE

MAZE_X_OFFSET = -(CUBE_SCALE * maze_x_dim) / 2 + CUBE_SCALE / 2
MAZE_Y_OFFSET = -(CUBE_SCALE * maze_y_dim) / 2 + CUBE_SCALE / 2

OPEN_SPACE = 0
WALL_NORM = 2
WALL_RIGHT = 3
WALL_LEFT = 4
WALL_GOAL = 5

floor = spawn_actor(Vector(8, 8, 0), "/Game/Floor.Floor")
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
