#!/usr/bin/env python

"""Generate mazes using the growing tree algorithm

This code is based on that of Jamis Buck found here:
https://weblog.jamisbuck.org/2011/1/27/maze-generation-growing-tree-algorithm
https://weblog.jamisbuck.org/2015/10/31/mazes-blockwise-geometry.html

And from Bob Nystrom:
https://journal.stuffwithstuff.com/2014/12/21/rooms-and-mazes/
"""

from argparse import ArgumentParser
from enum import IntEnum
from math import copysign
from random import randrange, choice, seed
from time import sleep
from typing import Dict, List, Optional, Tuple

import numpy as np


class Colors(object):
    def __init__(
        self, *, wall: int, hall: int, path: int, left: int, right: int
    ) -> None:
        self.wall = wall
        self.hall = hall
        self.path = path
        self.left = left
        self.right = right


class Dir(IntEnum):
    NORTH = 1
    EAST = 2
    SOUTH = 4
    WEST = 8

    @classmethod
    def opposite(cls, dir: int) -> int:
        if dir == cls.NORTH:
            return cls.SOUTH
        elif dir == cls.EAST:
            return cls.WEST
        elif dir == cls.SOUTH:
            return cls.NORTH
        else:
            return cls.EAST


class Turn(IntEnum):
    LEFT = 0
    RIGHT = 1
    NO = 2


class Coord(object):
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


Path = List[Coord]


def isPath(cell: int, dir: Dir) -> bool:
    return (cell & dir) != 0


def isPathNorth(cell: int) -> bool:
    return isPath(cell, Dir.NORTH)


def isPathEast(cell: int) -> bool:
    return isPath(cell, Dir.EAST)


def isPathSouth(cell: int) -> bool:
    return isPath(cell, Dir.SOUTH)


def isPathWest(cell: int) -> bool:
    return isPath(cell, Dir.WEST)


class Maze(object):
    def __init__(
        self, width: int, height: int, *, val: int = 0, animate: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.min = 0
        self.maze = [[val] * width for _ in range(height)]
        self.generate_maze(animate)

    def __getitem__(self, xy: Coord) -> int:
        return self.maze[xy.y][xy.x]

    def __setitem__(self, xy: Coord, value: int) -> None:
        self.maze[xy.y][xy.x] = value

    def __repr__(self) -> str:

        w, h = self.width, self.height

        # Create the top row
        rows = [" " + "_" * (w * 2 - 1)]

        # Reverse maze to put (0,0) at bottom left
        for y, row in enumerate(reversed(self.maze)):
            row_str = "|"
            for x, cell in enumerate(row):
                xy = Coord(x, y)
                is_zero = self.coordIsZero(xy)
                xynorth = self.northNeighborZero(xy)

                if (is_zero and xynorth) or isPathSouth(cell):
                    row_str += " "
                else:
                    row_str += "_"

                if is_zero and self.eastNeighborZero(xy):
                    xynortheast = y + 1 < h and self[Coord(x + 1, y + 1)] == 0
                    row_str += " " if (xynorth or xynortheast) else "_"

                elif isPathEast(cell):
                    row_str += " " if isPathSouth(cell | row[x + 1]) else "_"

                else:
                    row_str += "|"

            rows.append(row_str)
        return "\n".join(rows)

    def coordIsZero(self, xy: Coord) -> bool:
        return self[xy] == 0

    def coordIfZero(self, xy: Coord) -> Optional[Coord]:
        return xy if self.coordIsZero(xy) else None

    def northOf(self, xy: Coord) -> Optional[Coord]:
        return Coord(xy.x, xy.y + 1) if xy.y < self.height - 1 else None

    def eastOf(self, xy: Coord) -> Optional[Coord]:
        return Coord(xy.x + 1, xy.y) if xy.x < self.width - 1 else None

    def southOf(self, xy: Coord) -> Optional[Coord]:
        return Coord(xy.x, xy.y - 1) if xy.y > self.min else None

    def westOf(self, xy: Coord) -> Optional[Coord]:
        return Coord(xy.x - 1, xy.y) if xy.x > self.min else None

    def northNeighborZero(self, xy: Coord) -> Optional[Coord]:
        northerner = self.northOf(xy)
        return northerner if northerner and self[northerner] == 0 else None

    def eastNeighborZero(self, xy: Coord) -> Optional[Coord]:
        easterner = self.eastOf(xy)
        return easterner if easterner and self[easterner] == 0 else None

    def southNeighborZero(self, xy: Coord) -> Optional[Coord]:
        southerner = self.southOf(xy)
        return southerner if southerner and self[southerner] == 0 else None

    def westNeighborZero(self, xy: Coord) -> Optional[Coord]:
        westerner = self.westOf(xy)
        return westerner if westerner and self[westerner] == 0 else None

    def get_random_neighbor(self, xy: Coord) -> Optional[Tuple[Dir, Coord]]:

        neighbors = []
        if northerner := self.northNeighborZero(xy):
            neighbors.append((Dir.NORTH, northerner))
        if easterner := self.eastNeighborZero(xy):
            neighbors.append((Dir.EAST, easterner))
        if southerner := self.southNeighborZero(xy):
            neighbors.append((Dir.SOUTH, southerner))
        if westerner := self.westNeighborZero(xy):
            neighbors.append((Dir.WEST, westerner))

        return choice((neighbors)) if len(neighbors) else None

    def generate_maze(self, animate: bool) -> None:

        x, y = randrange(args.width), randrange(args.height)
        locations = [Coord(x, y)]

        if animate:
            # Clear the screen
            print(chr(27) + "[2J")
            print(self)

        while locations:
            # TODO: or choose cell at random, oldest, middle, some split
            # Choose most recently added cell and visit a neighbor
            xy = locations[-1]
            neighbor = self.get_random_neighbor(xy)

            if neighbor:
                neigh_dir, neigh_cell = neighbor
                self[xy] |= neigh_dir
                self[neigh_cell] |= Dir.opposite(neigh_dir)
                locations.append(neigh_cell)

                if animate:
                    sleep(0.02)
                    # Reset cursor
                    print(chr(27) + "[H")
                    print(self)

            else:
                locations.pop(-1)

    def get_path_bfs(self, start: Coord, end: Coord) -> Path:

        queue = [start]
        predecessors = {start: start}
        visited = {start}

        def visit_coord_if(path_exists, cfrom, cto):
            if path_exists and cto not in visited:
                queue.append(cto)
                predecessors[cto] = cfrom
                visited.add(cto)

        while True:
            xy = queue.pop(0)
            cell = self[xy]

            if xy == end:
                break

            visit_coord_if(isPathNorth(cell), xy, self.northOf(xy))
            visit_coord_if(isPathEast(cell), xy, self.eastOf(xy))
            visit_coord_if(isPathSouth(cell), xy, self.southOf(xy))
            visit_coord_if(isPathWest(cell), xy, self.westOf(xy))

        path = [end]
        while True:
            path.append(predecessors[path[-1]])
            if path[-1] == start:
                break

        return list(reversed(path))


def generate_block_maze(
    maze: Maze, path: Path, colors: Colors
) -> Tuple[np.ndarray, List]:

    N, E, S, W = 1, 1, -1, -1

    def sc(v):
        """Scale coordinate to block style."""
        return 2 * v + 1

    # def contract(v: int, amount=0.0) -> float:
    #     """Contract number so that it is closer to zero."""
    #     return v - copysign(amount, v)

    def new_dir(
        d: Dir, cfrom: Coord, cto: Coord
    ) -> Tuple[Dir, Turn, Tuple[float, float]]:
        turns: Dict[Tuple[Dir, Dir], Tuple[Turn, Tuple[float, float]]] = {
            (Dir.NORTH, Dir.NORTH): (Turn.NO, (0, 0)),
            (Dir.NORTH, Dir.EAST): (Turn.RIGHT, (0, N)),
            (Dir.NORTH, Dir.WEST): (Turn.LEFT, (0, N)),
            (Dir.EAST, Dir.EAST): (Turn.NO, (0, 0)),
            (Dir.EAST, Dir.NORTH): (Turn.LEFT, (E, 0)),
            (Dir.EAST, Dir.SOUTH): (Turn.RIGHT, (E, 0)),
            (Dir.SOUTH, Dir.SOUTH): (Turn.NO, (0, 0)),
            (Dir.SOUTH, Dir.EAST): (Turn.LEFT, (0, S)),
            (Dir.SOUTH, Dir.WEST): (Turn.RIGHT, (0, S)),
            (Dir.WEST, Dir.WEST): (Turn.NO, (0, 0)),
            (Dir.WEST, Dir.NORTH): (Turn.RIGHT, (W, 0)),
            (Dir.WEST, Dir.SOUTH): (Turn.LEFT, (W, 0)),
        }

        # TODO: dependent upon +- 1 for N/E/S/W
        if cfrom.y < cto.y:
            ndir = Dir.NORTH
        elif cfrom.x < cto.x:
            ndir = Dir.EAST
        elif cto.y < cfrom.y:
            ndir = Dir.SOUTH
        else:
            ndir = Dir.WEST

        turn, step = turns[(d, ndir)]
        return ndir, turn, step

    np_maze = (
        np.ones((args.height * 2 + 1, args.width * 2 + 1), dtype=np.int) * colors.hall
    )

    # Walls along boundaries
    np_maze[0, :] = colors.wall
    np_maze[-1, :] = colors.wall
    np_maze[:, 0] = colors.wall
    np_maze[:, -1] = colors.wall

    directions = []

    for y, row in enumerate(maze.maze):
        for x, cell in enumerate(row):
            if not isPathSouth(cell):
                np_maze[sc(y) + S, sc(x)] = colors.wall
            if not isPathEast(cell):
                np_maze[sc(y), sc(x) + E] = colors.wall
            np_maze[2 * y, 2 * x] = colors.wall

    # Initial direction (compare x values of first two on path)
    cdir = Dir.NORTH if path[0].x == path[1].x else Dir.EAST

    # Draw shortest path
    for cfrom, cto in zip(path, path[1:]):

        ndir, turn, (xstep, ystep) = new_dir(cdir, cfrom, cto)

        # TODO: dependent upon +- 1 for N/E/S/W
        if ndir == Dir.NORTH:
            np_maze[sc(cfrom.y) : sc(cto.y), sc(cfrom.x)] = colors.path
        elif ndir == Dir.EAST:
            np_maze[sc(cfrom.y), sc(cfrom.x) : sc(cto.x)] = colors.path
        elif ndir == Dir.SOUTH:
            np_maze[sc(cto.y) + 1 : sc(cfrom.y) + 1, sc(cfrom.x)] = colors.path
        elif ndir == Dir.WEST:
            np_maze[sc(cfrom.y), sc(cto.x) + 1 : sc(cfrom.x) + 1] = colors.path

        if ndir != cdir:
            # x, y = sc(cfrom.x) + xstep + 0.5, sc(cfrom.y) + ystep + 0.5
            # directions.append((turn, x, y))
            x, y = sc(cfrom.x) + xstep, sc(cfrom.y) + ystep
            np_maze[y, x] = colors.left if turn == Turn.LEFT else colors.right
            # print("From:", cdir, "To:", ndir, "Is a:", turn)
        cdir = ndir

    return np_maze, directions


if __name__ == "__main__":
    arg_parser = ArgumentParser("Generate perfect mazes.")
    arg_parser.add_argument("--width", type=int, default=10, help="Maze width")
    arg_parser.add_argument("--height", type=int, default=10, help="Maze height")
    arg_parser.add_argument("--out", action="store_true", help="Output for file")
    arg_parser.add_argument("--show", action="store_true", help="Show the maze")
    arg_parser.add_argument("--animate", action="store_true", help="Animate the maze")
    arg_parser.add_argument("--seed", type=int)
    args = arg_parser.parse_args()

    if args.seed != None:
        seed(args.seed)

    maze = Maze(args.width, args.height, animate=args.animate)

    start = Coord(0, 0)
    end = Coord(args.width - 1, args.height - 1)
    path = maze.get_path_bfs(start, end)

    colors = Colors(wall=0, hall=4, path=3, left=1, right=2)
    block_maze, directions = generate_block_maze(maze, path, colors)

    if args.out:
        # Print texture file paths
        textures = [
            "../textures/wood.png",
            "../textures/redbrick.png",
            "../textures/redbrick-right.png",
            "../textures/redbrick-left.png",
        ]
        print(len(textures))
        print("\n".join(textures))

        # Print width by height
        print(args.width * 2 + 1, args.height * 2 + 1)

        # Print maze
        bm = np.zeros_like(block_maze)
        bm[block_maze == colors.wall] = 1
        bm[block_maze == colors.right] = 2
        bm[block_maze == colors.left] = 3

        bm = np.flipud(bm)

        s = np.array2string(bm).replace("[", "").replace("]", "")
        s = [line.strip() for line in s.split("\n")]
        print("\n".join(s))

        # # Print turn by turn
        # for (turn, x, y) in directions:
        #     tex = 2 if turn == Turn.LEFT else 3
        #     print(f"{x:.2f} {y:.2f}", tex)

    if args.show:
        import matplotlib.pyplot as plt

        print(maze)
        plt.imshow(block_maze, origin="lower")
        plt.show()
