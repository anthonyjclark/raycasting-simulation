#!/usr/bin/env python

# TODO:
# - work with regression (be pickier about going down middle of hallway)
# - work with image sequences
# - radians(30)
# - give some leeway when in the center of the hall (not close to wall)

from __future__ import annotations

from argparse import ArgumentParser
from math import atan2, degrees, pi, radians
from pathlib import Path
from random import choice, random
from typing import Tuple, Optional

import sys

sys.path.append("../PycastWorld")
from pycaster import PycastWorld  # type: ignore

sys.path.append("../MazeGen")
from MazeUtils import read_maze_file, bfs_dist_maze  # type: ignore


class Pt:
    def __init__(self, x: float = 0, y: float = 0) -> None:
        self.x = x
        self.y = y

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __sub__(self, other) -> Pt:
        return Pt(self.x - other.x, self.y - other.y)

    def __add__(self, other) -> Pt:
        return Pt(self.x + other.x, self.y + other.y)

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def rand(
        cls, lims: Tuple[float, float], ylims: Optional[Tuple[float, float]] = None
    ) -> Pt:
        xlo, xhi = lims
        ylo, yhi = ylims if ylims else lims
        return Pt((xhi - xlo) * random() + xlo, (yhi - ylo) * random() + ylo)

    @classmethod
    def angle(cls, pt: Pt) -> float:
        return atan2(pt.y, pt.x)


def angle_from_negpi_to_pospi(angle: float) -> float:
    angle = angle % (2 * pi)
    if angle < -pi:
        angle += 2 * pi
    elif angle > pi:
        angle -= 2 * pi
    return angle


def angle_from_zero_to_2pi(angle: float) -> float:
    angle = angle % (2 * pi)
    return angle if angle >= 0 else angle + (2 * pi)


def main():

    ANG_NOISE_MAG = radians(30)
    POS_NOISE_MAG = 0.4

    arg_parser = ArgumentParser("Run a maze utility")
    arg_parser.add_argument("maze_filepath", help="Path to a maze file.")
    arg_parser.add_argument("save_dir", help="Save location (directory must exist).")
    arg_parser.add_argument(
        "num_straight_images", type=int, help="Number of straightaway images."
    )
    arg_parser.add_argument(
        "num_turn_images", type=int, help="Number of turning specific images."
    )
    arg_parser.add_argument(
        "--demo", action="store_true", help="Display images instead of saving."
    )

    # TODO: currently unused
    arg_parser.add_argument("sequence", type=int, help="Number of images per sequence.")

    args = arg_parser.parse_args()

    # Compute path from maze file
    maze, _, _, maze_directions, _ = read_maze_file(args.maze_filepath)
    x_start, y_start, _ = maze_directions[0]
    x_end, y_end, _ = maze_directions[-1]
    _, correct_path = bfs_dist_maze(maze, x_start, y_start, x_end, y_end)

    dirs_to_turn = {
        ("NORTH", "EAST"): "RIGHT",
        ("NORTH", "WEST"): "LEFT",
        ("EAST", "NORTH"): "LEFT",
        ("EAST", "SOUTH"): "RIGHT",
        ("SOUTH", "EAST"): "LEFT",
        ("SOUTH", "WEST"): "RIGHT",
        ("WEST", "NORTH"): "RIGHT",
        ("WEST", "SOUTH"): "LEFT",
    }

    dir_to_angle = {
        "EAST": radians(0),
        "WEST": radians(180),
        "NORTH": radians(90),
        "SOUTH": radians(270),
    }

    # Compute the direction and next_turn cell for each cell along the path
    maze_directions = iter(maze_directions)
    _, _, direction = next(maze_directions)  # First directions

    # Loop to find the next change in direction
    turnx, turny, next_direction = next(maze_directions)
    while next_direction == direction:
        turnx, turny, next_direction = next(maze_directions)

    # Set these so that they are at reasonable defaults for the first iteration
    is_turn_cell = False
    prev_direction = ""
    prev_turn = None
    prev_heading = ""
    upcoming_turn = None

    path_cells = []
    turn_cells = []
    for (x, y) in correct_path[:-1]:  # We don't need images for the final cell
        if x == turnx and y == turny:
            is_turn_cell = True
            prev_direction = direction
            prev_turn = upcoming_turn
            direction = next_direction
            try:
                turnx, turny, next_direction = next(maze_directions)
                while next_direction == direction:
                    turnx, turny, next_direction = next(maze_directions)
            except StopIteration:
                pass

        heading = direction[4:]
        next_heading = next_direction[4:]
        prev_heading = prev_direction[4:]

        # Default to FORWARD when heading toward goal cell
        upcoming_turn = dirs_to_turn.get((heading, next_heading), "FORWARD")

        # Right and left corners depend on heading
        rightx = turnx if heading in ("SOUTH", "EAST") else turnx + 1
        righty = turny if heading in ("NORTH", "EAST") else turny + 1
        right_corner = Pt(rightx, righty)

        leftx = turnx if heading in ("NORTH", "EAST") else turnx + 1
        lefty = turny if heading in ("NORTH", "WEST") else turny + 1
        left_corner = Pt(leftx, lefty)

        # Add 0.5 offset to center in the hallway
        pos = Pt(x + 0.5, y + 0.5)

        path_cells.append((pos, right_corner, left_corner, heading, turnx, turny,))

        if is_turn_cell:
            turn_cells.append((pos, prev_heading, prev_turn))

        # Reset for the next cell
        is_turn_cell = False

    # Create world
    world = PycastWorld(320, 240, args.maze_filepath)
    FOV = radians(66)

    # Select position along path
    for i in range(args.num_straight_images):

        # TODO: not using turnx or turny
        pos, right_corner, left_corner, heading, turnx, turny = choice(path_cells)

        # Perturb the position and heading
        perturbed_pos = pos + Pt.rand((-POS_NOISE_MAG, POS_NOISE_MAG))
        x, y = perturbed_pos.xy

        ang_noise = ANG_NOISE_MAG * (2 * random() - 1)
        angle = angle_from_zero_to_2pi(dir_to_angle[heading] + ang_noise)

        world.position(x, y, 0)  # z=0 is at vertical center
        world.direction(angle, FOV)  # 1.152 is the default FOV

        # Compute angles to the two corners of the turn
        angle_to_right = angle_from_zero_to_2pi(Pt.angle(right_corner - perturbed_pos))
        angle_to_left = angle_from_zero_to_2pi(Pt.angle(left_corner - perturbed_pos))

        # Angles can straddle 0/360 when facing EAST
        if angle_to_right > angle_to_left:
            angle = angle_from_negpi_to_pospi(angle)
            angle_to_right = angle_from_negpi_to_pospi(angle_to_right)
            angle_to_left = angle_from_negpi_to_pospi(angle_to_left)

        # Compute the "correct" action

        if angle < angle_to_right:
            action = "LEFT"
        elif angle > angle_to_left:
            action = "RIGHT"
        else:
            action = "FORWARD"

        print(
            f"{i:>6} : ",
            f"{x:5.2f}",
            f"{y:5.2f}",
            heading.rjust(5),
            f"{degrees(angle_to_right): 7.2f}",
            f"{degrees(angle_to_left): 7.2f}",
            f"{degrees(angle): 7.2f}",
            action.rjust(7),
        )

        filename = f"{i:>06}.png"

        if args.demo:
            import matplotlib.pyplot as plt
            import numpy as np

            image = np.array(world)
            plt.imshow(image)
            plt.show()

            print(f"File not saved in demo mode ({filename}).")

        else:
            world.savePNG(str(Path(args.save_dir) / action.lower() / filename))

    for i in range(args.num_turn_images):

        i += args.num_straight_images

        pos, heading, turn = choice(turn_cells)

        # Perturb the position and heading
        perturbed_pos = pos + Pt.rand((-POS_NOISE_MAG, POS_NOISE_MAG))
        x, y = perturbed_pos.xy

        ang_noise_mag = radians(30)
        ang_noise = ang_noise_mag * (2 * random() - 1)
        angle = angle_from_zero_to_2pi(dir_to_angle[heading] + ang_noise)

        # TODO: check if arrow is in field of view

        world.position(x, y, 0)  # z=0 is at vertical center
        world.direction(angle, FOV)  # 1.152 is the default FOV

        action = turn

        print(
            f"{i:>6} : ",
            f"{x:5.2f}",
            f"{y:5.2f}",
            heading.rjust(5),
            f"{degrees(angle): 7.2f}",
            action.rjust(7),
        )

        filename = f"{i:>06}.png"

        if args.demo:
            import matplotlib.pyplot as plt
            import numpy as np

            image = np.array(world)
            plt.imshow(image)
            plt.show()

            print(f"File not saved in demo mode ({filename}).")

        else:
            world.savePNG(str(Path(args.save_dir) / action.lower() / filename))


if __name__ == "__main__":
    main()
