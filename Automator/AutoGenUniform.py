#!/usr/bin/env python

# TODO:
# - work with image sequences (just two for stacking)
# - unify to work with single function for straight and cornering
#   - need center of next turn cell for turning?
#   - need to know if we are moving to a new cell (no longer turning)
# - remove "Dir." from python files
# - pick FOV

from __future__ import annotations

import sys
from argparse import ArgumentParser
from math import atan2, degrees, pi, radians
from pathlib import Path
from random import choice, random
from typing import List, Optional, Tuple

sys.path.append("../PycastWorld")
from pycaster import PycastWorld  # type: ignore

sys.path.append("../MazeGen")
from MazeUtils import bfs_dist_maze, read_maze_file  # type: ignore

ANG_NOISE_MAG = radians(45)
POS_NOISE_MAG = 0.4

DIRS_TO_TURN = {
    ("NORTH", "EAST"): "RIGHT",
    ("NORTH", "WEST"): "LEFT",
    ("EAST", "NORTH"): "LEFT",
    ("EAST", "SOUTH"): "RIGHT",
    ("SOUTH", "EAST"): "LEFT",
    ("SOUTH", "WEST"): "RIGHT",
    ("WEST", "NORTH"): "RIGHT",
    ("WEST", "SOUTH"): "LEFT",
}

DIR_TO_ANGLE = {
    "EAST": radians(0),
    "WEST": radians(180),
    "NORTH": radians(90),
    "SOUTH": radians(270),
}


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
        return f"Pt({self.x}, {self.y})"

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


def filename_from_angle_deg(angle: float, i: int) -> str:
    return f"{i:>06}_{angle:.3f}".replace(".", "p") + ".png"


def process_maze(
    maze_filepath: str,
) -> Tuple[List[Tuple[Pt, Pt, Pt, str, int, int]], List[Tuple[Pt, str, str]]]:

    # Loop to find the first change in direction
    def get_next_dir(maze_directions, curr_heading):
        turnx, turny, next_heading = next(maze_directions)
        while next_heading == curr_heading:
            turnx, turny, next_heading = next(maze_directions)
        return turnx, turny, next_heading

    # Compute path from maze file
    maze, _, _, maze_directions, _ = read_maze_file(maze_filepath)
    x_start, y_start, _ = maze_directions[0]
    x_end, y_end, _ = maze_directions[-1]
    _, correct_path = bfs_dist_maze(maze, x_start, y_start, x_end, y_end)

    # Compute the direction and next_turn cell for each cell along the path
    maze_directions = iter(maze_directions)
    _, _, curr_heading = next(maze_directions)  # First directions
    curr_heading = curr_heading

    # Set these so that they are at reasonable defaults for the first iteration
    turnx, turny, next_heading = get_next_dir(maze_directions, curr_heading)

    prev_heading = ""
    prev_action = None
    next_action = DIRS_TO_TURN.get((curr_heading, next_heading), "FORWARD")

    path_cells = []
    turn_cells = []

    for (x, y) in correct_path[:-1]:  # We don't need images for the final cell

        # Update information when we make a turn
        if x == turnx and y == turny:

            is_turn_cell = True
            prev_heading = curr_heading
            curr_heading = next_heading
            prev_action = next_action

            try:
                turnx, turny, next_heading = get_next_dir(maze_directions, curr_heading)
            except StopIteration:
                pass

            # Default to FORWARD when curr_heading toward goal cell and we don't
            # have a next heading
            next_action = DIRS_TO_TURN.get((curr_heading, next_heading), "FORWARD")

        else:
            is_turn_cell = False

        # Right and left corners depend on curr_heading
        rightx = turnx if curr_heading in ("SOUTH", "EAST") else turnx + 1
        righty = turny if curr_heading in ("NORTH", "EAST") else turny + 1
        right_corner = Pt(rightx, righty)

        leftx = turnx if curr_heading in ("NORTH", "EAST") else turnx + 1
        lefty = turny if curr_heading in ("NORTH", "WEST") else turny + 1
        left_corner = Pt(leftx, lefty)

        # Add 0.5 offset to center position in the hallway
        pos = Pt(x + 0.5, y + 0.5)

        path_cells.append((pos, right_corner, left_corner, curr_heading, turnx, turny))

        if is_turn_cell:
            corner = left_corner if prev_action == "RIGHT" else right_corner
            turn_cells.append((pos, prev_heading, prev_action, corner))

    return path_cells, turn_cells


def capture_image(
    angle: float, i: int, save_dir: str, demo: bool, world: PycastWorld,
):
    angle = degrees(angle_from_negpi_to_pospi(angle))
    filename = filename_from_angle_deg(angle, i)

    # Display image instead of saving
    if demo:
        import matplotlib.pyplot as plt
        import numpy as np

        image = np.array(world)
        plt.imshow(image)
        plt.show()

        print(f"File not saved in demo mode ({filename}).")

    else:
        print("File saved: ", filename)
        world.save_png(str(Path(save_dir) / filename))


def capture_straightaway_images(
    world: PycastWorld, num_images: int, path_cells, save_dir: str, demo: bool
):
    # Select position along path
    for i in range(num_images):

        # TODO: not using turnx or turny
        pos, right_corner, left_corner, heading, turnx, turny = choice(path_cells)

        # Perturb the position and heading
        perturbed_pos = pos + Pt.rand((-POS_NOISE_MAG, POS_NOISE_MAG))
        x, y = perturbed_pos.xy

        ang_noise = ANG_NOISE_MAG * (2 * random() - 1)
        angle = angle_from_zero_to_2pi(DIR_TO_ANGLE[heading] + ang_noise)

        world.set_position(x, y)
        world.set_direction(angle)

        # Compute angles to the two corners of the turn
        angle_to_right = angle_from_zero_to_2pi(Pt.angle(right_corner - perturbed_pos))
        angle_to_left = angle_from_zero_to_2pi(Pt.angle(left_corner - perturbed_pos))

        # Angles can straddle 0/360 when facing EAST
        if angle_to_right > angle_to_left:
            angle = angle_from_negpi_to_pospi(angle)
            angle_to_right = angle_from_negpi_to_pospi(angle_to_right)
            angle_to_left = angle_from_negpi_to_pospi(angle_to_left)

        # Now compute the "correct" action
        if angle < angle_to_right:
            angle_label = angle_to_right - angle
            action = "LEFT"
        elif angle > angle_to_left:
            angle_label = angle_to_left - angle
            action = "RIGHT"
        else:
            angle_label = 0
            action = "FORWARD"

        print(
            f"{i:>6} : ",
            f"{x:5.2f}",
            f"{y:5.2f}",
            heading.rjust(5),
            f"{degrees(angle_to_right): 7.2f}",
            f"{degrees(angle_to_left): 7.2f}",
            f"{degrees(angle): 7.2f}",
            f"{degrees(angle_label): 7.2f}",
            action.rjust(7),
        )

        capture_image(angle_label, i, save_dir, demo, world)


def capture_cornering_images(
    world: PycastWorld,
    num_images: int,
    turn_cells,
    save_dir: str,
    demo: bool,
    istart: int,
):
    for i in range(num_images):

        i += istart

        pos, heading, action, corner = choice(turn_cells)

        # Perturb the position and heading
        perturbed_pos = pos + Pt.rand((-POS_NOISE_MAG, POS_NOISE_MAG))
        x, y = perturbed_pos.xy

        ang_noise = ANG_NOISE_MAG * (2 * random() - 1)
        angle = angle_from_zero_to_2pi(DIR_TO_ANGLE[heading] + ang_noise)

        # TODO: check if arrow is in field of view

        world.set_position(x, y)
        world.set_direction(angle)

        angle_to_corner = angle_from_zero_to_2pi(Pt.angle(corner - perturbed_pos))

        angle_label = (
            angle - angle_to_corner
            if angle > angle_to_corner
            else angle_to_corner - angle
        )
        angle_label = angle_from_negpi_to_pospi(angle_label)

        print(
            f"{i:>6} : ",
            f"{x:5.2f}",
            f"{y:5.2f}",
            heading.rjust(5),
            f"{degrees(angle): 7.2f}",
            f"{degrees(angle_label): 7.2f}",
            action.rjust(7),
        )

        capture_image(angle_label, i, save_dir, demo, world)


def main():

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
        "image_width", type=int, default=224, help="Width of generated images."
    )
    arg_parser.add_argument(
        "image_height", type=int, default=224, help="Height of generated images."
    )

    # TODO: currently unused
    arg_parser.add_argument(
        "--sequence", type=int, default=1, help="Number of images per sequence."
    )

    arg_parser.add_argument(
        "--demo", action="store_true", help="Display images instead of saving."
    )

    args = arg_parser.parse_args()

    path_cells, turn_cells = process_maze(args.maze_filepath)
    print(turn_cells)

    # Create world
    world = PycastWorld(args.image_width, args.image_height, args.maze_filepath)

    capture_straightaway_images(
        world, args.num_straight_images, path_cells, args.save_dir, args.demo
    )
    capture_cornering_images(
        world,
        args.num_straight_images,
        turn_cells,
        args.save_dir,
        args.demo,
        args.num_straight_images,
    )


if __name__ == "__main__":
    main()
