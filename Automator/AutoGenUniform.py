#!/usr/bin/env python

# TODO:
# - work with image sequences (just two for stacking)
# - remove "Dir." from python files (get PR first)
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

ANG_NOISE_MAG = radians(40)
POS_NOISE_MAG = 0.35

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


def ang_plus_minus_pi(angle: float) -> float:
    angle = angle % (2 * pi)
    if angle < -pi:
        angle += 2 * pi
    elif angle > pi:
        angle -= 2 * pi
    return angle


def ang_zero_to_2pi(angle: float) -> float:
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
            try:
                turnx, turny, next_heading = next(maze_directions)
            except StopIteration:
                break
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

            turnx, turny, next_heading = get_next_dir(maze_directions, curr_heading)

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

        cell_info = (
            pos,
            right_corner,
            left_corner,
            curr_heading,
            prev_heading,
            prev_action,
        )

        path_cells.append(cell_info)

        if is_turn_cell:
            turn_cells.append(cell_info)

    return path_cells, turn_cells


def capture_image(
    angle: float,
    i: int,
    save_dir: str,
    demo: bool,
    world: PycastWorld,
):
    angle = degrees(ang_plus_minus_pi(angle))
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


def capture_images(
    world: PycastWorld,
    num_images: int,
    cells,
    save_dir: str,
    is_cornering: bool,
    ioffset: int,
    demo: bool,
):

    # Print column header first time through
    try:
        _ = capture_images.first_call
    except AttributeError:
        capture_images.first_call = False
        print("  I         X     Y   HEAD  NOISE  RIGHT  ANGLE  LEFT   ACTION   NAME")

    # Select position along path
    for i in range(num_images):

        i += ioffset

        pos, rcorner, lcorner, cheading, pheading, turn_action = choice(cells)

        # Heading is determined by cell case (cornering or not cornering)
        heading = pheading if is_cornering else cheading

        # Perturb the position
        perturbed_pos = pos + Pt.rand((-POS_NOISE_MAG, POS_NOISE_MAG))
        world.set_position(perturbed_pos.x, perturbed_pos.y)

        # Perturb the heading
        ang_noise = ANG_NOISE_MAG * (2 * random() - 1)
        angle = ang_zero_to_2pi(DIR_TO_ANGLE[heading] + ang_noise)
        world.set_direction(angle)

        # Compute angles to the two corners of the turn
        angle_to_right = ang_zero_to_2pi(Pt.angle(rcorner - perturbed_pos))
        angle_to_left = ang_zero_to_2pi(Pt.angle(lcorner - perturbed_pos))

        # Angles can straddle 0/360 when facing EAST
        if angle_to_right > angle_to_left:
            angle = ang_plus_minus_pi(angle)
            angle_to_right = ang_plus_minus_pi(angle_to_right)
            angle_to_left = ang_plus_minus_pi(angle_to_left)

        # Now compute the "correct" action
        if is_cornering and turn_action == "LEFT":
            action = turn_action
            action_angle = ang_plus_minus_pi(angle_to_right) - ang_plus_minus_pi(angle)
        elif is_cornering and turn_action == "RIGHT":
            action = turn_action
            action_angle = ang_plus_minus_pi(angle) - ang_plus_minus_pi(angle_to_left)
        elif angle_to_right <= angle <= angle_to_left:
            action = "FORWARD"
            action_angle = 0.0
        elif angle < angle_to_right:
            action = "LEFT"
            action_angle = ang_plus_minus_pi(angle_to_right) - ang_plus_minus_pi(angle)
        else:
            action = "RIGHT"
            action_angle = ang_plus_minus_pi(angle) - ang_plus_minus_pi(angle_to_left)

        print(
            f"{i:>6} :",
            f"{perturbed_pos.x:5.1f}",
            f"{perturbed_pos.y:5.1f}",
            heading.rjust(5),
            f"{degrees(ang_noise):6.1f}",
            f"{degrees(angle_to_right): 6.1f}",
            f"{degrees(angle): 6.1f}",
            f"{degrees(angle_to_left): 6.1f}",
            f"{degrees(action_angle): 6.1f}",
            action.rjust(7),
        )

        capture_image(action_angle, i, save_dir, demo, world)


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

    # Create world
    world = PycastWorld(args.image_width, args.image_height, args.maze_filepath)

    # Capture images on corridor straightaways
    capture_images(
        world=world,
        num_images=args.num_straight_images,
        cells=path_cells,
        save_dir=args.save_dir,
        is_cornering=False,
        ioffset=0,
        demo=args.demo,
    )

    # Capture images from corners while turning
    capture_images(
        world=world,
        num_images=args.num_turn_images,
        cells=turn_cells,
        save_dir=args.save_dir,
        is_cornering=True,
        ioffset=args.num_straight_images,
        demo=args.demo,
    )


if __name__ == "__main__":
    main()
