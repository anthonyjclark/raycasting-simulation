#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from subprocess import run

arg_parser = ArgumentParser("Generate images using list of provided mazes")
arg_parser.add_argument("maze_dir", help="Path to a maze filess.")
arg_parser.add_argument("output_dir", help="Directory in which to store images.")
arg_parser.add_argument("total_images", type=int, help="Total number of images.")
args = arg_parser.parse_args()

maze_files = list(Path(args.maze_dir).glob("*.txt"))
images_per_maze = args.total_images // len(maze_files)

ioffset = 0
for maze_filename in maze_files:

    cmd = [
        "python",
        "AutoGenUniform.py",
        str(maze_filename),
        args.output_dir,
        str(images_per_maze // 2),
        str(images_per_maze // 2),
        "224",
        "224",
        str(ioffset),
    ]

    print(" ".join(cmd))

    run(cmd)

    ioffset += images_per_maze
