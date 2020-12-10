#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from math import radians

# Needed to import pycaster from relative path
import sys

sys.path.append("../PycastWorld")
from pycaster import RaycastWorld, Turn, Walk

world = RaycastWorld(320, 240, "../Worlds/maze.txt")
world.direction(0, 1.152)

for frame in range(10):

    # Get image
    image_data = np.array(world)

    # Convert image_data and give to network... work for Jared
    # x = ... x is the direction to move

    # Move in world
    if frame % 2 == 0:
        world.walk(Walk.Forward)
        world.turn(Turn.Stop)
    elif frame % 3 == 0:
        world.walk(Walk.Stopped)
        world.turn(Turn.Left)
    else:
        world.walk(Walk.Stopped)
        world.turn(Turn.Right)

    world.update()

    print("Showing frame", frame)
    plt.imshow(image_data)
    plt.show()
