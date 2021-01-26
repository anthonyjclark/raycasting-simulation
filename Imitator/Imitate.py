#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from fastai.vision.all import *
from math import radians

# Needed to import pycaster from relative path
import sys

sys.path.append("../PycastWorld")
sys.path.append("../Models")
from pycaster import RaycastWorld, Turn, Walk

# TODO: this stuff probably shouldn't be hardcoded
world = RaycastWorld(320, 240, "../Worlds/maze.txt")
world.direction(0, 1.152)

path = Path("../")
model_inf = load_learner(path / "Models/export3.pkl")

for frame in range(400):

    # Get image
    image_data = np.array(world)

    # Convert image_data and give to network
    move = model_inf.predict(image_data)[0]
    # print(move)

    # Move in world
    if move == "forward":
        world.walk(Walk.Forward)
        world.turn(Turn.Stop)
    elif move == "left":
        world.walk(Walk.Stopped)
        world.turn(Turn.Left)
    else:
        world.walk(Walk.Stopped)
        world.turn(Turn.Right)

    world.update()

    if frame % 1 == 0:
        print("Showing frame", frame)
        plt.imshow(image_data)
        plt.show()
