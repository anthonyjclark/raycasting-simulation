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

# functions defined for model required by fastai

def parent_to_deg(f):
    parent = parent_label(f)
    if parent == 'left': return 90.
    elif parent == 'right': return -90.
    else: return 0.

def sin_cos_loss(preds, targs):
    rad_targs = targs / 180 * np.pi
    x_targs = torch.cos(rad_targs)
    y_targs = torch.sin(rad_targs)
    x_preds = preds[:, 0]
    y_preds = preds[:, 1]
    return ((x_preds - x_targs)**2 + (y_preds - y_targs)**2).mean()

def within_angle(preds, targs, angle):
    rad_targs = targs / 180 * np.pi
    angle_pred = torch.atan2(preds[:,1], preds[:,0])
    abs_diff = torch.abs(rad_targs - angle_pred)
    angle_diff = torch.where(abs_diff > np.pi, np.pi*2. - abs_diff, abs_diff)
    return torch.where(angle_diff < angle, 1., 0.).mean()

def within_45_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 4)

def within_30_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 6)

def within_15_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 12)

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
