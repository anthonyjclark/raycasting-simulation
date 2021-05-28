#!/usr/bin/env python

# TODO: add gif output using matplotlib animation

import matplotlib.pyplot as plt
import numpy as np

from fastai.vision.all import *
from math import radians

# Needed to import pycaster from relative path
import sys

sys.path.append("../PycastWorld")
sys.path.append("../Models")
from pycaster import PycastWorld, Turn, Walk

# functions defined for model required by fastai
def parent_to_deg(f):
    parent = parent_label(f)
    if parent == "left":
        return 90.0
    elif parent == "right":
        return -90.0
    else:
        return 0.0


def sin_cos_loss(preds, targs):
    rad_targs = targs / 180 * np.pi
    x_targs = torch.cos(rad_targs)
    y_targs = torch.sin(rad_targs)
    x_preds = preds[:, 0]
    y_preds = preds[:, 1]
    return ((x_preds - x_targs) ** 2 + (y_preds - y_targs) ** 2).mean()


def within_angle(preds, targs, angle):
    rad_targs = targs / 180 * np.pi
    angle_pred = torch.atan2(preds[:, 1], preds[:, 0])
    abs_diff = torch.abs(rad_targs - angle_pred)
    angle_diff = torch.where(abs_diff > np.pi, np.pi * 2.0 - abs_diff, abs_diff)
    return torch.where(angle_diff < angle, 1.0, 0.0).mean()


def within_45_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 4)


def within_30_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 6)


def within_15_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 12)


def main():
    maze = sys.argv[1] if len(sys.argv) > 1 else "../Worlds/maze.txt"
    model = sys.argv[2] if len(sys.argv) > 2 else "../Models/auto-gen-c.pkl"
    show_freq = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # frequency to show frames
    model_type = sys.argv[4] if len(sys.argv) > 4 else 'c'  # 'c' for classification, 'r' for regresssion


    world = PycastWorld(320, 240, maze)

    path = Path("../")
    model_inf = load_learner(path / model)
    prev_move = None

    for frame in range(2000):

        # Get image
        image_data = np.array(world)

        # Convert image_data and give to network
        if model_type == "c":
            move = model_inf.predict(image_data)[0]
        elif model_type == "r":
            pred_coords, _, _ = model_inf.predict(image_data)
            pred_angle = np.arctan2(pred_coords[1], pred_coords[0]) / np.pi * 180
            pred_angle = pred_angle % (360)

            if pred_angle > 45 and pred_angle <= 180:
                move = "left"
            elif pred_angle > 180 and pred_angle < 315:
                move = "right"
            else: 
                move = "straight"

        print(move)

        if move == "left" and prev_move == "right":
            move = "straight"
        elif move =="right" and prev_move == "left":
            move = "straight"

        # Move in world
        if move == "straight":
            world.walk(Walk.Forward)
            world.turn(Turn.Stop)
        elif move == "left":
            world.walk(Walk.Stopped)
            world.turn(Turn.Left)
        else:
            world.walk(Walk.Stopped)
            world.turn(Turn.Right)
        
        prev_move = move

        world.update()

        if show_freq != 0 and frame % show_freq == 0:
            print("Showing frame", frame)
            plt.imshow(image_data)
            plt.show()

if __name__ == "__main__":
    main()
