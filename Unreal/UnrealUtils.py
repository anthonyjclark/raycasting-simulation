#!/usr/bin/env python

from numpy import pi, cos, sin
import matplotlib.pyplot as plt

import sys

sys.path.append("/home/eoca2018/unrealcv/client/python")
from unrealcv import client as ue4  # type: ignore
from unrealcv.util import read_png  # type: ignore


class UE4EnvWrapper:
    def __init__(self, rc_x: 0, rc_y: 0):

        ue4.connect(timeout=5)

        self.connected = False
        if not ue4.isconnected():
            print("UnrealCV server is not running.")
        else:
            self.connected = True
            print(ue4.request("vget /unrealcv/status"))

        # TODO: these are specific to the default maze
#         self.initial_x = 700
#         self.initial_y = -700
        self.initial_x = 1050 - (100 * rc_x)
        self.initial_y = -1050 + (100 * rc_y)
        self.initial_angle = 0

        self.x = self.initial_x
        self.y = self.initial_y
        self.angle = self.initial_angle

        self.turn_speed = 5
        self.walk_speed = 50

        if self.connected:
            self.set_pose()

    def isconnected(self):
        return self.connected

    def reset(self):
        self.set_pose(x=self.initial_x, y=self.initial_y, angle=self.initial_angle)

    def set_pose(self, *, x=None, y=None, angle=None):
        self.x = x if x else self.x
        self.y = y if y else self.y
        self.angle = angle if angle else self.angle
        ue4.request(f"vset /camera/0/pose {self.x} {self.y} 100 0 {self.angle} 0")
    
    def set_rotation(self, *, angle = None):
        self.angle = angle if angle else self.angle
        ue4.request(f"vset /camera/0/rotation 0 {self.angle} 0")

    def left(self):
        self.set_rotation(angle=self.angle - self.turn_speed)

    def right(self):
        self.set_rotation(angle=self.angle + self.turn_speed)

    def forward(self):
        ue4.request(f"vset /action/keyboard up .075")

    def request_image(self):
        image_data = ue4.request(f"vget /camera/0/lit png")
        return read_png(image_data)

    def show(self):
        plt.imshow(self.request_image())
