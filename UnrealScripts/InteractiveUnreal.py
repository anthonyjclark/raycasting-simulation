#!/usr/bin/env python

import matplotlib.pyplot as plt
from numpy import pi, cos, sin

import sys

sys.path.append("/home/ajc/UE4/unrealcv/client/python")
from unrealcv import client as ue4  # type: ignore
from unrealcv.util import read_png  # type: ignore


class UE4EnvWrapper:
    def __init__(self, ue4):
        self.ue4 = ue4

        # TODO: these are specific to the default maze
        self.x = 700
        self.y = -700
        self.angle = 0

        self.turn_speed = 5
        self.walk_speed = 50

        self.set_pose()

    def set_pose(self, *, x=None, y=None, angle=None):
        self.x = x if x else self.x
        self.y = y if y else self.y
        self.angle = angle if angle else self.angle
        self.ue4.request(f"vset /camera/0/pose {self.x} {self.y} 100 0 {self.angle} 0")

    def left(self):
        self.set_pose(angle=self.angle - self.turn_speed)

    def right(self):
        self.set_pose(angle=self.angle + self.turn_speed)

    def forward(self):
        new_x = self.x + cos(self.angle * pi / 180.0) * self.walk_speed
        new_y = self.y + sin(self.angle * pi / 180.0) * self.walk_speed
        self.set_pose(x=new_x, y=new_y)

    def request_image(self):
        image_data = ue4.request(f"vget /camera/0/lit png")
        return read_png(image_data)

    def show(self):
        plt.imshow(self.request_image())


def main():

    ue4.connect()

    if not ue4.isconnected():
        print("UnrealCV server is not running.")
        raise SystemExit
    else:
        print(ue4.request("vget /unrealcv/status"))

    env = UE4EnvWrapper(ue4)
    fig, ax = plt.subplots()
    ax.imshow(env.request_image())

    def onpress(event):

        if event.key == "w" or event.key == "up":
            env.forward()

        elif event.key == "a" or event.key == "left":
            env.left()

        elif event.key == "d" or event.key == "right":
            env.right()

        else:
            return

        ax.imshow(env.request_image())
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", onpress)
    plt.title("Unreal Engine View")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
