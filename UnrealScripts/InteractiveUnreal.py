#!/usr/bin/env python

import matplotlib.pyplot as plt

import sys

sys.path.append("../Utilities")
from UnrealUtilities import UE4EnvWrapper  # type: ignore


def main():
    env = UE4EnvWrapper()
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
