#!/usr/bin/env python3

"""
Adapted from:
https://www.cs.cmu.edu/~rdriley/112/notes/notes-animations-part1.html
https://lodev.org/cgtutor/raycasting.html
"""

from math import sin, cos, tan, pi
from tkinter import Tk, Canvas, ALL
from pycaster import PycastWorld, Turn, Walk
from enum import IntEnum
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageTk
import time

inside = time.time()
outside1, outisde2 = time.time(), time.time()

# TODO: these should not be hard-coded
NUM_ROWS = 17
NUM_COLS = 17
TILE_SIZE = 64
WIDTH = 640  # NUM_COLS * TILE_SIZE
HEIGHT = 480  # NUM_ROWS * TILE_SIZE

MINIMAP_SCALE = 0.2
AGENT_RADIUS = 64
NUM_RAYS_DISPLAY = 5

VIEW_MINIMAP = False


def init(data, canvas):
    data.world = PycastWorld(WIDTH, HEIGHT, "../Mazes/maze01.txt")
    data.mapWidth = 17  # TODO: hardcoded
    data.mapHeight = 17  # TODO: hardcoded
    data.screenWidth = WIDTH
    data.screenHeight = HEIGHT

    data.array = np.array(data.world, copy=False, dtype=np.uint8)
    data.photo = ImageTk.PhotoImage(image=Image.fromarray(data.array))
    data.image = canvas.create_image(0, 0, image=data.photo, anchor="nw")


def mousePressed(event, data):
    pass


def keyPressed(event, data):
    if event.keysym == "Up":
        data.world.walk(Walk.Forward)
    elif event.keysym == "Down":
        data.world.walk(Walk.Backward)

    if event.keysym == "Right":
        data.world.turn(Turn.Right)
    elif event.keysym == "Left":
        data.world.turn(Turn.Left)

    if event.char == "q":
        data.tkroot.quit()


def keyReleased(event, data):
    if event.keysym == "Up" or event.keysym == "Down":
        data.world.walk(Walk.Stopped)
    elif event.keysym == "Left" or event.keysym == "Right":
        data.world.turn(Turn.Stop)


def updateScene(data):
    data.world.update()


def redrawAll(canvas, data):

    # Draw the raycaster view
    data.world.render()
    data.photo = ImageTk.PhotoImage(image=Image.fromarray(data.array))
    canvas.itemconfigure(data.image, image=data.photo)

    if VIEW_MINIMAP:
        # Draw the map
        for i in range(data.mapHeight):
            for j in range(data.mapWidth):

                xstart = j * TILE_SIZE
                xend = (j + 1) * TILE_SIZE

                ystart = i * TILE_SIZE
                yend = (i + 1) * TILE_SIZE

                color = "#222" if data.world.world[i][j] > 0 else "#999"

                canvas.create_rectangle(
                    xstart * MINIMAP_SCALE,
                    ystart * MINIMAP_SCALE,
                    xend * MINIMAP_SCALE,
                    yend * MINIMAP_SCALE,
                    fill=color,
                )

        # Draw the agent
        xscale = data.screenWidth / data.mapWidth
        yscale = data.screenHeight / data.mapHeight
        canvas.create_oval(
            (data.world.y * yscale - AGENT_RADIUS) * MINIMAP_SCALE,
            (data.world.x * xscale - AGENT_RADIUS) * MINIMAP_SCALE,
            (data.world.y * yscale + AGENT_RADIUS) * MINIMAP_SCALE,
            (data.world.x * xscale + AGENT_RADIUS) * MINIMAP_SCALE,
            fill="#090",
        )

        # # Draw the rays
        # for rayx, rayy, _, _ in data.rays[:: NUM_RAYS // NUM_RAYS_DISPLAY]:
        #     canvas.create_line(
        #         data.arraygent.x * MINIMAP_SCALE,
        #         data.arraygent.y * MINIMAP_SCALE,
        #         rayx * MINIMAP_SCALE,
        #         rayy * MINIMAP_SCALE,
        #         fill="red",
        #     )


class AnimationData(object):
    def __init__(self, w, h, tkroot) -> None:
        self.width = w
        self.height = h
        self.tkroot = tkroot


def run(width, height):
    def redrawAllWrapper(canvas, data):
        # canvas.delete(ALL)
        # canvas.create_rectangle(0, 0, width, height, fill="white", width=0)
        redrawAll(canvas, data)
        # canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)

    def keyReleasedWrapper(event, canvas, data):
        keyReleased(event, data)

    def updateSceneWrapper(canvas, data):
        global inside, outside1, outside2

        outside2 = time.time()

        a = time.time()
        updateScene(data)
        b = time.time()
        redrawAllWrapper(canvas, data)
        c = time.time()

        total = c - a
        # print(f"Update: {b - a:1.04f} {(b - a) / total * 100:1.04f}%")
        # print(f"Draw  : {c - b:1.04f} {(c - b) / total * 100:1.04f}%")
        # print(f"total : {total:1.04f} {int(1/total)} FPS")
        # print(f"Actual FPS: {int(1/(time.time() - inside))}")
        # print(
        #     f"Time outside: {outside2 - outside1:1.04f} {int(1/(outside2 - outside1))} FPS"
        # )
        # print()
        inside = time.time()
        outside1 = time.time()

        data.tkroot.after(1, updateSceneWrapper, canvas, data)

    root = Tk()
    root.resizable(width=False, height=False)

    data = AnimationData(width, height, root)

    canvas = Canvas(root, width=width, height=height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()

    init(data, canvas)

    root.bind("<Button-1>", lambda event: mousePressedWrapper(event, canvas, data))
    root.bind("<KeyPress>", lambda event: keyPressedWrapper(event, canvas, data))
    root.bind("<KeyRelease>", lambda event: keyReleasedWrapper(event, canvas, data))

    # Position the window
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (width / 2)
    y = (hs / 2) - (height / 2)

    # set the dimensions of the screen and where it is placed
    root.geometry("%dx%d+%d+%d" % (width, height, x, y))

    # Kickoff event handling
    updateSceneWrapper(canvas, data)

    # Blocks until window is closed
    root.mainloop()


if __name__ == "__main__":
    run(WIDTH, HEIGHT)
