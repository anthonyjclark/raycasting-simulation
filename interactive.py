#!/usr/bin/env python3

"""
Adapted from:
https://www.cs.cmu.edu/~rdriley/112/notes/notes-animations-part1.html
https://courses.pikuma.com/courses/raycasting
"""

from math import sin, cos, tan, pi
from tkinter import Tk, Canvas, ALL
from raycasting import RayWorld, Agent, Turn, Walk


WORLD = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

WALL_COLOR = "#222"
FLOOR_COLOR = "#fff"

NUM_ROWS = len(WORLD)
NUM_COLS = len(WORLD[0])
TILE_SIZE = 64
WIDTH = NUM_COLS * TILE_SIZE
HEIGHT = NUM_ROWS * TILE_SIZE

MINIMAP_SCALE_FACTOR = 0.2

AGENT_RADIUS = 3

FOV_DEG = 60
FOV_RAD = FOV_DEG * pi / 180

DIST_PROJ_PLANE = (WIDTH / 2) / tan(FOV_RAD / 2)

WALL_STRIP_WIDTH = 4
NUM_RAYS = WIDTH // WALL_STRIP_WIDTH
NUM_RAYS_DISPLAY = 5


def init(data):
    data.world = RayWorld(WORLD, TILE_SIZE)
    data.agent = Agent(WIDTH // 2, HEIGHT // 2, data.world)
    data.rays = []


def mousePressed(event, data):
    pass


def keyPressed(event, data):
    if event.keysym == "Up":
        data.agent.walk_direction = Walk.FORWARD
    elif event.keysym == "Down":
        data.agent.walk_direction = Walk.BACKWARD

    if event.keysym == "Right":
        data.agent.turn_direction = Turn.RIGHT
    elif event.keysym == "Left":
        data.agent.turn_direction = Turn.LEFT

    if event.char == "q":
        data.tkroot.quit()


def keyReleased(event, data):
    if event.keysym == "Up" or event.keysym == "Down":
        data.agent.walk_direction = Walk.NONE
    elif event.keysym == "Left" or event.keysym == "Right":
        data.agent.turn_direction = Turn.NONE


def updateScene(data):
    if data.agent.move():
        data.needs_drawn = True
        data.rays = data.world.get_raycast_view(
            data.agent.x, data.agent.y, data.agent.rotation_angle, FOV_RAD, NUM_RAYS
        )


def redrawAll(canvas, data):

    # Draw the walls
    for i, (_, _, rayd, wall) in enumerate(data.rays):
        wall_height = (TILE_SIZE / rayd) * DIST_PROJ_PLANE

        xstart = i * WALL_STRIP_WIDTH
        xend = (i + 1) * WALL_STRIP_WIDTH

        ystart = (HEIGHT - wall_height) / 2
        yend = ystart + wall_height

        brightness = 255 if wall == "vert" else 200
        cdist = brightness - int(brightness * rayd / WIDTH)
        color = "#" + f"{cdist:02x}" * 3

        canvas.create_rectangle(xstart, ystart, xend, yend, outline="", fill=color)

    # Draw the map
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):

            xstart = j * TILE_SIZE
            xend = (j + 1) * TILE_SIZE

            ystart = i * TILE_SIZE
            yend = (i + 1) * TILE_SIZE

            color = WALL_COLOR if data.world[i, j] == 1 else FLOOR_COLOR

            canvas.create_rectangle(
                xstart * MINIMAP_SCALE_FACTOR,
                ystart * MINIMAP_SCALE_FACTOR,
                xend * MINIMAP_SCALE_FACTOR,
                yend * MINIMAP_SCALE_FACTOR,
                fill=color,
            )

    # Draw the agent
    canvas.create_oval(
        (data.agent.x - AGENT_RADIUS) * MINIMAP_SCALE_FACTOR,
        (data.agent.y - AGENT_RADIUS) * MINIMAP_SCALE_FACTOR,
        (data.agent.x + AGENT_RADIUS) * MINIMAP_SCALE_FACTOR,
        (data.agent.y + AGENT_RADIUS) * MINIMAP_SCALE_FACTOR,
    )

    # Draw the rays
    for rayx, rayy, _, _ in data.rays[:: NUM_RAYS // NUM_RAYS_DISPLAY]:
        canvas.create_line(
            data.agent.x * MINIMAP_SCALE_FACTOR,
            data.agent.y * MINIMAP_SCALE_FACTOR,
            rayx * MINIMAP_SCALE_FACTOR,
            rayy * MINIMAP_SCALE_FACTOR,
            fill="red",
        )


# ---------------------------------------------------------
# The framework below does not need to be changed
# ---------------------------------------------------------


class AnimationData(object):
    def __init__(self, w, h, tkroot) -> None:
        self.width = w
        self.height = h
        self.tkroot = tkroot
        self.needs_drawn = True


def run(width, height):
    def redrawAllWrapper(canvas, data):
        data.needs_drawn = False
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, width, height, fill="white", width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        # redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        # redrawAllWrapper(canvas, data)

    def keyReleasedWrapper(event, canvas, data):
        keyReleased(event, data)
        # redrawAllWrapper(canvas, data)

    def updateSceneWrapper(canvas, data):
        updateScene(data)
        if data.needs_drawn:
            redrawAllWrapper(canvas, data)
        data.tkroot.after(20, updateSceneWrapper, canvas, data)

    root = Tk()
    root.resizable(width=False, height=False)

    data = AnimationData(width, height, root)
    init(data)

    canvas = Canvas(root, width=width, height=height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()

    root.bind("<Button-1>", lambda event: mousePressedWrapper(event, canvas, data))
    root.bind("<KeyPress>", lambda event: keyPressedWrapper(event, canvas, data))
    root.bind("<KeyRelease>", lambda event: keyReleasedWrapper(event, canvas, data))

    # Kickoff event handling
    updateSceneWrapper(canvas, data)

    # Blocks until window is closed
    root.mainloop()


if __name__ == "__main__":
    run(WIDTH, HEIGHT)
