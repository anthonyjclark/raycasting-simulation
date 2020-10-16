#!/usr/bin/env python3

from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TILE_SIZE = 32
MAP_NUM_ROWS = 11
MAP_NUM_COLS = 15

WINDOW_WIDTH = MAP_NUM_COLS * TILE_SIZE
WINDOW_HEIGHT = MAP_NUM_ROWS * TILE_SIZE


class Turn(IntEnum):
    RIGHT = -1
    NONE = 0
    LEFT = 1


class Walk(IntEnum):
    BACKWARD = -1
    NONE = 0
    FORWARD = 1


class GridMap(object):
    def __init__(self, ax) -> None:
        self.grid = [
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

        self.data = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        self.update()
        self.image = ax.imshow(self.data)

    def update(self) -> None:
        for i in range(MAP_NUM_ROWS):
            for j in range(MAP_NUM_COLS):
                xstart = j * TILE_SIZE
                xend = (j + 1) * TILE_SIZE
                ystart = i * TILE_SIZE
                yend = (i + 1) * TILE_SIZE
                tile_color = self.grid[i][j]
                self.data[ystart:yend, xstart:xend] = tile_color

    def artists(self):
        self.image.set_array(self.data)
        return self.image


class Player(object):
    def __init__(self, ax) -> None:
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.radius = 3
        self.turn_direction = Turn.NONE
        self.walk_direction = Walk.NONE
        self.rotation_angle = np.pi / 2
        self.move_speed = 5
        self.rotation_speed = 20 * (np.pi / 180)

        (self.line,) = ax.plot([self.x, self.x], [self.y, self.y + self.radius])

        self.circle = plt.Circle((self.x, self.y), self.radius, color="r")
        ax.add_artist(self.circle)

        self.updated = False

    def update(self) -> None:
        if self.turn_direction != Turn.NONE:
            self.rotation_angle += self.turn_direction * self.rotation_speed
            self.updated = True

        if self.walk_direction != Walk.NONE:
            step = self.walk_direction * self.move_speed
            newx = self.x + np.cos(self.rotation_angle) * step
            newy = self.y + np.sin(self.rotation_angle) * step
            self.x = newx
            self.y = newy
            self.updated = True

    def artists(self):
        self.circle.set_center((self.x, self.y))
        self.line.set_data(
            [self.x, self.x + self.radius * np.cos(self.rotation_angle)],
            [self.y, self.y + self.radius * np.sin(self.rotation_angle)],
        )
        self.updated = False
        return [self.line, self.circle]


class World(object):
    def __init__(self, ax):
        self.grid = GridMap(ax)
        self.player = Player(ax)

    def __call__(self, _):
        self.player.update()
        if self.player.updated:
            return self.player.artists()
        else:
            return []


def main() -> None:

    fig, ax = plt.subplots()

    ax.axis("equal")
    ax.set_xlim((0, WINDOW_WIDTH))
    ax.set_ylim((0, WINDOW_HEIGHT))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    def control_player(player: Player):
        def control(event):
            if event.key == "up":
                player.walk_direction = Walk.FORWARD
            elif event.key == "down":
                player.walk_direction = Walk.BACKWARD
            elif event.key == "right":
                player.turn_direction = Turn.RIGHT
            elif event.key == "left":
                player.turn_direction = Turn.LEFT

        return control

    def stop_player(player: Player):
        def control(event):
            if event.key == "up" or event.key == "down":
                player.walk_direction = Walk.NONE
            elif event.key == "left" or event.key == "right":
                player.turn_direction = Turn.NONE

        return control

    world = World(ax)

    fig.canvas.mpl_connect("key_press_event", control_player(world.player))
    fig.canvas.mpl_connect("key_release_event", stop_player(world.player))

    anim = FuncAnimation(fig, world, frames=100, interval=100)

    plt.show()


main()
