#!/usr/bin/env python3

from enum import IntEnum
import numpy as np
from typing import List, Tuple


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class RayWorld(object):
    def __init__(self, grid: List[List[int]], tile_size: int) -> None:
        self.grid = grid
        self.tile_size = tile_size
        self.width = len(grid[0]) * tile_size
        self.height = len(grid) * tile_size
        # self.data = np.zeros((height, width), dtype=np.uint8)

    def __getitem__(self, xy: Tuple[int, int]) -> int:
        x, y = xy
        return self.grid[x][y]

    def is_wall_at(self, x: int, y: int):
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True
        gridx = x // self.tile_size
        gridy = y // self.tile_size
        return self[gridy, gridx] == 1

    def get_raycast_view(self, x, y, forward, fov, num_rays) -> List:
        angle = forward - fov / 2
        rays = []
        for _ in range(num_rays):
            rays.append(self.cast_ray(x, y, angle, angle - forward))
            angle += fov / (num_rays - 1)
            angle %= 2 * np.pi
        return rays

    def cast_ray(self, x: int, y: int, angle: float, theta: float):
        facing_down = 0 < angle < np.pi
        facing_up = not facing_down
        facing_right = angle < np.pi / 2 or angle > 3 * np.pi / 2
        facing_left = not facing_right

        #
        # Search for horizontal walls
        #

        yintercept = (y // self.tile_size) * self.tile_size
        yintercept += self.tile_size if facing_down else 0

        xintercept = x + (yintercept - y) / np.tan(angle)

        dx = self.tile_size / np.tan(angle)
        dx *= -1 if facing_left and dx > 0 else 1
        dx *= -1 if facing_right and dx < 0 else 1

        dy = self.tile_size
        dy *= -1 if facing_up else 1

        # Shift one pixel so that the location is inside of a cell and not on the line
        yshift = 1 if facing_down else -1

        horz_hit_x = None
        horz_hit_y = None
        while 0 <= xintercept <= self.width and 0 <= yintercept <= self.height:
            if self.is_wall_at(int(xintercept), int(yintercept + yshift)):
                horz_hit_x = xintercept
                horz_hit_y = yintercept
                break
            xintercept += dx
            yintercept += dy

        #
        # Search for vertical walls
        #

        xintercept = (x // self.tile_size) * self.tile_size
        xintercept += self.tile_size if facing_right else 0

        yintercept = y + (xintercept - x) * np.tan(angle)

        dx = self.tile_size
        dx *= -1 if facing_left else 1

        dy = self.tile_size * np.tan(angle)
        dy *= -1 if facing_up and dy > 0 else 1
        dy *= -1 if facing_down and dy < 0 else 1

        # Shift one pixel so that the location is inside of a cell
        xshift = 1 if facing_right else -1

        vert_hit_x = None
        vert_hit_y = None
        while 0 <= xintercept <= self.width and 0 <= yintercept <= self.height:
            if self.is_wall_at(int(xintercept + xshift), int(yintercept)):
                vert_hit_x = xintercept
                vert_hit_y = yintercept
                break
            xintercept += dx
            yintercept += dy

        #
        # Compare distances to horizontal and vertical hits
        #
        horz_dist = dist(x, y, horz_hit_x, horz_hit_y) if horz_hit_x else np.inf
        vert_dist = dist(x, y, vert_hit_x, vert_hit_y) if vert_hit_x else np.inf

        if horz_dist < vert_dist:
            return horz_hit_x, horz_hit_y, horz_dist * np.cos(theta), "horz"
        else:
            return vert_hit_x, vert_hit_y, vert_dist * np.cos(theta), "vert"


class Turn(IntEnum):
    LEFT = -1
    NONE = 0
    RIGHT = 1


class Walk(IntEnum):
    BACKWARD = -1
    NONE = 0
    FORWARD = 1


class Agent(object):
    def __init__(self, x: int, y: int, world: RayWorld) -> None:
        self.x = x
        self.y = y
        self.rotation_angle = np.pi / 2
        self.move_speed = 10
        self.rotation_speed = 5 * (np.pi / 180)

        self.turn_direction = Turn.NONE
        self.walk_direction = Walk.NONE

        self.world = world

    def move(self) -> None:
        moved = False
        if self.turn_direction != Turn.NONE:
            self.rotation_angle += self.turn_direction * self.rotation_speed
            self.rotation_angle %= 2 * np.pi
            if self.rotation_angle < 0:
                self.rotation_angle = self.rotation_angle + (2 * np.pi)
            moved = True

        if self.walk_direction != Walk.NONE:
            step = self.walk_direction * self.move_speed
            newx = self.x + np.cos(self.rotation_angle) * step
            newy = self.y + np.sin(self.rotation_angle) * step
            if not self.world.is_wall_at(int(newx), int(newy)):
                self.x = newx
                self.y = newy
                moved = True

        return moved


def main() -> None:

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
    NUM_ROWS = len(WORLD)
    NUM_COLS = len(WORLD[0])
    TILE_SIZE = 64
    WIDTH = NUM_COLS * TILE_SIZE
    HEIGHT = NUM_ROWS * TILE_SIZE
    FOV_DEG = 60
    FOV_RAD = FOV_DEG * 3.145926 / 180
    WALL_STRIP_WIDTH = 4
    NUM_RAYS = WIDTH // WALL_STRIP_WIDTH

    world = RayWorld(WORLD, TILE_SIZE)
    agent = Agent(WIDTH // 2, HEIGHT // 2, world)

    from time import time

    start = time()
    num_trials = 100
    for _ in range(num_trials):
        world.get_raycast_view(
            agent.x, agent.y, agent.rotation_angle, FOV_RAD, NUM_RAYS
        )
    fps = 1 / ((time() - start) / num_trials)
    print(f"FPS: {fps}")


if __name__ == "__main__":
    main()
