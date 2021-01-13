import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../PycastWorld")

from math import acos, asin, cos, sin, pi
from math import floor
from math import radians
from pycaster import RaycastWorld, Turn, Walk

# are we in target cell?
def pos_check(curr_x, curr_y, targ_x, targ_y, base_dir):
    if base_dir == "Dir.WEST":
        if curr_x - targ_x > 0.5:
            return True
        else:
            return False
    elif base_dir == "Dir.EAST":
        if curr_x - targ_x < 0.5:
            return True
        else:
            return False
    elif base_dir == "Dir.NORTH":
        if curr_y - targ_y < 0.5:
            return True
        else:
            return False
    elif base_dir == "Dir.SOUTH":
        if curr_y - targ_y > 0.5:
            return True
        else:
            return False


# which direction to go?
def turn_check(curr_dir, base_dir, targ_dir):
    if targ_dir == "Dir.WEST":
        if base_dir == "Dir.NORTH":
            if curr_dir < pi:
                return "left"

        elif base_dir == "Dir.SOUTH":
            if curr_dir > pi:
                return "right"

    elif targ_dir == "Dir.EAST":
        if base_dir == "Dir.NORTH":
            if curr_dir > 0:
                return "right"

        elif base_dir == "Dir.SOUTH":
            if curr_dir < 0:
                return "left"

    elif targ_dir == "Dir.NORTH":
        if base_dir == "Dir.WEST":
            if curr_dir > pi / 2:
                return "right"

        elif base_dir == "Dir.EAST":
            if curr_dir < pi / 2:
                return "left"

    elif targ_dir == "Dir.SOUTH":
        if base_dir == "Dir.WEST":
            if curr_dir < pi * 3 / 2:
                return "left"

        elif base_dir == "Dir.EAST":
            if curr_dir > -pi / 2:
                return "right"

    return "straight"


def getDir(dirX, dirY):
    # fix from RaycastWorld
    if not -1 <= dirX <= 1:
        dirX = round(dirX)
    if not -1 <= dirY <= 1:
        dirY = round(dirY)

    if dirX > 0 and dirY > 0:
        return acos(dirX)
    elif dirX < 0 and dirY > 0:
        return acos(dirX)
    elif dirX < 0 and dirY < 0:
        return pi - asin(dirY)
    elif dirX > 0 and dirY < 0:
        return asin(dirY)


def main():
    world = RaycastWorld(320, 240, "../Worlds/maze.txt")
    print(f"dirX: {acos(world.getDirX())}")

    # getting directions
    with open("../Worlds/maze.txt", "r") as in_file:
        png_count = int(in_file.readline())
        for i in range(png_count):
            in_file.readline()

        _, dim_y = in_file.readline().split()
        for _ in range(int(dim_y)):
            in_file.readline()

        directions = in_file.readlines()

    i = 0
    while i < len(directions) - 1:
        _, _, base_dir = directions[i].split()
        targ_x, targ_y, targ_dir = directions[i + 1].split()
        targ_x, targ_y = int(targ_x), int(targ_y)
        curr_x, curr_y = world.getX(), world.getY()

        print(targ_x, targ_y, base_dir)

        # moving forward
        while pos_check(curr_x, curr_y, targ_x, targ_y, base_dir):

            world.turn(Turn.Stop)
            world.walk(Walk.Forward)
            world.update()

            image_data = np.array(world)
            plt.imshow(image_data)
            plt.show()

            curr_x, curr_y = world.getX(), world.getY()

        curr_dir = getDir(world.getDirX(), world.getDirY())
        decide = turn_check(curr_dir, base_dir, targ_dir)

        # turning
        while decide != "straight":
            world.walk(Walk.Stopped)

            if decide == "right":
                world.turn(Turn.Right)
            elif decide == "left":
                world.turn(Turn.Left)

            world.update()

            image_data = np.array(world)
            plt.imshow(image_data)
            plt.show()

            curr_dir = getDir(world.getDirX(), world.getDirY())
            decide = turn_check(curr_dir, base_dir, targ_dir)

        i += 1


if __name__ == "__main__":
    main()
