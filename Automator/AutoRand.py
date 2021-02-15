import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

# assuming running from raycasting-simulation/Automator
sys.path.append("../PycastWorld")

from math import acos, asin, atan, cos, sin, pi
from math import floor
from math import radians
from pycaster import RaycastWorld, Turn, Walk
from numpy.random import default_rng

rng = default_rng()


def getDir(dirX, dirY):
    """
    :param dirX: the x-coordinate of the direction vector of the camera
    :param dirY: the y-coordinate of the direction vector of the camera
    :rtype: float
    :return: direction the camera is facing (radians)
    """
    
    # fixing the fact that X,Y coordinates not always within [-1,1]
    if not -1 <= dirX <= 1:
        dirX = round(dirX)
    if not -1 <= dirY <= 1:
        dirY = round(dirY)

    if dirX > 0 and dirY >= 0:
        return acos(dirX)
    elif dirX <= 0 and dirY >= 0:
        return acos(dirX)
    elif dirX < 0 and dirY < 0:
        return pi - asin(dirY)
    elif dirX >= 0 and dirY < 0:
        return asin(dirY)


def get_rand_angle(curr_x, curr_y, targ_x, targ_y):
    """
    :param curr_x: the current x-coordinate of the camera
    :param curr_y: the current y-coordinate of the camera
    :param targ_x: the target x-coordinate of the camera
    :param targ_y: the target y-coordinate of the camera
    :rtype: float
    :return: an random angle in the direction of the target with added noise
    """

    # case where target pos is up and to the right
    if curr_x <= targ_x and curr_y <= targ_y:
        if targ_x == curr_x:
            theta = pi / 2
        else:
            theta = atan((targ_y - curr_y) / (targ_x - curr_x))

    # case where target pos is up and to the left
    elif curr_x > targ_x and curr_y <= targ_y:
        if targ_y == curr_y:
            theta = pi
        else:
            theta = atan((curr_x - targ_x) / (targ_y - curr_y)) + pi/2

    # case where target pos is down and to the left
    elif curr_x > targ_x and curr_y > targ_y:
        if targ_x == curr_x:
            theta = 3 * pi / 2
        else:
            theta = atan((curr_y - targ_y) / (curr_x - targ_x)) + pi
    
    # case where target pos is down and to the right
    else:
        if curr_y == targ_y:
            theta = 0
        else:
            theta = atan((targ_x - curr_x) / (curr_y - targ_y)) + 3 * pi / 2
    
    return rng.normal(loc=theta, scale=pi/12)


def l2_dist(curr_x, curr_y, targ_x, targ_y):
    """
    :param curr_x: the current x-coordinate of the camera
    :param curr_y: the current y-coordinate of the camera
    :param targ_x: the target x-coordinate
    :param targ_y: the target y-coordinate
    :rtype: float
    :return: The L2 distance between the current position and the target position
    """
    return math.sqrt((targ_x - curr_x) ** 2 + (targ_y - curr_y) ** 2)


def head_near_bound(curr_x, curr_y, targ_x, targ_y, angle, base_dir):
    """
    :param curr_x: the current x-coordinate of the camera
    :param curr_y: the current y-coordinate of the camera
    :param targ_x: the target x-coordinate
    :param targ_y: the target y-coordinate
    :param base_dir: the direction from the previous step (NESW)
    :rtype: boolean
    :return: True if camera is close to wall and moving towards wall
    """
    if base_dir == "Dir.WEST":
        if pi < angle < 3 * pi / 2 and (curr_y - targ_y) < 0.1:
            return True
        elif pi / 2 < angle < pi and (targ_y + 1 - curr_y) < 0.1:
            return True
        else:
            return False
    elif base_dir == "Dir.EAST":
        if (3 * pi / 2 < angle < 2 * pi or (- pi / 2) < angle < 0) and (curr_y - targ_y) < 0.1:
            return True
        elif 0 < angle < pi / 2 and (targ_y + 1 - curr_y) < 0.1:
            return True
        else:
            return False
    elif base_dir == "Dir.NORTH":
        if pi / 2 < angle < pi and (curr_x - targ_x) < 0.1:
            return True
        elif 0 < angle < pi / 2 and (targ_x + 1 - curr_x) < 0.1:
            return True
        else:
            return False
    elif base_dir == "Dir.SOUTH":
        if pi < angle < 3 * pi / 2 and (curr_x - targ_x) < 0.1:
            return True
        elif (3 * pi / 2 < angle < 2 * pi or (- pi / 2) < angle < 0) and (curr_x + 1 - targ_x) < 0.1:
            return True
        else:
            return False
    return False
    

def keep_straight(curr_x, curr_y, targ_x, targ_y, angle, base_dir, step):
    """
    :param curr_x: the current x-coordinate of the camera
    :param curr_y: the current y-coordinate of the camera
    :param targ_x: the (base) target x-coordinate
    :param targ_y: the (base) target y-coordinate
    :param step: the (specified) amount left to move towards target
    :rtype: boolean
    :return: True if camera within 0.1 of a side wall
    """
    dist = l2_dist(curr_x, curr_y, targ_x, targ_y)

    if dist < 0.1:
        return False
    elif head_near_bound(curr_x, curr_y, targ_x, targ_y, angle, base_dir):
        return False
    elif step < 0.2:
        return False
    else:
        return True

def get_better_targ(targ_x, targ_y, base_dir):
    """
    :param targ_x: the (base) target x-coordinate
    :param targ_y: the (base) target y-coordinate
    :param base_dir: the direction from the previous step (NESW)
    :rtype: tuple
    :return: (better x target, better y target)
    """
    if base_dir == "Dir.WEST":
        return targ_x + 0.25, targ_y + 0.5
    elif base_dir == "Dir.EAST":
        return targ_x + 0.75, targ_y + 0.5
    elif base_dir == "Dir.NORTH":
        return targ_x + 0.5, targ_y + 0.75
    elif base_dir == "Dir.SOUTH":
        return targ_x + 0.5, targ_y + 0.25
        


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "../Images/AutoRand"
    maze = sys.argv[2] if len(sys.argv) > 2 else "../Worlds/new_maze4.txt"

    world = RaycastWorld(320, 240, maze)
    # print(f"dirX: {acos(world.getDirX())}")

    # getting directions
    with open(maze, "r") as in_file:
        png_count = int(in_file.readline())
        for i in range(png_count):
            in_file.readline()

        _, dim_y = in_file.readline().split()
        for _ in range(int(dim_y)):
            in_file.readline()

        directions = in_file.readlines()

    # if 100 images in directory, start counting at 101
    img_num_l = len(os.listdir(f"{img_dir}/left")) + 1
    img_num_r = len(os.listdir(f"{img_dir}/right")) + 1
    img_num_s = len(os.listdir(f"{img_dir}/straight")) + 1

    i = 0
    while i < len(directions) - 1:
        _, _, base_dir = directions[i].split()
        targ_x, targ_y, targ_dir = directions[i + 1].split()
        targ_x, targ_y = int(targ_x), int(targ_y)
        curr_x, curr_y = world.getX(), world.getY()

        print(f"Directions: {targ_x}, {targ_y}, {targ_dir}")

        # random point in target cell to aim for
        # r_targ_x = rng.uniform(targ_x + .1, targ_x + .9)
        # r_targ_y = rng.uniform(targ_x + .1, targ_x + .9)
        
        # center of target cell
        c_targ_x = targ_x + .5
        c_targ_y = targ_y + .5

        # moving towards target
        abs_dist = l2_dist(curr_x, curr_y, c_targ_x, c_targ_y)
        while  abs_dist > 0.7:
            # choosing next angle direction
            rand_angle = get_rand_angle(curr_x, curr_y, c_targ_x, c_targ_y)

            # need this since getDir in range (-pi/2, 3pi/2)
            if rand_angle > 3 * pi / 2:
                rand_angle = rand_angle - 2 * pi
            
            world.walk(Walk.Stopped)

            # turning towards rand_angle
            curr_dir = getDir(world.getDirX(), world.getDirY())
            while abs(curr_dir - rand_angle) > .2:
                if curr_dir > rand_angle:
                    world.turn(Turn.Right)
                    world.update()

                    # save image right
                    world.savePNG(f"{img_dir}/right/{img_num_r:05}.png")
                    img_num_r += 1

                elif curr_dir < rand_angle:
                    world.turn(Turn.Left)
                    world.update()
                    
                    # save image left
                    world.savePNG(f"{img_dir}/left/{img_num_l:05}")
                    img_num_l += 1
                
                # image_data = np.array(world)
                # plt.imshow(image_data)
                # plt.show()
                
                curr_dir = getDir(world.getDirX(), world.getDirY())

            world.turn(Turn.Stop)

            # choosing how much to move in direction of random angle
            step = rng.uniform(0.3, abs_dist)

            # moving forward in direction of rand angle
            move_straight = keep_straight(curr_x, curr_y, c_targ_x, c_targ_y, curr_dir, base_dir, step)
            while move_straight:
                world.walk(Walk.Forward)
                world.update()

                # saving image straight
                world.savePNG(f"{img_dir}/straight/{img_num_s:05}.png")
                img_num_s += 1

                # image_data = np.array(world)
                # plt.imshow(image_data)
                # plt.show()

                step -= world.getWalkSpeed()
                curr_x, curr_y = world.getX(), world.getY()
                curr_dir = getDir(world.getDirX(), world.getDirY())

                move_straight = keep_straight(curr_x, curr_y, targ_x, targ_y, curr_dir, base_dir, step)

            abs_dist = l2_dist(curr_x, curr_y, c_targ_x, c_targ_y)

        i += 1


if __name__ == "__main__":
    main()
