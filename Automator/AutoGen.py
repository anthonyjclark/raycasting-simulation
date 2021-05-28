import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

# assuming running from raycasting-simulation/Automator
sys.path.append("../PycastWorld")

from math import acos, asin, atan, cos, sin, tan, pi
from math import floor
from math import radians
from pycaster import PycastWorld, Turn, Walk
from numpy.random import default_rng

rng = default_rng()

# NOISE CONTROL
# the standard deviation of the Gaussian that random angles are drawn from
rand_angle_scale = pi / 36  # 5 degree s.d.

# the minimum of the uniform distribution that random distances (to move) are drawn from
rand_step_scale = 0.4

enws = {"Dir.EAST": 0, "Dir.NORTH": 90, "Dir.WEST": 180, "Dir.SOUTH": 270}

def in_targ_cell(base_dir, c_targ_x, c_targ_y, x, y):
    if base_dir == 0 or base_dir == 180:
        if abs(c_targ_x - x) < 0.4:
            return True
    else:
        if abs(c_targ_y - y) < 0.4:
            return True
    return False

class Driver:
    def __init__(
        self,
        c_targ_x,
        c_targ_y,
        base_dir,
        targ_dir,
        world,
        img_dir=None,
        show_freq=0,
    ):
        self.c_targ_x = c_targ_x
        self.c_targ_y = c_targ_y
        self.base_dir = base_dir
        self.targ_dir = targ_dir

        self.world = world
        self.curr_x = self.world.getX()
        self.curr_y = self.world.getY()

        self.direction = 0
        self.update_direction()

        self.dist = math.inf
        self.update_dist()

        self.angle = 0
        self.step = math.inf

        self.img_dir = img_dir
        if self.img_dir != None:
            self.img_num_l = len(os.listdir(os.path.join(img_dir, "left")))
            self.img_num_r = len(os.listdir(os.path.join(img_dir, "right")))
            self.img_num_s = len(os.listdir(os.path.join(img_dir, "straight")))

        self.show_freq = show_freq

    def update_dist(self):
        self.dist = math.sqrt(
            (self.c_targ_x - self.world.getX()) ** 2
            + (self.c_targ_y - self.world.getY()) ** 2
        )

    def update_direction(self):
        if not -1 <= self.world.getDirX() <= 1:
            dir_x = round(self.world.getDirX())
        else:
            dir_x = self.world.getDirX()

        if not -1 <= self.world.getDirY() <= 1:
            dir_y = round(self.world.getDirY())
        else:
            dir_y = self.world.getDirY()

        if dir_x > 0 and dir_y >= 0:
            dir = acos(dir_x)
        elif dir_x <= 0 and dir_y >= 0:
            dir = acos(dir_x)
        elif dir_x < 0 and dir_y < 0:
            dir = pi - asin(dir_y)
        elif dir_x >= 0 and dir_y < 0:
            dir = asin(dir_y)

        self.direction = dir % (2 * pi)
    
    # adjust for smoother path
    def modified_targ(self, delta):
        if self.base_dir == 0 or self.base_dir == 180:
            if self.targ_dir == 90:
                return self.c_targ_x, self.c_targ_y + delta
            elif self.targ_dir == 270:
                return self.c_targ_x, self.c_targ_y - delta
        elif self.base_dir == 90 or self.base_dir == 270:
            if self.targ_dir == 0:
                return self.c_targ_x + delta, self.c_targ_y
            elif self.targ_dir == 180:
                return self.c_targ_x - delta, self.c_targ_y
        return self.c_targ_x, self.c_targ_y

    def get_angle(self):
        mod_x, mod_y = self.modified_targ(0.15)
        if self.curr_x <= mod_x and self.curr_y <= mod_y:
            if mod_x == self.curr_x:
                theta = pi / 2
            else:
                theta = (
                    atan((mod_y - self.curr_y) / (mod_x - self.curr_x))
                ) % (2 * pi)

        # case where target pos is up and to the left
        elif self.curr_x > mod_x and self.curr_y <= mod_y:
            if mod_y == self.curr_y:
                theta = pi
            else:
                theta = (
                    atan((self.curr_x - mod_x) / (mod_y - self.curr_y))
                ) % (2 * pi) + pi / 2

        # case where target pos is down and to the left
        elif self.curr_x > mod_x and self.curr_y > mod_y:
            if mod_x == self.curr_x:
                theta = 3 * pi / 2
            else:
                theta = (
                    atan((self.curr_y - mod_y) / (self.curr_x - mod_x))
                ) % (2 * pi) + pi

        # case where target pos is down and to the right
        else:
            if self.curr_y == mod_y:
                theta = 0
            else:
                theta = (
                    atan((mod_x - self.curr_x) / (self.curr_y - mod_y))
                ) % (2 * pi) + 3 * pi / 2

        return theta

    def set_rand_angle(self):
        theta = self.get_angle()
        self.angle = rng.normal(loc=theta, scale=rand_angle_scale) % (2 * pi)

    def set_rand_step(self):
        self.step = rng.uniform(rand_step_scale, self.dist_to_wall())

    def abs_angle_diff(self, angle):
        abs_diff = abs(self.direction - angle)
        return abs_diff % (2 * pi)

    def turn_right(self, angle):
        if self.direction > angle:
            if self.direction - angle > pi:
                return False
            else:
                return True
        else:
            if angle - self.direction > pi:
                return True
            else:
                return False

    def turn_to_angle(self):
        self.world.walk(Walk.Stopped)
        i = 0
        prev_turn = None
        while self.abs_angle_diff(self.angle) > 0.1:
            if self.turn_right(self.angle):
                
                if prev_turn == "left":
                    print("no left to right allowed")
                    break

                # save image right
                if self.img_dir != None:
                    self.world.savePNG(
                        os.path.join(self.img_dir, "right", f"{self.img_num_r:05}.png")
                    )
                    self.img_num_r += 1

                self.world.turn(Turn.Right)
                self.world.update()

                prev_turn = "right"

            else:
                if prev_turn == "right":
                    print("no right to left allowed")
                    break

                # save image left
                if self.img_dir != None:
                    self.world.savePNG(
                        os.path.join(self.img_dir, "left", f"{self.img_num_l:05}.png")
                    )
                    self.img_num_l += 1

                self.world.turn(Turn.Left)
                self.world.update()

                prev_turn = "left"

            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.update_direction()

        self.world.turn(Turn.Stop)

    @staticmethod
    def solve_triangle(theta, a):
        b = a * tan(theta)
        c = a / cos(theta)
        return b, c

    def dist_to_wall(self):
        if self.targ_dir == 0:
            if (3 * pi / 2) <= self.direction <= (2 * pi):
                a = self.world.getY() - (self.c_targ_y - 0.5)
                theta = self.direction - (3 * pi / 2)
            else:
                a = (self.c_targ_y + 0.5) - self.world.getY()
                theta = self.direction
        elif self.targ_dir == 90:
            if 0 <= self.direction <= (pi / 2):
                a = (self.c_targ_x + 0.5) - self.world.getX()
                theta = self.direction
            else:
                a = self.world.getX() - (self.c_targ_x - 0.5)
                theta = pi - self.direction
        elif self.targ_dir == 180:
            if (pi / 2) <= self.direction <= pi:
                a = (self.c_targ_y + 0.5) - self.world.getY()
                theta = self.direction - (pi / 2)
            else:
                a = self.world.getY() - (self.c_targ_y - 0.5)
                theta = (3 * pi / 2) - self.direction
        elif self.targ_dir == 270:
            if pi <= self.direction <= 3 * pi / 2:
                a = self.world.getX() - (self.c_targ_x - 0.5)
                theta = self.direction - pi
            else:
                a = (self.c_targ_x + 0.5) - self.world.getX()
                theta = (2 * pi) - self.direction

        b, c = self.solve_triangle(theta, a)

        if b < self.dist:
            return c
        else:
            return b

    def move_to_step(self):
        self.world.turn(Turn.Stop)
        i = 0
        while not in_targ_cell(self.base_dir, self.c_targ_x, self.c_targ_y, self.curr_x, self.curr_y) and self.step > 0.1:

            if self.img_dir != None:
                self.world.savePNG(
                    os.path.join(self.img_dir, "straight", f"{self.img_num_s:05}.png")
                )
                self.img_num_s += 1

            self.world.walk(Walk.Forward)
            self.world.update()

            self.curr_x = self.world.getX()
            self.curr_y = self.world.getY()

            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.step -= self.world.getWalkSpeed()
            self.update_dist()

        self.world.walk(Walk.Stopped)


class Navigator:
    def __init__(self, maze, img_dir=None):
        self.world = PycastWorld(320, 240, maze)
        self.img_dir = img_dir

        # getting directions
        with open(maze, "r") as in_file:
            png_count = int(in_file.readline())
            for _ in range(png_count):
                in_file.readline()

            _, dim_y = in_file.readline().split()
            for _ in range(int(dim_y)):
                in_file.readline()

            self.directions = in_file.readlines()

        self.num_directions = len(self.directions)

    def navigate(self, index, show_dir=False, show_freq=0):
        _, _, s_base_dir = self.directions[index].split()
        targ_x, targ_y, s_targ_dir = self.directions[index + 1].split()
        targ_x, targ_y = int(targ_x), int(targ_y)

        # convert from string
        base_dir = enws[s_base_dir]
        targ_dir = enws[s_targ_dir]

        if show_dir:
            print(f"Directions: {targ_x}, {targ_y}, {s_targ_dir}")

        # center of target cell
        c_targ_x = targ_x + 0.5
        c_targ_y = targ_y + 0.5

        driver = Driver(
            c_targ_x, c_targ_y, base_dir, targ_dir, self.world, self.img_dir, show_freq
        )

        while not in_targ_cell(base_dir, c_targ_x, c_targ_y, driver.curr_x, driver.curr_y):
            driver.set_rand_angle()
            driver.turn_to_angle()
            driver.set_rand_step()
            driver.move_to_step()


def main():
    maze = sys.argv[1] if len(sys.argv) > 1 else "../Worlds/maze.txt"
    show_freq = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # frequency to show frames
    img_dir = sys.argv[3] if len(sys.argv) > 3 else None  # directory to save images to

    navigator = Navigator(maze, img_dir)

    j = 0
    while j < navigator.num_directions - 1:
        navigator.navigate(j, show_dir=True, show_freq=show_freq)
        j += 1


if __name__ == "__main__":
    main()
