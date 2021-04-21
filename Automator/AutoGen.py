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
from pycaster import RaycastWorld, Turn, Walk
from numpy.random import default_rng

rng = default_rng()

# NOISE CONTROL
# the standard deviation of the Gaussian that random angles are drawn from
rand_angle_scale = pi/36  # 5 degree s.d.

# the minimum of the uniform distribution that random distances (to move) are drawn from 
rand_step_scale = 0.3

enws = {"Dir.EAST": 0, "Dir.NORTH": 90, "Dir.WEST": 180, "Dir.SOUTH": 270}


class Driver:

    def __init__(self, c_targ_x, c_targ_y, base_dir, targ_dir, world, img_dir=None, show_freq=None):
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
            self.img_num_l = len(os.listdir(os.path.join(img_dir, 'left')))
            self.img_num_r = len(os.listdir(os.path.join(img_dir, 'right')))
            self.img_num_s = len(os.listdir(os.path.join(img_dir, 'straight')))

        self.show_freq = show_freq


    def update_dist(self):
        self.dist = math.sqrt((self.c_targ_x - self.world.getX()) ** 2 + (self.c_targ_y - self.world.getY()) ** 2)


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


    def get_angle(self):
        if self.curr_x <= self.c_targ_x and self.curr_y <= self.c_targ_y:
            if self.c_targ_x == self.curr_x:
                theta = pi / 2
            else:
                theta = (atan((self.c_targ_y - self.curr_y) / (self.c_targ_x - self.curr_x))) % (2*pi)

        # case where target pos is up and to the left
        elif self.curr_x > self.c_targ_x and self.curr_y <= self.c_targ_y:
            if self.c_targ_y == self.curr_y:
                theta = pi
            else:
                theta = (atan((self.curr_x - self.c_targ_x) / (self.c_targ_y - self.curr_y))) % (2*pi) + pi/2

        # case where target pos is down and to the left
        elif self.curr_x > self.c_targ_x and self.curr_y > self.c_targ_y:
            if self.c_targ_x == self.curr_x:
                theta = 3 * pi / 2
            else:
                theta = (atan((self.curr_y - self.c_targ_y) / (self.curr_x - self.c_targ_x))) % (2*pi) + pi
        
        # case where target pos is down and to the right
        else:
            if self.curr_y == self.c_targ_y:
                theta = 0
            else:
                theta = (atan((self.c_targ_x - self.curr_x) / (self.curr_y - self.c_targ_y))) % (2*pi) + 3 * pi / 2
            
        return theta            


    def set_rand_angle(self):
        theta = self.get_angle()
        self.angle =  rng.normal(loc=theta, scale=rand_angle_scale) % (2 * pi)

    
    def set_rand_step(self):
        self.step =  rng.uniform(rand_step_scale, self.dist_to_wall())

    
    def abs_angle_diff(self, angle):
        abs_diff = abs(self.direction - angle)
        return abs_diff % (2*pi)

    
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
        while self.abs_angle_diff(self.angle) > .1:
            if self.turn_right(self.angle):

                # save image right
                if self.img_dir != None:
                    self.world.savePNG(os.path.join(self.img_dir, 'right', f"{self.img_num_r:05}.png"))
                    self.img_num_r += 1

                self.world.turn(Turn.Right)
                self.world.update()

            else:

                # save image left
                if self.img_dir != None:
                    self.world.savePNG(os.path.join(self.img_dir, 'left', f"{self.img_num_l:05}.png"))
                    self.img_num_l += 1

                self.world.turn(Turn.Left)
                self.world.update()
            
            if self.show_freq != None:
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
            if (3*pi/2) <= self.direction <= (2*pi):
                a = self.world.getY() - (self.c_targ_y - 0.5)
            else:
                a = (self.c_targ_y + 0.5) - self.world.getY()
        elif self.targ_dir == 90:
            if 0 <= self.direction <= (pi/2):
                a = (self.c_targ_x + 0.5) - self.world.getX()
            else:
                a = self.world.getX() - (self.c_targ_x - 0.5)
        elif self.targ_dir == 180:
            if (pi/2) <= self.direction <= pi:
                a = (self.c_targ_y + 0.5) - self.world.getY()
            else:
                a = self.world.getY() - (self.c_targ_y - 0.5)
        elif self.targ_dir == 270:
            if pi <= self.direction <= 3*pi/2:
                a = self.world.getX() - (self.c_targ_x - 0.5)
            else:
                a = (self.c_targ_x + 0.5) - self.world.getX()

        b, c = self.solve_triangle(self.direction, a)

        if b < self.dist:
            return c
        else:
            return b
            

    def move_to_step(self):
        self.world.turn(Turn.Stop)
        i = 0
        while self.dist > 0.1 and self.step > 0.1:
            
            if self.img_dir != None:
                self.world.savePNG(os.path.join(self.img_dir, 'straight', f"{self.img_num_s:05}.png"))
                self.img_num_s += 1

            self.world.walk(Walk.Forward)
            self.world.update()

            if self.show_freq != None:
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
        self.world = RaycastWorld(320, 240, maze)
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
        

    def navigate(self, index, show_dir=False, show_freq=None):
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

        driver = Driver(c_targ_x, c_targ_y, base_dir, targ_dir, self.world, self.img_dir, show_freq)

        while driver.dist > 0.4:
            driver.set_rand_angle()
            driver.turn_to_angle()
            driver.set_rand_step()
            driver.move_to_step()


def main():
    maze = sys.argv[1] if len(sys.argv) > 1 else "../Worlds/new_maze.txt"
    img_dir = sys.argv[2] if len(sys.argv) > 2 else None

    navigator = Navigator(maze, img_dir)

    j = 0
    while j < navigator.num_directions - 1:
        navigator.navigate(j, show_dir=True, show_freq=1)


if __name__ == "__main__":
    main()
