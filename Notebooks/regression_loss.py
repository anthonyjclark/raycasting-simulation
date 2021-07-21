# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import distutils.util
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
        self, c_targ_x, c_targ_y, base_dir, targ_dir, world, img_dir=None, show_freq=0, abs_base_dir=0
        
    ):
        self.c_targ_x = c_targ_x
        self.c_targ_y = c_targ_y
        self.base_dir = base_dir
        self.targ_dir = targ_dir

        self.prev_targ_angle = 0

        self.world = world
        self.curr_x = self.world.x()
        self.curr_y = self.world.y()

        self.direction = 0
        self.update_direction()

        self.dist = math.inf
        self.update_dist()

        self.angle = 0
        self.step = math.inf

        self.all_angles = np.array([])
        self.all_base_angles = np.array([])
        
        self.abs_base_dir = abs_base_dir
        
        self.left_throttle = 0
        self.right_throttle = 0

        self.img_dir = img_dir
        if self.img_dir != None:
            stack_conds = []
            stack_conds.append(os.path.isdir(os.path.join(img_dir, "left")))
            stack_conds.append(os.path.isdir(os.path.join(img_dir, "right")))
            stack_conds.append(os.path.isdir(os.path.join(img_dir, "straight")))

            # if subdirectories exist, then stacking method not used
            if all(stack_conds):
                self.img_num_l = len(os.listdir(os.path.join(img_dir, "left")))
                self.img_num_r = len(os.listdir(os.path.join(img_dir, "right")))
                self.img_num_s = len(os.listdir(os.path.join(img_dir, "straight")))
                self.stack_dir = False
            else:
                self.img_num = len(os.listdir(img_dir))
                self.stack_dir = True

        self.show_freq = show_freq

    def update_dist(self):
        self.dist = math.sqrt(
            (self.c_targ_x - self.world.x()) ** 2
            + (self.c_targ_y - self.world.y()) ** 2
        )

    def update_direction(self):
        if not -1 <= self.world.get_dir_x() <= 1:
            dir_x = round(self.world.get_dir_x())
        else:
            dir_x = self.world.get_dir_x()

        if not -1 <= self.world.get_dir_y() <= 1:
            dir_y = round(self.world.get_dir_y())
        else:
            dir_y = self.world.get_dir_y()

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
                theta = (atan((mod_y - self.curr_y) / (mod_x - self.curr_x))) % (2 * pi)

        # case where target pos is up and to the left
        elif self.curr_x > mod_x and self.curr_y <= mod_y:
            if mod_y == self.curr_y:
                theta = pi
            else:
                theta = (atan((self.curr_x - mod_x) / (mod_y - self.curr_y))) % (2 * pi) + pi / 2

        # case where target pos is down and to the left
        elif self.curr_x > mod_x and self.curr_y > mod_y:
            if mod_x == self.curr_x:
                theta = 3 * pi / 2
            else:
                theta = (atan((self.curr_y - mod_y) / (self.curr_x - mod_x))) % (2 * pi) + pi

        # case where target pos is down and to the right
        else:
            if self.curr_y == mod_y:
                theta = 0
            else:
                theta = (atan((mod_x - self.curr_x) / (self.curr_y - mod_y))) % (2 * pi) + 3 * pi / 2
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
        self.world.walk(Walk.Stop)
        i = 0
        prev_turn = None
        turn_angle = 0.0
        angle_deg = self.angle * (180/pi)
        num_turns = 1

        while self.abs_angle_diff(self.angle) > 0.1:
            if self.turn_right(self.angle):
                if prev_turn == "left":
                    print("no left to right allowed")
                    break                
                
                agent_dir = turn_angle  
                angle_label = agent_dir
                if self.img_dir != None:
                    if self.stack_dir:
                        self.world.save_png(
                            os.path.join(
                                self.img_dir, f"{self.img_num:05}_{angle_label}.png",
                            )
                        )
                        self.img_num += 1
                    else:
                        self.world.save_png(
                            os.path.join(
                                self.img_dir,
                                "right",
                                f"{self.img_num_r:05}_{angle_label}.png",
                            )
                        )
                        self.img_num_r += 1

                self.world.turn(Turn.Right)
                turn_angle += 2.5  
                self.world.update()

                prev_turn = "right"

            else:
                if prev_turn == "right":
                    print("no right to left allowed")
                    break
                    
                agent_dir = turn_angle
                angle_label = agent_dir
                if self.img_dir != None:
                    if self.stack_dir:
                        self.world.save_png(
                            os.path.join(
                                self.img_dir, f"{self.img_num:05}_{angle_label}.png",
                            )
                        )
                        self.img_num += 1
                    else:
                        self.world.save_png(
                            os.path.join(
                                self.img_dir,
                                "left",
                                f"{self.img_num_l:05}_{angle_label}.png",
                            )
                        )
                        self.img_num_l += 1

                self.world.turn(Turn.Left)
                turn_angle += 2.5
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
                a = self.world.y() - (self.c_targ_y - 0.5)
                theta = self.direction - (3 * pi / 2)
            else:
                a = (self.c_targ_y + 0.5) - self.world.y()
                theta = self.direction
        elif self.targ_dir == 90:
            if 0 <= self.direction <= (pi / 2):
                a = (self.c_targ_x + 0.5) - self.world.x()
                theta = self.direction
            else:
                a = self.world.x() - (self.c_targ_x - 0.5)
                theta = pi - self.direction
        elif self.targ_dir == 180:
            if (pi / 2) <= self.direction <= pi:
                a = (self.c_targ_y + 0.5) - self.world.y()
                theta = self.direction - (pi / 2)
            else:
                a = self.world.y() - (self.c_targ_y - 0.5)
                theta = (3 * pi / 2) - self.direction
        elif self.targ_dir == 270:
            if pi <= self.direction <= 3 * pi / 2:
                a = self.world.x() - (self.c_targ_x - 0.5)
                theta = self.direction - pi
            else:
                a = (self.c_targ_x + 0.5) - self.world.x()
                theta = (2 * pi) - self.direction

        b, c = self.solve_triangle(theta, a)

        if b < self.dist:
            return c
        else:
            return b

    def move_to_step(self):
        self.world.turn(Turn.Stop)
        i = 0
        while (
            not in_targ_cell(
                self.base_dir, self.c_targ_x, self.c_targ_y, self.curr_x, self.curr_y
            )
            and self.step > 0.1
        ):
            angle_label = 0
            self.all_angles = np.append(self.all_angles, angle_label)
            if self.img_dir != None:
                if self.stack_dir:
                    self.world.save_png(
                        os.path.join(
                            self.img_dir, f"{self.img_num:05}_{angle_label}.png",
                        )
                    )
                    self.img_num += 1
                else:
                    self.world.save_png(
                        os.path.join(
                            self.img_dir,
                            "straight",
                            f"{self.img_num_s:05}_{angle_label}.png",
                        )
                    )
                    self.img_num_s += 1

            self.world.walk(Walk.Forward)
            self.world.update()

            self.curr_x = self.world.x()
            self.curr_y = self.world.y()

            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.step -= self.world.walk_speed()
            self.update_dist()

        self.world.walk(Walk.Stop)


class Navigator:
    def __init__(self, maze, img_dir=None):
        self.world = PycastWorld(320, 240, maze)
        self.img_dir = img_dir
        self.abs_base_dir = 0
        self.abd_targ_dir = 0

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

        self.angles = np.array([])
        self.base_angles = np.array([])
        self.dirs = []

    def navigate(self, index, show_dir=False, show_freq=0):
        _, _, s_base_dir = self.directions[index].split()
        targ_x, targ_y, s_targ_dir = self.directions[index + 1].split()
        targ_x, targ_y = int(targ_x), int(targ_y)
        

        # convert from string
        base_dir = enws[s_base_dir]
        targ_dir = enws[s_targ_dir]        
        
        if base_dir != targ_dir :
            self.abs_base_dir = base_dir
            self.abs_targ_dir = targ_dir
        if self.abs_base_dir == 0:
            self.abs_base_dir = 360
            
        _, _, s_prev_dir = None, None, None
        prev_dir = None
        if index >= 1:            
            s_prev_dir = s_targ_dir
            prev_dir = enws[s_prev_dir]
            if base_dir != prev_dir:
                self.abs_base_dir = prev_dir
        # we know prev direction; calculate the angle direction we need and 
        # keep that angle direction until the current direction changes again. 
        # e.g. West -> North means turn right until North -> East so on and so forth
        if show_dir:
            print(f"Directions: {targ_x}, {targ_y}, {s_targ_dir}, basedir: {s_base_dir}, \
            abs base dir: {self.abs_base_dir}\
            prev dir: {s_prev_dir}")

        # center of target cell
        c_targ_x = targ_x + 0.5
        c_targ_y = targ_y + 0.5

        driver = Driver(
            c_targ_x, c_targ_y, base_dir, targ_dir, self.world, self.img_dir, show_freq, self.abs_base_dir
        )

        while not in_targ_cell(
            base_dir, c_targ_x, c_targ_y, driver.curr_x, driver.curr_y
        ):
            driver.set_rand_angle()
            driver.turn_to_angle()
            driver.set_rand_step()
            driver.move_to_step()
        
#         print("made it to targ cell..moving to next")
        self.angles = np.append(self.angles, driver.all_angles)
        return prev_dir

    def plot_angles(self):
        plt.plot(self.angles)
        plt.show()

    def plot_directions(self):
        plt.plot(self.dirs)
        plt.show()

    def plot_label_dir(self):
        plt.plot(self.directions)
        plt.show()

# +
mazes = ["../Mazes/15_mazes_test_07-07-2021_16-07/maze_1.txt",
         "../Mazes/15_mazes_test_07-07-2021_16-07/maze_2.txt", 
         "../Mazes/15_mazes_test_07-07-2021_16-07/maze_3.txt",
         "../Mazes/15_mazes_test_07-07-2021_16-07/maze_4.txt"]
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_5.txt"]
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_6.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_7.txt", 
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_8.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_9.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_10.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_11.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_12.txt", 
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_13.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_14.txt",
#          "../Mazes/15_mazes_test_07-07-2021_16-07/maze_15.txt"]

for m in mazes:
    maze = m
    print(maze)
    show_freq = 0  # frequency to show frames
    img_dir = "/raid/Images/test" # directory to save images to
    show_dir = True

    navigator = Navigator(maze, img_dir)

    j = 0
    while j < navigator.num_directions - 1:
        navigator.navigate(j, show_dir=show_dir, show_freq=show_freq)
        j += 1
# -

plt.plot(navigator.angles)
plt.show()

from fastai.vision.all import *
from fastai import *
import datetime;
from fastai.vision.widgets import *
import torch 
from math import pi

# +
path = Path('/raid/Images/test')

# num_img = !ls -l '/raid/Images/test' | wc -l

int(num_img[0])
# -

get_image_files(path)

r_path = Path('/raid/Images/test/right')
l_path = Path('/raid/Images/test/left')

torch.cuda.current_device()

torch.cuda.set_device(1)

torch.cuda.current_device()

sample_lab = "02905_0_straight.png"
sample_lab.split('_')[2][:-4]
tensor((1,12), (23,155))

img_files = get_image_files(path)

im = PILImage.create(img_files[0])
im.shape

im.to_thumb(160)


def get_deg(f):
    sample_lab = f.name
    deg = float(sample_lab.split('_')[1][:-4])
    return tensor(deg).unsqueeze(0)


s = "04504_-72.5.png"
split_name = s.split('_')
angle = float(split_name[1][:-4])
if angle < 0:
    throttle = (0,-angle)
elif angle > 0:
    throtttle = (angle,0)
else:
    throttle = (2.5, 2.5)
throttle


def get_throttles(f):
    split_name = f.name.split('_')
    angle = float(split_name[1][:-4])
    if angle < 0:
        return tensor([2.5, -2.5])#torch.stack((tensor(0.),tensor(-angle)))
    elif angle > 0:
        return tensor([-2.5, 2.5])#torch.stack((tensor(angle),tensor(0.)))
    else:
        return tensor([2.5, 2.5])#torch.stack((tensor(2.5),tensor(2.5)))


db_r = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_items=get_image_files,
    get_y=get_throttles,
    splitter=RandomSplitter(valid_pct=0.2, seed=47),
)

dls_r = db_r.dataloaders(path)
dls_r.show_batch(max_n=9, figsize=(8,6))

xb,yb = dls_r.one_batch()
xb.shape,yb.shape

torch.where(torch.abs((y[:, 1] - y[:, 0]) - (yhat[:, 1] - yhat[:, 0])) < 0.1, 1., 0.).mean()

yb[0]


def angle_loss(preds, targs):
    pred_angle_list = [e[0] for e in preds]
    pred_turn_list = [e[1] for e in preds]
    
    targs_angle_list = [e[0] for e in targs]
    targs_turn_list = [e[1] for e in targs]
    
    angle_preds = torch.stack(pred_angle_list)
    turn_preds = torch.stack(pred_turn_list)
    
    angle_targs = torch.stack(targs_angle_list)
    turn_targs = torch.stack(targs_turn_list)
    return ((angle_preds - angle_targs)**2 + ((turn_preds - turn_targs)**2)).mean()


def steering_loss(preds, targs):
    angle_true = targs[:, 1] - targs[:, 0]
    angle_pred = preds[:, 1] - preds[:, 0]
    weight = torch.abs(angle_true + 0.05)
    return (
        torch.square(weight)
        * (torch.nn.functional.mse_loss(angle_true, angle_pred, reduction="none").T)
    ).mean() + torch.nn.functional.mse_loss(targs, preds)


num_train_examples = 10
num_output_neurons = 2
yhat = torch.vstack(
    [
        torch.linspace(1, 10, num_train_examples),
        torch.linspace(1, 10, num_train_examples),
    ]
).T
y = torch.ones((num_train_examples, num_output_neurons))
steering_loss(yhat, y)

y,yhat

yhat.shape

torch.nn.functional.mse_loss(y, yhat, reduction="none")

torch.nn.functional.mse_loss(y, yhat)

angle_true = y[:, 1] - y[:, 0]
angle_pred = yhat[:, 1] - yhat[:, 0]

(torch.square(torch.abs((y[:,1] - yhat[:,0]) + 0.05)) * 
 torch.nn.functional.mse_loss(angle_true, angle_pred, reduction='none').T).mean() + torch.nn.functional.mse_loss(y, yhat)

torch.nn.functional.mse_loss(y, yhat, reduction='none').T


def angle_metric(preds, targs):
    angle_true = targs[:, 1] - targs[:, 0]
    angle_pred = preds[:, 1] - preds[:, 0]
    return torch.where(torch.abs(angle_true - angle_pred) < 0.1, 1., 0.).mean()


def direction_metric(preds, targs):
    angle_true = targs[:, 1] - targs[:, 0]
    angle_pred = preds[:, 1] - preds[:, 0]
    return torch.where(
        torch.logical_or(
            torch.sign(angle_pred) == torch.sign(angle_true),
            torch.abs(angle_pred) < 0.1,
        ),
        1.0,
        0.0,
    ).mean()


learn = cnn_learner(
    dls_r,
    resnet18,
    y_range=(-100, 100),
    metrics=[mse, angle_metric, direction_metric],
)
learn.fine_tune(
    20,
    cbs=[SaveModelCallback(), EarlyStoppingCallback(monitor="valid_loss", patience=5)],
)

learn2 = cnn_learner(
    dls_r,
    resnet18,
    y_range=(-100, 100),
    loss_func=steering_loss,
    metrics=[mse, angle_metric, direction_metric],
)
learn2.fine_tune(
    20,
    cbs=[SaveModelCallback(), EarlyStoppingCallback(monitor="valid_loss", patience=5)],
)

learn2.loss_func

0-30

learn.export('/home/CAMPUS/eoca2018/raycasting-simulation/Models/point_block_reg.pkl')

learn.show_results(ds_idx=1, nrows=3, figsize=(15,15))

learn.predict(get_image_files(path)[0])

get_image_files(path)[0]

# # Try Predicting with the Regression Model

import matplotlib.pyplot as plt
import sys
sys.path.append("../PycastWorld")
sys.path.append("../Gym")
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# +
steps_per_episode = 2000

env = PycastWorldEnv("../Mazes/maze01-long.txt", 320, 240)

path = Path('/home/CAMPUS/eoca2018/raycasting-simulation/Models/steering_loss.pkl')
# Run some number of trials all starting from the
# initial location. We might eventually randomize
# the maze and the starting location.

# Grab the initial observation (not used here)
observation = env.reset()
frames = [observation.copy()]
model_inf = load_learner(path)
prev_pred = 0

plt.imshow(observation)
# -



# +
print("Predicting...")
for t in range(steps_per_episode):    
    pred_angle, _, _ =  model_inf.predict(observation)
    pred_angle = math.ceil(pred_angle[0])
    print(pred_angle)
    num_movements = abs(int(pred_angle / 20))
    
    if num_movements == 0:
        action_index = 1
        observation, reward, done, info = env.step(action_index)
        frames.append(observation.copy())
        prev_pred = pred_angle
        
        if done:
            print(f"  Found goal in {t+1} steps")
            break
        continue
    
    if (prev_pred > 0 and pred_angle < 0) or (prev_pred < 0 and pred_angle > 0):
        print("left-right mixup")
        action_index = 1
        observation, reward, done, info = env.step(action_index)
        frames.append(observation.copy())
        prev_pred = pred_angle        
        if done:
            print(f"  Found goal in {t+1} steps")
            break        
        continue
        
    # check if we have to move opposite
#     if abs(prev_pred) > abs(pred_angle) + 5:
#         print("preventing overturn")
#         action_index = 1
#         observation, reward, done, info = env.step(action_index)
#         frames.append(observation.copy())
#         continue
             
    action_index = 1
    if pred_angle > 0 and num_movements > 0:
        for i in range(num_movements):
            action_index = 0 # turn left
            observation, reward, done, info = env.step(action_index)
            frames.append(observation.copy())
    elif pred_angle < 0 and num_movements > 0:
        for i in range(num_movements):
            action_index = 2 # turn right
            observation, reward, done, info = env.step(action_index)
            frames.append(observation.copy())

    prev_pred = pred_angle
    # Check if we reached the end goal
    if done:
        print(f"  Found goal in {t+1} steps")
        break

print(f"  Ended at position {env.world.x()}, {env.world.y()}")
env.close();

# +
fig, ax = plt.subplots()
ln = plt.imshow(frames[0])
def init():
    ln.set_data(frames[0])
    return [ln]

def update(frame):
    ln.set_array(frame)
    return [ln] 

ani = FuncAnimation(fig, update, frames, init_func=init, interval=60)
smaller_frames = frames[::3] 
ani = FuncAnimation(fig, update, smaller_frames, init_func=init, interval=60)
HTML(ani.to_html5_video())
# -
# # Loss vs Non-Loss Comparison

path = Path('/home/CAMPUS/eoca2018/raycasting-simulation/Models/proxy_regression_model_loss.pkl')
path2 = Path('/home/CAMPUS/eoca2018/raycasting-simulation/Models/proxy_regression_model_norm.pkl')


model_inf_loss = load_learner(path)
model_inf = load_learner(path2)


