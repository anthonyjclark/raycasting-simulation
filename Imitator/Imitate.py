#!/usr/bin/env python

from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import distutils.util

from fastai.vision.all import *
from math import radians

# Needed to import pycaster from relative path
import sys

sys.path.append("../PycastWorld")
sys.path.append("../Models")
from pycaster import PycastWorld, Turn, Walk

# Needed for Timeout 
import signal

# Needed to calculate maze percentage
# import sys
sys.path.append("../Utilities")
from MazeUtils import read_maze_file, percent_through_maze, bfs_dist_maze, is_on_path

# for animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time


# functions defined for model required by fastai
def parent_to_deg(f):
    parent = parent_label(f)
    if parent == "left":
        return 90.0
    elif parent == "right":
        return -90.0
    else:
        return 0.0


def sin_cos_loss(preds, targs):
    rad_targs = targs / 180 * np.pi
    x_targs = torch.cos(rad_targs)
    y_targs = torch.sin(rad_targs)
    x_preds = preds[:, 0]
    y_preds = preds[:, 1]
    return ((x_preds - x_targs) ** 2 + (y_preds - y_targs) ** 2).mean()


def within_angle(preds, targs, angle):
    rad_targs = targs / 180 * np.pi
    angle_pred = torch.atan2(preds[:, 1], preds[:, 0])
    abs_diff = torch.abs(rad_targs - angle_pred)
    angle_diff = torch.where(abs_diff > np.pi, np.pi * 2.0 - abs_diff, abs_diff)
    return torch.where(angle_diff < angle, 1.0, 0.0).mean()


def within_45_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 4)


def within_30_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 6)


def within_15_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 12)


def name_to_deg(f):
    label = f.name[6:-4]
    if label == "left":
        return 90.0
    elif label == "right":
        return -90.0
    else:
        return 0.0


def get_label(o):
    return o.name[6:-4]


def get_pair_2(o):
    curr_im_num = Path(o).name[:5]
    if not int(curr_im_num):
        prev_im_num = curr_im_num
    else:
        prev_im_num = int(curr_im_num) - 1

    prev_im = None
    for item in Path(o).parent.ls():
        if int(item.name[:5]) == prev_im_num:
            prev_im = item
    if prev_im is None:
        prev_im = Path(o)
    assert prev_im != None

    img1 = Image.open(o).convert("RGB")
    img2 = Image.open(prev_im).convert("RGB")
    img1_arr = np.array(img1, dtype=np.uint8)
    img2_arr = np.array(img2, dtype=np.uint8)

    new_shape = list(img1_arr.shape)
    new_shape[-1] = new_shape[-1] * 2
    img3_arr = np.zeros(new_shape, dtype=np.uint8)

    img3_arr[:, :, :3] = img1_arr
    img3_arr[:, :, 3:] = img2_arr

    return img3_arr.T.astype(np.float32)


# helper functions
def stacked_input(prev_im, curr_im):
    if prev_im is None:
        prev_im = curr_im

    new_shape = list(curr_im.shape)
    new_shape[-1] = new_shape[-1] * 2
    stacked_im = np.zeros(new_shape, dtype=np.uint8)

    stacked_im[:, :, :3] = curr_im
    stacked_im[:, :, 3:] = prev_im

    return stacked_im.T.astype(np.float32)


def reg_predict(pred_coords):
#     print(f"type: {type(pred_coords[1])} ")
#     print(f"pred_coord[1]: {pred_coords} ")
    pred_angle = np.arctan2(pred_coords[1], pred_coords[0]) / np.pi * 180
    pred_angle = pred_angle % (360)

    if pred_angle > 53 and pred_angle <= 180:
        return "left"
    elif pred_angle > 180 and pred_angle < 307:
        return "right"
    else:
        return "straight"

# Animation function.TODO: make it output an embedded HTML figure 
def animate(image_frames, name, dir_name):    
    """
    Generate a GIF animation of the saved frames
    
    Keyword arguments:
    image_frames -- array of frames    
    name -- name of model
    """
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    os.mkdir(dir_name)
    os.system(dir_name)
    save_path = os.path.abspath(dir_name)
    name = str(name).split('/')[-1][:-4]
    fig, ax = plt.subplots()
    ln = plt.imshow(image_frames[0])
    def init():
        ln.set_data(image_frames[0])
        return [ln]

    def update(frame):
    #     print(frame)
        ln.set_array(frame)
        return [ln]

    ani = FuncAnimation(fig, update, image_frames, init_func=init, interval=100)
    # plt.show()
    ani.save(os.path.join(save_path, name + "_.mp4"))
    
def main(argv):
    maze = argv[0] if len(argv) > 0 else "../Mazes/maze01.txt"
    model = argv[1] if len(argv) > 1 else "../Models/auto-gen-c.pkl"
    show_freq = int(argv[2]) if len(argv) > 2 else 0  # frequency to show frames
    directory_name = argv[5] if len(argv) > 5 else "tmp_diagnostics"
    print("DIR NAME: " + directory_name)

    model_type = (
        argv[3] if len(argv) > 3 else "c"
    )  # 'c' for classification, 'r' for regresssion
    stacked = (
        bool(distutils.util.strtobool(argv[4])) if len(argv) > 4 else False
    )  # True for stacked input

    world = PycastWorld(320, 240, maze)

    path = Path("../")
    model_inf = load_learner(model)
    prev_move = None
    prev_image_data = None
    frame = 0
    num_static = 0
    prev_x, prev_y = world.getX(), world.getY()
    animation_frames = []
    
    outcome = "At goal? "
    stuck = False
    # Initialize maximum number of steps in case the robot travels in a completely incorrect direction
    max_steps = 3000 
    step_count = 0
    
    # Initialize Maze Check
    maze_rvs, _, _, maze_directions, _ = read_maze_file(maze)
    start_x, start_y, _ = maze_directions[0]
    end_x, end_y, _ = maze_directions[-1]
    _, maze_path = bfs_dist_maze(maze_rvs, start_x, start_y, end_x, end_y)
    
    while not world.atGoal() and num_static < 5:
        
        if is_on_path(maze_path, int(world.getX()), int(world.getY())) is False:
            print("Off Path")
            break

        # Get image
        image_data = np.array(world)

        # Convert image_data and give to network
        if model_type == "c":
            if stacked:
                move = model_inf.predict(stacked_input(prev_image_data, image_data))[0]
            else:
                move = model_inf.predict(image_data)[0]
        elif model_type == "r":
            if stacked:
                pred_coords, _, _ = model_inf.predict(
                    stacked_input(prev_image_data, image_data)
                )
            else:
                pred_coords, _, _ = model_inf.predict(image_data)
            move = reg_predict(pred_coords)

        # print(move)

        if move == "left" and prev_move == "right":
            move = "straight"
        elif move == "right" and prev_move == "left":
            move = "straight"

        # Move in world
        if move == "straight":
            world.walk(Walk.Forward)
            world.turn(Turn.Stop)
        elif move == "left":
            world.walk(Walk.Stopped)
            world.turn(Turn.Left)
        else:
            world.walk(Walk.Stopped)
            world.turn(Turn.Right)

        prev_move = move
        world.update()

        curr_x, curr_y = round(world.getX(), 5), round(world.getY(), 5)

        if show_freq != 0 and frame % show_freq == 0:
            if curr_x == prev_x and curr_y == prev_y:
                num_static += 1
            else:
                num_static = 0
            animation_frames.append(image_data.copy())
#             plt.imshow(image_data)
#             plt.show()
            # update previous coordinates
            prev_x = curr_x
            prev_y = curr_y        

        frame += 1
        prev_image_data = image_data
        if frame == max_steps:
            print("Exceeds step limit")
            break
            
    # this chunk gets the completion percentage
    lost = False
    if num_static >= 5:
        stuck = True
    if frame >= max_steps:
        lost = True
    outcome = "At Goal? " + str(world.atGoal()) + "\n Stuck? " + str(stuck) + "\n Exceed step limit? " + str(lost)
    print(outcome)

    completion_per = percent_through_maze(maze_rvs, int(world.getX()), int(world.getY()), start_x, start_y, end_x, end_y)

    plt.imshow(image_data)
    plt.show()
    print("DIR NAME: ")
 
    animate(animation_frames, model, directory_name)

    if num_static >= 5 and not world.atGoal():  # model failed to navigate maze
        return frame, False, completion_per
    else:  # model successfully navigated maze
        return frame, True, completion_per


if __name__ == "__main__":
    main(sys.argv[1:])
