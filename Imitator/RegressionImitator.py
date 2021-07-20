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
sys.path.append("../Gym")
from pycaster import PycastWorld, Turn, Walk
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv
# Needed for Timeout
import signal
# Needed to calculate maze percentage
sys.path.append("../MazeGen")
from MazeUtils import read_maze_file, percent_through_maze, bfs_dist_maze, is_on_path
# for animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time


def get_deg(f):
    sample_lab = f.name
    deg = float(sample_lab.split('_')[1][:-4])
    return tensor(deg).unsqueeze(0)

def get_turn_label(f):
    sample_lab = f.name
    split_name = sample_lab.split('_')
    deg = float(split_name[1])
    action = 1
    label = split_name[2][:-4]
    if label == 'right':
        action = 2
    elif label == 'left':
        action = 0
    return torch.stack((tensor(deg), tensor(action)))

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

# Animation function.TODO: make it output an embedded HTML figure
def animate(image_frames, name, dir_name):
    """
    Generate a GIF animation of the saved frames

    Keyword arguments:
    image_frames -- array of frames
    name -- name of model
    dir_name -- name of directory
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    else: 
        os.system(dir_name)
    save_path = os.path.abspath(dir_name)
    name = str(name).split("/")[-1][:-4]
    fig, ax = plt.subplots()
    ln = plt.imshow(image_frames[0])

    def init():
        ln.set_data(image_frames[0])
        return [ln]

    def update(frame):
        #     print(frame)
        ln.set_array(frame)
        return [ln]

    ani = FuncAnimation(fig, update, image_frames, init_func=init)
    ani.save(os.path.join(save_path, name + "_" + str(now) + ".mp4"))


def main(argv):
    maze = argv[0] if len(argv) > 0 else "../Mazes/maze01.txt"
    model = argv[1] if len(argv) > 1 else "../Models/auto-gen-c.pkl"
    show_freq = int(argv[2]) if len(argv) > 2 else 0  # frequency to show frames
    directory_name = argv[5] if len(argv) > 5 else "tmp_diagnostics"
    print("DIR NAME: " + directory_name)

    env = PycastWorldEnv(maze, 320, 240)

    path = Path("../")
    observation = env.reset()
    model_inf = load_learner(model)
    prev_move = None
    prev_image_data = None
    frame = 0
    frame_freq = 5
    num_static = 0
    prev_x, prev_y = env.world.x(), env.world.y()
    animation_frames = []
    prev_pred = 0

    outcome = "At goal? "
    stuck = False
    # Initialize maximum number of steps in case the robot travels in a completely incorrect direction
    max_steps = 3500

    # Initialize Maze Check
    maze_rvs, _, _, maze_directions, _ = read_maze_file(maze)
    start_x, start_y, _ = maze_directions[0]
    end_x, end_y, _ = maze_directions[-1]
    _, maze_path = bfs_dist_maze(maze_rvs, start_x, start_y, end_x, end_y)

    while not env.world.at_goal() and num_static < 5:

        if is_on_path(maze_path, int(env.world.x()), int(env.world.y())) is False:
            print("Off Path")
            break

        # Get image
        image_data = np.array(env.world)

        # Convert image_data and give to network
        pred_angle, _, _ =  model_inf.predict(observation)
        pred_angle = math.ceil(pred_angle[0])
        num_movements = abs(int(pred_angle / 20))
        
        if num_movements == 0:
            action_index = 1
            observation, reward, done, info = env.step(action_index)
            curr_x, curr_y = round(env.world.x(), 5), round(env.world.y(), 5)
            prev_pred = pred_angle
            
            if show_freq != 0 and frame % show_freq == 0:
                if curr_x == prev_x and curr_y == prev_y:
                    num_static += 1
                else:
                    num_static = 0
                animation_frames.append(image_data.copy())
                prev_x = curr_x
                prev_y = curr_y
            frame += 1
            prev_image_data = image_data

            if frame == max_steps:
                print("Exceeds step limit")
                break
            continue
        
        if (prev_pred > 0 and pred_angle < 0) or (prev_pred < 0 and pred_angle > 0):
            action_index = 1
            observation, reward, done, info = env.step(action_index)
            curr_x, curr_y = round(env.world.x(), 5), round(env.world.y(), 5)
            prev_pred = pred_angle
            
            if show_freq != 0 and frame % show_freq == 0:
                if curr_x == prev_x and curr_y == prev_y:
                    num_static += 1
                else:
                    num_static = 0
                animation_frames.append(image_data.copy())
                prev_x = curr_x
                prev_y = curr_y
            frame += 1
            prev_image_data = image_data

            if frame == max_steps:
                print("Exceeds step limit")
                break
            continue

        action_index = 1
        if pred_angle > 0 and num_movements > 0:
            for i in range(num_movements):
                action_index = 0 # turn left
                observation, reward, done, info = env.step(action_index)
        elif pred_angle < 0 and num_movements > 0:
            for i in range(num_movements):
                action_index = 2 # turn right
                observation, reward, done, info = env.step(action_index)

        prev_pred = pred_angle
        env.world.update()
        curr_x, curr_y = round(env.world.x(), 5), round(env.world.y(), 5)

        if show_freq != 0 and frame % show_freq == 0:
            if curr_x == prev_x and curr_y == prev_y:
                num_static += 1
            else:
                num_static = 0            
            prev_x = curr_x
            prev_y = curr_y
        if frame % frame_freq == 0:
            animation_frames.append(image_data.copy())
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
    outcome = (
        "At Goal? "
        + str(env.world.at_goal())
        + "\n Stuck? "
        + str(stuck)
        + "\n Exceed step limit? "
        + str(lost)
    )
    print(outcome)

    completion_per = percent_through_maze(
        maze_rvs, int(env.world.x()), int(env.world.y()), start_x, start_y, end_x, end_y
    )

#     plt.imshow(image_data)
#     plt.show()
#     print("DIR NAME: ")

    animate(animation_frames, model, directory_name)

    if num_static >= 5 and not env.world.at_goal():  # model failed to navigate maze
        return frame, False, completion_per
    else:  # model successfully navigated maze
        return frame, True, completion_per


if __name__ == "__main__":
    main(sys.argv[1:])

