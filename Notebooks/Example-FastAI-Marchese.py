# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import sys
from fastai.vision.all import *

sys.path.append("../PycastWorld")
sys.path.append("../Gym")
sys.path.append("../Notebooks")

from gym_pycastworld.PycastWorldEnv import PycastWorldEnv
from cmd_classes_funcs_Marchese import *
from RNN_classes_funcs_Marchese import *
# -

# loading in trained NN
net = ConvRNN()
net.load_state_dict(torch.load('torch_RNN.pth'))
net.eval()

steps_per_episode = 2000

env = PycastWorldEnv("../Mazes/maze_test00.txt", 320, 240)

# Grab the initial observation (not used here)
observation = env.reset()
frames = [observation.copy()]

# Variable to keep track of the previous move
prev_move = 1
# Variable to keep track of current move's name
action_name = 'straight'

# +
for t in range(steps_per_episode):

    # Remove this render call when actually training;
    # it will needlessly slow things down if you don't want
    # to watch.
    # TODO: cannot render on HPC
    # env.render()
    
    #Non-RNN input
    #inp = (tensor(observation/255).permute(2,0,1).unsqueeze(0), tensor(prev_move).unsqueeze(0)) 
    #RNN input
    inp = tensor(observation/255).permute(2,0,1).unsqueeze(0).unsqueeze(0)
    
    # Use a trained neural network to select the action
    output = net(inp)
    
    # Output is a multi-value tensor; in order to know the action_index
    # we must take max and see what move is associated with that max
    
    # Getting the most probable move
    #print(output)
    action_index = int(torch.argmax(output[0]))
                
    # Looking at what move has that probability
    if action_index == 0:
        action_name = 'left'
    elif action_index == 1:
        # Must account for differing move indexing
        # Model sees 'right' as index 1 and 'straight' as index 2;
        # this needs to be flipped
        action_index = 2
        action_name = 'right'
    else:
        action_index = 1
        action_name = 'straight'
        
    """
    # Account for left-right stuck mistake
    if action_index == 0 and prev_move == 2:
        action_index = 2
    elif action_index == 2 and prev_move == 0:
        action_index = 0"""
    
    # Update previous move
    prev_move = action_index
    
    print(action_name + " " + str(action_index))

    # Advance the world one step. We could also have the step
    # method advance more than step so that it takes fewer
    # steps in total to get to the end goal.
    observation, reward, done, info = env.step(action_index)
    frames.append(observation.copy())
    # Check if we reached the end goal
    if done:
        print(f"  Found goal in {t+1} steps")
        break
        
plt.imshow(observation)
print(f"  Ended at position {env.world.get_x()}, {env.world.get_y()}")
env.close()

# +
# TODO: matplotlib animation
# https://matplotlib.org/stable/api/animation_api.html
fig, ax = plt.subplots()
ax.set(xlim=(0,300), ylim=(240,0))

frame = plt.imshow(frames[0])
print(f"  Ended at position {env.world.getX()}, {env.world.getY()}")


# +
def init():
    frame.set_data(frames[0])
    return [frame]

def animate(i):
    frame.set_array(i)
    return [frame]


# +
from IPython.display import HTML

ani = FuncAnimation(fig, animate, frames[::25], init_func = init, interval = 200)
HTML(ani.to_jshtml())

# +
#ani.save("prediction_next" + ".gif")
# -






