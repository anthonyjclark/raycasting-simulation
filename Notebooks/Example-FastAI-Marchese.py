# ---
# jupyter:
#   jupytext:
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
# Import necessary packages and libraries
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
from fastai.vision.all import *

# Append paths to access certain .py files
sys.path.append("../PycastWorld")
sys.path.append("../Gym")
sys.path.append("../Notebooks")

# Import functions and classes from .py files
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv
from cmd_classes_funcs_Marchese import *
from RNN_classes_funcs_Marchese import *
# -

# load in trained network
net = ConvRNN()
model_name = 'fai_RNN.pth'
net.load_state_dict(torch.load(model_name))
net.eval()

# Number of steps the model should walk before it stops or reaches the end of the maze
steps_per_episode = 2500

# Load in the world environment
env = PycastWorldEnv("../Mazes/maze_test01.txt", 224, 224)

# Grab the initial observation 
observation = env.reset()
frames = [observation.copy()]

# +
# Variable to keep track of the previous move
prev_move = 1

# Variable to keep track of current move's name
action_name = 'straight'

# +
# Loop to have the model walk through the maze
for t in range(steps_per_episode):
    
    # Checks what kind of model the network is in order to give the appropriate inputs
    if "cmd" in model_name:
        inp = (tensor(observation/255).permute(2,0,1).unsqueeze(0), tensor(prev_move).unsqueeze(0)) 
    elif "RNN" in model_name:
        inp = tensor(observation/255).permute(2,0,1).unsqueeze(0).unsqueeze(0)
    else:
        inp = tensor(observation/255).permute(2,0,1)
    
    # Use a trained neural network to select the action
    output = net(inp)
    
    # Getting the most probable move
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
        
    # Update previous move
    prev_move = action_index
    
    print(action_name + " " + str(action_index))

    # Advance the world one step. We could also have the step
    # method advance more than step so that it takes fewer
    # steps in total to get to the end goal.
    observation, reward, done, info = env.step(action_index)
    
    # Collect frames one by one
    frames.append(observation.copy())
    
    # Check if we reached the end goal
    if done:
        print(f"  Found goal in {t+1} steps")
        break

# Show final image
plt.imshow(observation)

# Print ending position
#print(f"  Ended at position {env.world.get_x()}, {env.world.get_y()}")

# Close environment
env.close()

# +
# Initialize subplots
fig, ax = plt.subplots()

# Set axis of animation
ax.set(xlim=(0,300), ylim=(240,0))

# Initialize frame
frame = plt.imshow(frames[0])

def init():
    """
    Creates initial frame for animation.
    
    :return: (list) a list of the initial frame for animation
    """
    frame.set_data(frames[0])
    return [frame]

def animate(i):
    """
    Updates the state of the frame at each point of the animation.
    
    :param i: (ndarray) current image/frame
    :return: (list) a list of the current frame for animation
    """
    # Define the updated content of frame
    frame.set_array(i)
    return [frame]

from IPython.display import HTML

# Take images, functions, and other parameters and create an animation
ani = FuncAnimation(fig, animate, frames[::35], init_func = init, interval = 200)
HTML(ani.to_jshtml())
# -

# Save animation
ani.save("animation" + ".gif")






