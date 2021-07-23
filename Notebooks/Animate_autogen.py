# ---
# jupyter:
#   jupytext:
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
# Import matplotlib for animation and other libraries/functions
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pathlib import Path
from PIL import Image
from RNN_classes_funcs_Marchese import get_filenames

# +
# Get filenames of images for animation
path = Path("data_RNN")
all_filenames = get_filenames(path)

# Sort image filenames in sequential order
all_filenames.sort()
all_filenames[:5]
# -

# Get list of images given their filenames
frames = [Image.open(file) for file in all_filenames]

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
ani = FuncAnimation(fig, animate, frames[::20], init_func = init, interval = 200)
HTML(ani.to_jshtml())
# -




