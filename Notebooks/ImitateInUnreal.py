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

# # Imitate in Unreal Engine

# !pwd

# ## Import Packages

# +
from fastai.vision.all import *

sys.path.append("../Utilities")
from UnrealUtils import UE4EnvWrapper

from time import sleep

sys.path.append("../Automator")
from AutoGen import Navigator
# -

# ### Load Model

model_path = Path("../Models/learner_2021-06-01 16:40:23.194027.pkl")
model_inf = load_learner(model_path)

# ## Get Maze Start Coordinates

maze = "../Mazes/maze01.txt"

navigator = Navigator(maze)

start_x, start_y, _  = navigator.directions[0].split()

start_x, start_y = int(start_x) + 0.5, int(start_y) + 0.5

start_x, start_y


def get_start_coords


# ## Initialize Unreal Engine Environment Wrapper

# +
env = UE4EnvWrapper(3.5, 3.5)

if env.isconnected():
    fig, ax = plt.subplots()
    ax.imshow(env.request_image())
# -

# !pwd

# +
from fastai.vision.all import *

sys.path.append("../Unreal")
from UnrealUtils import UE4EnvWrapper

from time import sleep

# +

model_path = Path("./Models/auto-gen-c.pkl")
model_inf = load_learner(model_path)
# -

env.reset()
frames = []
for _ in range(1000):
    img = env.request_image()
    # Remove alpha channel
    img = img[:,:,:3]
    action = model_inf.predict(img)
    print(action)

    if action[0] == 'straight':
        env.forward()
    elif action[0] == 'left':
        env.left()
    elif action[0] == 'right':
        env.right()
    else:
        print("ERROR:", action)
    frames.append(img)
    sleep(1)

img.shape

type(img)

img = img[:,:,:3]

plt.imshow(img)


