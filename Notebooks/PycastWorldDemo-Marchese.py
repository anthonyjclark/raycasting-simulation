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
import matplotlib.pyplot as plt
import sys

sys.path.append("../PycastWorld")
sys.path.append("../Gym")

from gym_pycastworld.PycastWorldEnv import PycastWorldEnv

# +
env = PycastWorldEnv("../Mazes/maze01.txt", 320, 240)

# Grab the initial observation (not used here)
observation = env.reset()

# Random action selection. This should be done in some
# sort of "intelligent" manner.
action = env.action_space.sample()

# Advance the world one step. We could also have the step
# method advance more than step so that it takes fewer
# steps in total to get to the end goal.
observation, reward, done, info = env.step(action)

# Check if we reached the end goal
if done:
    print(f"  Found goal in {t+1} steps")

print(f"  Ended at position {env.world.x()}, {env.world.y()}")
env.close()
# -

plt.imshow(observation)

# ## Collecting Images Manually (First Try)

observation = env.reset()

# +
plt.axis('off')
print(f"  Ended at position {env.world.x()}, {env.world.y()}")

# Getting photos for forward movement
for i in range(35):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i%5 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

# +
plt.axis('off')

# Photos for right move
for i in range(36):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/right_move" + str(i))
# -

plt.axis('off')
# Getting photos for forward movement
for i in range(35,148):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i % 20 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

plt.axis('off')
# Photos for right move
for i in range(36,72):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/right_move" + str(i))

plt.axis('off')
# Getting photos for forward movement
for i in range(148,178):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i % 10 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

plt.axis('off')
# Photos for left move
for i in range(36):    
    observation, reward, done, info = env.step(0)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/left/left_move" + str(i))

plt.axis('off')
# Getting photos for forward movement
for i in range(178,300):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i % 20 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

plt.axis('off')
# Photos for left move
for i in range(36, 72):    
    observation, reward, done, info = env.step(0)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/left/left_move" + str(i))

plt.axis('off')
# Getting photos for forward movement
for i in range(300,330):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i % 10 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

plt.axis('off')
# Photos for right move
for i in range(72,108):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/right_move" + str(i))

plt.axis('off')
# Getting photos for forward movement
for i in range(330,370):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i % 10 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

plt.axis('off')
# Photos for right move
for i in range(108,144):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/right_move" + str(i))

plt.axis('off')
# Getting photos for forward movement
for i in range(370,430):    
    observation, reward, done, info = env.step(1)
    plt.imshow(observation)
    if i % 10 == 0:
        print(f"  Ended at position {env.world.x()}, {env.world.y()}")
    plt.savefig("Images-Marchese/straight/straight_move" + str(i))

# ## Collecting Images Automatically

# - Going to use Jared's automater code with a similar approach to Oliver
# - Want to begin considering how to develop a set of images that allows the model to be self-correcting
#     - wiggle movement to see multiple perspectives while moving down corridor (training adjustments)
#     - forward movements even when path isn't straight down the middle of the corridor
#     - note: if left and right movement frequent in fastai-example code maybe hard code to do multiple movements left/right
#
# Objective: collect enough images to the point where model has an underlying understanding of the corridor's constraints (a kind of spacial awareness)

sys.path.append("../Automator")
from AutoGen import Navigator

# +
maze = "../Mazes/maze01.txt"
img_dir = "../Notebooks/Images-Marchese"
show_freq = 0

navigator = Navigator(maze, img_dir)

j = 0
while j < navigator.num_directions - 1:
    print(navigator.navigate(j, show_dir=True, show_freq=show_freq))
    j += 1
    
print("Images saved.")
# -

# ## Approaching this issue of self-correction

# My plan: put model in bad situations (corners, drifting on walls, etc.) and classify what to do in this scenario

observation = env.reset()
plt.imshow(observation)

env.step(0)
env.step(0)
env.step(0)
env.step(0)
observation, reward, done, info = env.step(0)
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(5):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

observation = env.reset()

env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(1)
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(5,10):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(1)
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(10,46):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

print(f"  Ended at position {env.world.x()}, {env.world.y()}")

env.world.set_position(1.8297720445231849, 3.2429274262081447)
observation = env.render(mode='rgb_array')
plt.imshow(observation)

env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(2)
env.step(2)
env.step(2)
observation, reward, done, info = env.step(2)
plt.imshow(observation)

plt.axis('off')
# Photos for left move
for i in range(4):    
    observation, reward, done, info = env.step(0)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/left/sc" + str(i))

env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(1)
plt.imshow(observation)

plt.axis('off')
# Photos for left move
for i in range(4,17):    
    observation, reward, done, info = env.step(0)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/left/sc" + str(i))

env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(1)
plt.imshow(observation)

env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(1)
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(46,59):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(1)
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(59,72):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

# Found that this self-correcting approach is relatively effective when trying to prevent specific crashes

print(f"  Ended at position {env.world.x()}, {env.world.y()}")

env.world.set_position(1.14,9.7)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
observation = env.render(mode='rgb_array')
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(72,113):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

env.world.set_position(4.494416812870865, 17.887123927609718)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
observation = env.render(mode='rgb_array')
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(113,124):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))

env.world.set_position(3.9276393725869516, 14.065296090129824)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
observation = env.render(mode='rgb_array')
plt.imshow(observation)

plt.axis('off')
# Photos for left move
for i in range(17,41):    
    observation, reward, done, info = env.step(0)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/left/sc" + str(i))

env.world.set_position(1.1058645999344014, 5.8730592235719)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
observation = env.render(mode='rgb_array')
plt.imshow(observation)

plt.axis('off')
# Photos for right move
for i in range(124,142):    
    observation, reward, done, info = env.step(2)
    plt.imshow(observation)
    plt.savefig("Images-Marchese/right/sc" + str(i))




