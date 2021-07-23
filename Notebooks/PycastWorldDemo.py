# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt

import sys

sys.path.append("../PycastWorld")
sys.path.append("../Gym")
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv

# %%
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

# %%
plt.imshow(observation)

# %%
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
observation, reward, done, info = env.step(action)

# %%
plt.imshow(observation)
print(f"  Ended at position {env.world.x()}, {env.world.y()}")

# %%
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
observation, reward, done, info = env.step(0)

# %%
plt.imshow(observation)

# %%
observation = env.reset()

# %%
plt.imshow(observation)

# %%
dir(env.world)

# %%
env.world.set_position(3.5, 3.5)
observation = env.reset()
plt.imshow(observation)

# %%
env.reset()
env.world.set_position(6.5, 3.5)
observation = env.render(mode='rgb_array')
plt.imshow(observation)

# %%
env.reset()
print(f"Position: {env.world.x()}, {env.world.y()}")

# %%
env.step(0)
print(f"Position: {env.world.x()}, {env.world.y()}")

# %%
env.step(1)
print(f"Position: {env.world.x()}, {env.world.y()}")

# %%
env.step(2)
print(f"Position: {env.world.x()}, {env.world.y()}")

# %%
observation, reward, done, info = env.step(1)
plt.imshow(observation)

# %%
for _ in range(100):
    observation, reward, done, info = env.step(1)
plt.imshow(observation)

# %%
for _ in range(100):
    observation, reward, done, info = env.step(1)
plt.imshow(observation)

# %%
for _ in range(10):
    observation, reward, done, info = env.step(0)
plt.imshow(observation)

# %%
