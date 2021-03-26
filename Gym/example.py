import gym

import sys

sys.path.append("../PycastWorld")
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv

env = PycastWorldEnv()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
