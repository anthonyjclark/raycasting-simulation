import gym

import sys

sys.path.append("../PycastWorld")
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv  # type: ignore


num_episodes = 10
steps_per_episode = 100

env = PycastWorldEnv()

# Run some number of trials all starting from the
# initial location. We might eventually randomize
# the maze and the starting location.
for episode in range(num_episodes):

    # Grab the initial observation (not used here)
    observation = env.reset()

    print("Starting episode", episode)

    for t in range(steps_per_episode):

        # Remove this render call when actually training;
        # it will needlessly slow things down if you don't want
        # to watch.
        env.render()

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
            break

    print(f"  Ended at position {env.world.getX()}, {env.world.getY()}")
env.close()
