import sys

sys.path.append("../PycastWorld")
from gym_pycastworld.PycastWorldEnv import PycastWorldEnv  # type: ignore


steps_per_episode = 100

env = PycastWorldEnv()


# Grab the initial observation (not used here)
observation = env.reset()
frames = [observation.copy()]

for t in range(steps_per_episode):

    # Remove this render call when actually training;
    # it will needlessly slow things down if you don't want
    # to watch.
    # TODO: cannot render on HPC
    # env.render()

    # Use a trained neural network to select the action
    # TODO: action_index might need to be shuffled so that
    # 0 --> left
    # 1 --> forward
    # 2 --> right
    action_name, action_index, action_probs = model.predict(observation)

    # Advance the world one step. We could also have the step
    # method advance more than step so that it takes fewer
    # steps in total to get to the end goal.
    observation, reward, done, info = env.step(action_index)
    frames.append(observation.copy())

    # Check if we reached the end goal
    if done:
        print(f"  Found goal in {t+1} steps")
        break

print(f"  Ended at position {env.world.getX()}, {env.world.getY()}")
env.close()


#%%

# TODO: matplotlib animation
# https://matplotlib.org/stable/api/animation_api.html
