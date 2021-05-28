# TODO:
# - typing
# - need random seeding for anything?

# Based on
# https://github.com/openai/gym/blob/master/gym/core.py
# https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from typing import Dict, List, Tuple

from pycaster import PycastWorld, Turn, Walk  # type: ignore


class PycastWorldEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self) -> None:
        super().__init__()

        image_width = 320
        image_height = 240

        self.world = PycastWorld(image_width, image_height, "../Worlds/maze.txt")
        self.world.direction(0, 1.152)  # TODO: remove?

        self.seed()

        self._action_names = ["TurnLeft", "MoveForward", "TurnRight"]
        self.action_space = spaces.Discrete(len(self._action_names))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(image_width, image_height, 3), dtype=np.uint8
        )

        self.viewer = None

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, Dict]:
        # Update the world according to the current action
        action_name = self._action_names[action]

        if action_name == "TurnLeft":
            self.world.walk(Walk.Stopped)
            self.world.turn(Turn.Left)
        elif action_name == "MoveForward":
            self.world.walk(Walk.Forward)
            self.world.turn(Turn.Stop)
        elif action_name == "TurnRight":
            self.world.walk(Walk.Stopped)
            self.world.turn(Turn.Right)
        else:
            raise ValueError(f"Invalid action name: {action_name}")

        self.world.update()

        # Grab the new image frame
        ob = np.array(self.world)

        # Reward with 1 if at the end
        reward = 1 if self.world.atGoal() else -1

        # Episode is over if we reached the goal
        episode_over = reward == 1

        # Not returning any diagnostic information
        # TODO: return distance from end?
        return ob, reward, episode_over, {}

    def reset(self) -> np.ndarray:
        self.world.reset()
        return np.array(self.world)

    def render(self, mode="human") -> object:
        im = np.array(self.world)

        if mode == "rgb_array":
            return im

        elif mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(im)
            return self.viewer.isopen

        else:
            # Will raise an appropriate exception
            return super().render(mode=mode)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None) -> List[int]:
        # seed numpy and other?
        # Return list of seeds
        return []
