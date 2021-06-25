# TODO:
# - typing
# - need random seeding for anything?
# - set position as percentage along maze??

# Based on
# https://github.com/openai/gym/blob/master/gym/core.py
# https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from typing import Dict, List, Tuple

from pycaster import PycastWorld, Turn, Walk  # type: ignore

import sys

sys.path.append("../MazeGen")
from MazeUtils import read_maze_file, percent_through_maze, bfs_dist_maze, is_on_path  # type: ignore


class PycastWorldEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, mazefile: str, image_width: int, image_height: int) -> None:
        super().__init__()

        # image_width = 320
        # image_height = 240
        # mazefile = mazefile if mazefile else "../Mazes/maze01.txt"
        # "../Mazes/maze01.txt", 320, 240

        self.world = PycastWorld(image_width, image_height, mazefile)

        self.seed()

        self._action_names = ["TurnLeft", "MoveForward", "TurnRight"]
        self.action_space = spaces.Discrete(len(self._action_names))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(image_width, image_height, 3), dtype=np.uint8
        )

        self.viewer = None

        # maze, _, _, maze_directions, _ = read_maze_file(mazefile)
        # x_start, y_start, _ = maze_directions[0]
        # x_end, y_end, _ = maze_directions[-1]

        # _, maze_path = bfs_dist_maze(maze, x_start, y_start, x_end, y_end)
        # xy_on_path = is_on_path(maze_path, x, y)
        # xy_pct_path = percent_through_maze(maze, x, y, x_start, y_start, x_end, y_end)

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, Dict]:
        # Update the world according to the current action
        action_name = self._action_names[action]

        if action_name == "TurnLeft":
            self.world.walk(Walk.Stop)
            self.world.turn(Turn.Left)
        elif action_name == "MoveForward":
            self.world.walk(Walk.Forward)
            self.world.turn(Turn.Stop)
        elif action_name == "TurnRight":
            self.world.walk(Walk.Stop)
            self.world.turn(Turn.Right)
        else:
            raise ValueError(f"Invalid action name: {action_name}")

        self.world.update()

        # Grab the new image frame
        ob = np.array(self.world)

        # Reward based on percent traveled through maze
        reward = 1 if self.world.at_goal() else -1

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
