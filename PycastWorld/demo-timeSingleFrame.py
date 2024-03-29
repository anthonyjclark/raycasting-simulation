#!/usr/bin/env python3

from timeit import timeit

setup = """
import numpy as np
from pycaster import PycastWorld

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
"""

code = """
world = PycastWorld(WINDOW_WIDTH, WINDOW_HEIGHT, "../Mazes/maze01.txt")
"""

num_trials = 1
t = timeit(code, setup=setup, number=num_trials)
fps = 1 / (t / num_trials)
print(f"FPS: {fps}")
