#!/usr/bin/env python3

from timeit import timeit

setup = """
import numpy as np
from pycaster import Caster
"""

code = """
caster = Caster(640, 480)
caster.render(21.0, 11.5)
a = np.array(caster, copy=False)
"""

num_trials = 1
t = timeit(code, setup=setup, number=num_trials)
fps = 1 / (t / num_trials)
print(f"FPS: {fps}")
