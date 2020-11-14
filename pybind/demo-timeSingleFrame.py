#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pycaster import Caster

caster = Caster()

a = np.array(caster, copy=False)
print(a.shape)

plt.imshow(a)
plt.show()

from timeit import timeit

setup = """
from pycaster import Caster
import numpy as np
"""

code = """
caster = Caster()
a = np.array(caster, copy=False)
"""

num_trials = 100
t = timeit(code, setup=setup, number=num_trials)
fps = 1 / (t / num_trials)
print(f"FPS: {fps}")
