#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from caster2bind import Caster


cast = Caster()
a = np.array(cast.buffer, copy=False).reshape((480, 640))
img = np.zeros((480, 640, 3), dtype=np.int)
img[:, :, 0] = (a[:, :] & 0xFF0000) >> 16
img[:, :, 1] = (a[:, :] & 0xFF00) >> 8
img[:, :, 2] = a[:, :] & 0xFF

plt.imshow(img)
plt.show()

from timeit import timeit

setup = """
from caster2bind import Caster
import numpy as np
"""

code = """
cast = Caster()
a = np.array(cast.buffer, copy=False).reshape((480, 640))
img = np.zeros((480, 640, 3), dtype=np.int)
img[:, :, 0] = (a[:, :] & 0xFF0000) >> 16
img[:, :, 1] = (a[:, :] & 0xFF00) >> 8
img[:, :, 2] = a[:, :] & 0xFF
"""

num_trials = 100
t = timeit(code, setup=setup, number=num_trials)
fps = 1 / (t / num_trials)
print(f"FPS: {fps}")
