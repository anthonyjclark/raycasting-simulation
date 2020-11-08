#!/usr/bin/env python3

# import numpy as np
# from pycaster import Container

# x = Container(9)
# y = np.array(x.getInts(), copy=False)
# print(y.shape)
# print(y)
# y = np.array(x, copy=False)
# print(y.shape)
# print(y)

import numpy as np
import matplotlib.pyplot as plt
from pycaster import Caster

caster = Caster()
a = np.array(caster, copy=False)
print(a.shape)
# a = np.array(caster, copy=False).reshape((480, 640, 3))
# # img = np.zeros((480, 640, 3), dtype=np.int)
# # img[:, :, 0] = (a[:, :] & 0xFF0000) >> 16
# # img[:, :, 1] = (a[:, :] & 0xFF00) >> 8
# # img[:, :, 2] = a[:, :] & 0xFF

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
