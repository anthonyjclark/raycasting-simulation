#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pycaster import Caster


def display_frame(caster: Caster, x: float, y: float) -> None:
    caster.render(x, y)
    a = np.array(caster, copy=False)
    plt.imshow(a)
    plt.show()


# world = [[8] * 24] + [[8] + [0] * 22 + [8]] * 22 + [[8] * 24]
world = [
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
    [8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 8],
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
]

width = 640
height = 480
caster = Caster(width, height, world)

# display_frame(caster, 21.0, 11.5)
# display_frame(caster, 21.5, 11.5)
# display_frame(caster, 22.0, 11.5)
# display_frame(caster, 22.5, 11.5)
# display_frame(caster, 23.0, 11.5)
display_frame(caster, len(world[0]) // 2, len(world) // 2)
