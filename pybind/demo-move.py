#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pycaster import Caster


def display_frame(caster: Caster, x: float, y: float) -> None:
    caster.render(x, y)
    a = np.array(caster, copy=False)
    plt.imshow(a)
    plt.show()


caster = Caster()
display_frame(caster, 21.0, 11.5)
display_frame(caster, 21.5, 11.5)
display_frame(caster, 22.0, 11.5)
display_frame(caster, 22.5, 11.5)
display_frame(caster, 23.0, 11.5)
