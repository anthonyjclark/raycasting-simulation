#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pycaster import Caster


def display_frame(x: float, y: float) -> None:

    caster = Caster(x, y)
    a = np.array(caster, copy=False)
    plt.imshow(a)
    plt.show()


display_frame(21.0, 11.5)
display_frame(21.5, 11.5)
display_frame(22.0, 11.5)
display_frame(22.5, 11.5)
display_frame(23.0, 11.5)
