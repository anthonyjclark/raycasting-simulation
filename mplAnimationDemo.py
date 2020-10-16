import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from time import time


class CameraView:
    def __init__(self, ax):
        num_rows = 240
        num_cols = 320
        self.data = np.random.random((num_rows, num_cols, 3))
        self.image = ax.imshow(self.data)

        self.time = time()

    def __call__(self, i):
        self.image.set_array(self.data * i / 100)

        print(time() - self.time)
        self.time = time()

        return [self.image]


# Fixing random state for reproducibility
np.random.seed(842)


fig, ax = plt.subplots()
ax.axis("off")
ud = CameraView(ax)
anim = FuncAnimation(fig, ud, frames=100, interval=20, blit=True)
plt.show()
