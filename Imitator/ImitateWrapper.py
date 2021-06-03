import distutils.util
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import sys
import time

import Imitate
from Imitate import *

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]


# assume running from Imitator dir
def main():
    num_mazes = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    #
    models = [("../Models/learner_2021-06-01 16:40:23.194027.pkl", 'c', 'n'),
              ("../Models/auto-stack-r.pkl", 'r', 'y')]

    maze_dir = "../Worlds/"
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    subdir = f"{num_mazes}_mazes_test_{now}"
    maze_sub_dir = os.path.join(maze_dir, subdir)
    os.mkdir(maze_sub_dir)
    
    min_size, max_size = 8, 14
    
    model_names = [Path(m_path).name for m_path, _, _ in models]
    data = {model_name: [] for model_name in model_names}
    for i in range(num_mazes):
        size = random.randint(min_size, max_size)
        maze_file = os.path.join(maze_sub_dir, f"maze_{i+1}.txt")

        print(f"Creating maze {i+1} with size {size}")
        os.system(f"python3 ../MazeGen/MazeGen.py --width {size} --height {size} --out > {maze_file}")

        for j, m in enumerate(models):
            model, model_type, stacked = m
            print(f"Testing model {j} on maze {i}")
            input_args = [maze_file, model, 10, model_type, stacked]
            num_frames, success, completion_per = Imitate.main(input_args)
            data[Path(model).name].append(num_frames)
            
    mazes = [f"maze {i+1}" for i in range(num_mazes)]
    ind = np.arange(num_mazes)
    width = 0.25
    bars = []
    plt.clf()
    for k, model_name in enumerate(model_names):
        xvals = data[model_name]
        color = colors[k]
        bar = plt.bar(ind+width*k, xvals, width, color=color)
        bars.append(bar)
    
    plt.xlabel("Mazes")
    plt.ylabel("Number of frames")
    plt.title("Frames to navigate")

    plt.xticks(ind+width, mazes)
    plt.legend(bars, model_names)
    plt.savefig(os.path.join(".", f"barchart_{now}.png"))



if __name__ == "__main__":
    main()
