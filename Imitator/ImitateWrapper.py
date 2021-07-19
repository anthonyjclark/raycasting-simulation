# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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
import RegressionImitator
from RegressionImitator import *
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]


# assume running from Imitator dir
def main():
    num_mazes = int(sys.argv[1]) if len(sys.argv) > 1 else 1

#     ("../Models/auto-gen-r_large-6-6.pkl", 'c', 'n'),
#               ("../Models/auto-gen-c_large-6-6.pkl", 'r', 'n'),
#               ("../Models/auto-stack-r.pkl", 'r', 'y'),
#               ("../Models/auto-stack-c3.pkl", 'c', 'y'),
# ("../Models/regression_model02.pkl", 'r', 'n')
# ("../Models/proxy_regression_model_loss.pkl", 'r', 'y', 'y')
# reg/c  Stck  Reg  CMD in
# ("../Models/cmd_fai_more2.pth", 'cmd', 'y', 'n')
    models = [("../Models/torch_RNN_finished6_10.pth", 'rnn', 'y', 'n')]
#               ("../Models/regression_model02.pkl", 'r', 'y', 'y'),
#               ("../Models/auto-stack-c3.pkl", 'c', 'y', 'n')]

    maze_dir = "../Mazes/"
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    subdir = f"{num_mazes}_mazes_test_{now}"
    maze_sub_dir = os.path.join(maze_dir, subdir)
    os.mkdir(maze_sub_dir)
    dir_name = f"diagnostics-{now}"
    print("In AutoWrapper dir name: " + dir_name)

    min_size, max_size = 8, 14

    model_names = [Path(m_path).name for m_path, _, _, _ in models]
    data = {model_name: [] for model_name in model_names}
    completion_data = {model_name: [] for model_name in model_names}
    for i in range(num_mazes):
        size = random.randint(min_size, max_size)
        maze_file = os.path.join(maze_sub_dir, f"maze_{i+1}.txt")

        print(f"Creating maze {i+1} with size {size}")
        os.system(
            f"python3 ../MazeGen/MazeGen.py --width {size} --height {size} --out > {maze_file}"
        )

        for j, m in enumerate(models):
            model, model_type, stacked, regression = m
            input_args = [maze_file, model, 10, model_type, stacked, dir_name]
            print(f"Testing model {j} on maze {i}")
            if regression == 'y':
                num_frames, success, completion_per = RegressionImitator.main(input_args)
                data[Path(model).name].append(num_frames)
                completion_data[Path(model).name].append(completion_per)
            else:                
                num_frames, success, completion_per = Imitate.main(input_args)
                data[Path(model).name].append(num_frames)
                completion_data[Path(model).name].append(completion_per)
            
    # Generate plots
    
    # Make new directory for plots
    os.system(dir_name) 
    save_path = os.path.abspath(dir_name)
    
    # Make Bar Plot
    mazes = [f"maze {i+1}" for i in range(num_mazes)]
    ind = np.arange(num_mazes)
    width = 0.15
    bars = []
    plt.clf()
    plt.figure(figsize=(20, 10)) 
    for k, model_name in enumerate(model_names):
        xvals = data[model_name]
        color = colors[k]
        bar = plt.bar(ind+width*k, xvals, width, color=color)
        bars.append(bar)
    plt.xlabel("Mazes")
    plt.xticks(rotation=90)
    plt.ylabel("Number of frames")
    plt.title("Frames to navigate")
    plt.xticks(ind+width, mazes)
    plt.legend(bars, model_names)
    plt.savefig(os.path.join(save_path, f"barchart_{now}.png"))
    plt.show()

    # Generate Completion bar plot
    plt.clf()
    plt.figure(figsize=(20, 10)) 
    bars = []
    for k, model_name in enumerate(model_names):
        xvals = completion_data[model_name]
        color = colors[k]
        bar = plt.bar(ind+width*k, xvals, width, color=color)
        bars.append(bar)
    plt.xlabel("Mazes")
    plt.xticks(rotation=90)
    plt.ylabel("Percentage Complete")
    plt.title("Percentage Navigated")
    plt.xticks(ind+width, mazes)
    plt.legend(bars, model_names)
    plt.savefig(os.path.join(save_path, f"completionchart_{now}.png"))
    plt.show()
    
    # Generate completion box plots
    plt.clf()
    plt.figure(figsize=(20, 10)) 
    bars = []
    boxplot_dict = {}
    df = []
    for k,model_name in enumerate(model_names):
        df.append(completion_data[model_name])
    plt.boxplot(x=df, labels=model_names);
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.ylabel("Percentage Complete")
    plt.title("Percentage Navigated")
    plt.savefig(os.path.join(save_path, f"boxplots_{now}.png"))
    plt.show()

if __name__ == "__main__":
    main()
