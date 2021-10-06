# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import Imitate
from Imitate import *
import RegressionImitator
from RegressionImitator import *
import pandas
from plot_helper import *

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
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         torch.cuda.set_device(3)
#     else:
#         device = torch.device('cpu')
    
    num_mazes = 20
    model_dir = '/raid/clark/summer2021/datasets/corrected-wander-full/regression_models1'
    models = os.listdir(model_dir)
#     models = list(filter(lambda x: "-notpretrained" in x, models))
    models.sort()
    for i, m in enumerate(models):
        models[i] = (model_dir + '/' + m, 'c', 'n', 'y') #model type, stacked, regression

    maze_dir = "../Mazes/validation_mazes8x8"
    mazes = os.listdir(maze_dir)
    mazes.sort()
    mazes = list(filter(lambda x: "maze" in x, mazes))
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
#     subdir = f"{num_mazes}_mazes_test_{now}"
#     maze_sub_dir = os.path.join(maze_dir, subdir)
#     os.mkdir(maze_sub_dir)
    dir_name = f"diagnostics-{now}"
#     print("In AutoWrapper dir name: " + dir_name)

    min_size, max_size = 8, 14

    model_names = [Path(m_path).name for m_path, _, _, _ in models]
    data = {model_name: [] for model_name in model_names}
    completion_data = {model_name: [] for model_name in model_names}
    for maze in mazes:
        maze_file = "../Mazes/validation_mazes8x8/" + maze
        for j, m in enumerate(models):
#             print("GPU: " + str(torch.cuda.current_device()))
#             if (j >= 4 and j < 28) or (j>=32):
#                 device = torch.device('cuda')
#                 torch.cuda.set_device(2)                
#             else:
#                 device = torch.device('cuda')
#                 torch.cuda.set_device(1)
            model, model_type, stacked, regression = m
            input_args = [maze_file, model, 17, model_type, stacked, dir_name]
            print(f"Testing model {j} on {maze}")
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
#     os.system(dir_name) 
#     save_path = os.path.abspath(dir_name)
    
    # Make Bar Plot
#     mazes = [f"maze_{i+1}" for i in range(num_mazes)]
    stepdata = get_df(data, mazes)
    cdata = get_df(completion_data, mazes)
    
    stepdata.to_csv(dir_name + "/regression_step.csv") #
    cdata.to_csv(dir_name + "/regression_percentage.csv")
    
    clean_names = list(map(get_network_name, model_names))
#     stepdata = get_df(data, mazes)
#     cdata = get_df(completion_data, mazes)
    
    # make step bar
    ax = plot_bars(stepdata, "Steps", clean_names)
    ax.figure.savefig(os.path.join(save_path, f"step_barchart_{now}.png"))
    
    # make completion bar
    ax = plot_bars(cdata, "Completion", clean_names) 
    ax.figure.savefig(os.path.join(save_path, f"completion_barchart_{now}.png"))

    # make scatter plots
    data_dir = "/raid/clark/summer2021/datasets/uniform-full/data"
    
    # valid average
    avg_df = merge_loss_data(data_dir, stepdata, "valid_loss", clean_names, True)
    ax = plot_average_scatter(avg_df)
    ax.set_title("Averaged Steps Taken Over Valid Loss Per Model")
    ax.set_xlabel("Valid Loss")
    ax.figure.savefig(os.path.join(save_path, f"scatter_valid_average_{now}.png"))

    # valid no average
    avg_df = merge_loss_data(data_dir, stepdata, "valid_loss", clean_names, False)
    ax = plot_average_scatter(avg_df)
    ax.set_xlabel("Valid Loss")
    ax.set_title("Averaged Steps Taken Over Valid Loss Per Model")
    ax.set_ylabel("Steps Taken Averaged Over Mazes")
    ax.figure.savefig(os.path.join(save_path, f"scatter_valid_{now}.png"))
    
    # train average
    tlosses_df = merge_loss_data(data_dir, stepdata, "train_loss", clean_names, True)
    ax = plot_average_scatter(tlosses_df)
    ax.figure.savefig(os.path.join(save_path, f"scatter_train_average_{now}.png"))

    
    # train no average
    tlosses_df = merge_loss_data(data_dir, stepdata, "train_loss", clean_names, False)
    ax = plot_average_scatter(tlosses_df)
    ax.set_ylabel("Steps Taken Averaged Over Mazes")
    ax.figure.savefig(os.path.join(save_path, f"scatter_train_{now}.png"))

    
    # Generate completion box plots
    all_data = []
    fig, ax = plt.subplots(figsize=(16, 9))
    for m in mazes:    
        x = list(stepdata[m])
        all_data.append(x)
    ax.boxplot(all_data, labels=mazes)
    ax.set_xlabel("Mazes")
    ax.set_ylabel("Steps")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_title("Distribution of Steps per Maze")
    ax.figure.savefig(os.path.join(save_path, f"boxplot_{now}.png"))



if __name__ == "__main__":
    main()
