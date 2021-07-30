# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
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
import matplotlib.ticker as mtick
import numpy as np
import os
from pathlib import Path
import sys
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

# %%
# assume running from Imitator dir
num_mazes = 2
model_dir = '/raid/clark/summer2021/datasets/uniform-small/models'
models = os.listdir(model_dir)
models.sort()
for i, m in enumerate(models):
    models[i] = (model_dir + '/' + m, 'c', 'n', 'n')

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
    size = 8#random.randint(min_size, max_size)
    maze_file = os.path.join(maze_sub_dir, f"maze_{i+1}.txt")

    print(f"Creating maze {i+1} with size {size}")
    os.system(
        f"python3 ../MazeGen/MazeGen.py --width {size} --height {size} --out > {maze_file}"
    )

    for j, m in enumerate(models):
        model, model_type, stacked, regression = m
        input_args = [maze_file, model, 17, model_type, stacked, dir_name]
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


# %% [markdown]
# # Bar Plot Code

# %%
def get_network_name(m):
    return m.split('-')[1]


# %%
def index_one(num):
    return num[0]


# %%
import pandas

# %%
mazes = [f"maze_{i+1}" for i in range(num_mazes)]

# %%
clean_names = list(map(get_network_name, model_names))


# %%
def get_df(data_dict):
    df = pd.DataFrame.from_dict(data_dict)
    df = df.assign(Maze=mazes)
    df = df.T
    df.columns = df.iloc[-1]
    df = df.drop(df.index[-1])
    # df['Network'] = df.index
    df = df.reset_index()
    df.rename(columns={'index':'Network'}, inplace=True)
    df['std'] = df[df.columns[1:]].std(axis=1)
    df['mean'] = df[df.columns[1:-1]].mean(axis=1)
    # df = df.assign(lower_error=1.96*df['std'])
    df = df.assign(error=1.96*df['std'])
    return df


# %%
stepdata = get_df(data)
stepdata

# %%
cdata = get_df(completion_data)
cdata


# %%
def get_error(df):
    ci_bounds = df['error'].to_numpy()
    return ci_bounds


# %%
cdata


# %%
def get_colors(num_unique_mazes) -> list:
    color_labels = []
    for i in range(num_unique_mazes):
        color = colors[i % len(colors)]
        for i in range(4):
            color_labels.append(color)
    return color_labels


# %%
def plot_bars(df, metric):
    full_names = list(df['Network'])
    clean_names = list(map(get_network_name, full_names))
    unique_names = list(set(clean_names))
    color_labels = get_colors(len(unique_names))
    ci_bounds = get_error(df)
    max_error = max(df['error'])
    increment = 5
    fig, ax = plt.subplots(figsize=(16, 9)) 
    
    y_lab = "Average Steps Needed over Averaged Mazes"
    if metric=="Completion":
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        y_lab = "Average Completion Over Averaged Mazes"
        increment = 1
    
    x = np.arange(len(clean_names))
    width = .15
    vals = list(df['mean'])
    ax.bar(x, vals, yerr = ci_bounds, color = color_labels, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, max(vals)+max_error, increment))
    ax.set_xticklabels(labels = clean_names, rotation = 45)
    # ax.legend()
    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    ax.set_xlabel('Network', labelpad=15)
    ax.set_ylabel(y_lab, labelpad=15)
    ax.set_title(f"{metric} Navigated per Network Replicate")
    return ax


# %%
ax = plot_bars(stepdata, "Steps")

# %%
ax = plot_bars(cdata, "Completion") 

# %% [markdown]
# # Scatter Plot Code

# %%
test_csv = '/raid/clark/summer2021/datasets/uniform-small/data/classification-resnet18-pretrained-trainlog-0.csv'


# %%
def get_csv(file):
    return "csv" in file


# %%
data_dir = '/raid/clark/summer2021/datasets/uniform-small/data'
data_files = os.listdir(data_dir)
data_files.sort()
data_files

# %%
csvs = list(filter(get_csv, data_files))
csvs

# %%
training_losses = []

for c in csvs:
    df = pandas.read_csv(data_dir + "/" + c)
    training_losses.append(min(df['train_loss']))

# %%
training_losses

# %%
average_losses = np.array(training_losses).reshape(-1,4).mean(axis=1)
average_losses

# %%
means = stepdata['mean']
means

# %%
means_over_models = np.array(means).reshape(-1,4).mean(axis=1)
means_over_models

# %%
unique_names = list(set(clean_names))

# %%
df = pandas.DataFrame()
df = df.assign(Network = unique_names)
df = df.assign(average_training_losses = average_losses)
df = df.assign(mean_steps = means_over_models)

# %%
df

# %%
get_colors(len(df['Network']))

# %%
fig, ax = plt.subplots(figsize=(16, 9))
x = list(df["average_training_losses"])
y = list(df["mean_steps"])
ax.scatter(x, y, s=200, c=df.average_training_losses, alpha=.5)
ax.set_xlabel("Training Loss")
# ax.legend(loc='upper left')
for i, label in enumerate(list(df['Network'])):
    ax.annotate(label, (x[i], y[i]))
