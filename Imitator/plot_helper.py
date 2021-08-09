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
import pandas

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]

def get_network_name(m):
    return m.split('-')[1]

def index_one(num):
    return num[0]

def get_df(data_dict, mazes):
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

def get_error(df):
    ci_bounds = df['error'].to_numpy()
    return ci_bounds

def get_colors(num_unique_mazes) -> list:
    color_labels = []
    for i in range(num_unique_mazes):
        color = colors[i % len(colors)]
        for i in range(4):
            color_labels.append(color)
    return color_labels

def plot_bars(df, metric, clean_names):
    full_names = list(df["Network"])
    clean_names = list(map(get_network_name, full_names))
    unique_names = list(set(clean_names))
    sparse_labels = []

    for i in range(0, len(clean_names)):
        if i % 4 == 0:
            sparse_labels.append(clean_names[i])
        else: 
            sparse_labels.append("")
    
    color_labels = get_colors(len(unique_names))
    ci_bounds = get_error(df)
    max_error = max(df["error"])
    increment = 5
    fig, ax = plt.subplots(figsize=(16, 9))
    y_lab = "Average Steps Needed Over Averaged Mazes"

    x = np.arange(len(clean_names))
    width = 0.65
    vals = list(df["mean"])
    
    if metric == "Completion":
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        y_lab = "Average Completion Over Averaged Mazes"
        ax.set_yticks(np.arange(0, 100, 10))
    else:
        ax.set_yticks(np.arange(0, max(vals), 500))
    
    ax.bar(
        x,
        vals,
        yerr=ci_bounds,
        color=color_labels,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=2,
    )
    ax.set_xticks(x)
#     ax.set_yticks(np.arange(0, max(vals) + max_error, increment))
    ax.set_xticklabels(labels=sparse_labels, rotation=45)
    # ax.legend()
    # Axis styling.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)

    ax.set_xlabel("Network", labelpad=15)
    ax.set_ylabel(y_lab, labelpad=15)
    ax.set_title(f"{metric} Navigated per Network Replicate")
    return ax 

# Scatter Plot Code
def get_csv(file):
    return "csv" in file

def get_losses(csv_files, data_dir, loss_type):
    training_losses = []
    for c in csv_files:
        df = pandas.read_csv(data_dir + "/" + c)
        training_losses.append(min(df[loss_type]))
    return training_losses

def merge_loss_data(data_dir, df, loss_type, clean_names, average=False):
    data_files = os.listdir(data_dir)
    data_files.sort()
    csvs = list(filter(get_csv, data_files))
    losses = get_losses(csvs, data_dir, loss_type)
    means = df['mean']
    names = clean_names    
    
    if average:        
        losses = np.array(losses).reshape(-1,4).mean(axis=1)
        means = np.array(means).reshape(-1,4).mean(axis=1)
        names = list(set(clean_names))
    
    df = pandas.DataFrame()
    df = df.assign(Network = names)
    df = df.assign(losses = losses)
    df = df.assign(mean_steps = means)
    return df

def plot_average_scatter(df):    
    fig, ax = plt.subplots(figsize=(16, 9))
    x = list(df["losses"])
    y = list(df["mean_steps"])
    ax.scatter(x, y, s=200, c=df.losses, alpha=.5)
    ax.set_xlabel("Training Loss")
    ax.set_ylabel("Steps Averaged Over Replicates")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_title("Averaged Steps Taken Over Training Loss per Model")
    for i, label in enumerate(list(df['Network'])):
        ax.annotate(label, (x[i], y[i]))
    return ax