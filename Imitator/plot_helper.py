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
import seaborn as sns

colors = [
    "tab:blue",
    "tab:orange",
    "tab:purple",
    "tab:red",
    "tab:green",
    "tab:brown",
    "tab:pink",
]

def get_network_name(m):
    name = m.split("-")[1]
    if name[0:3] == "xse":
        name = (
            name.replace("xse", "X-SE")
            .replace("_res", "-Res")
            .replace("net", "Net")
            .replace("next", "NeXt")
        )
    elif name[0] == "x":
        name = (
            name.capitalize()
            .replace("net", "Net")
            .replace("resnext", "ResNeXt")
            .replace("res", "-Res")
        )
    elif "squeeze" in name:
        name = name.capitalize().replace("_", "-").replace("net", "Net")
    elif "vgg" in name:
        name = name.replace("vgg", "VGG").replace("bn", "BN").replace("_", "-")
    elif "dense" in name:
        name = name.capitalize().replace("net", "Net")
    elif name[0:3] == "res":
        name = name.capitalize().replace("net", "Net")
    elif "alex" in name:
        name = name.capitalize().replace("net", "Net")

    if "deep" in name and "deeper" not in name:
        name = name.replace("_deep", "b")
    elif "deeper" in name:
        name = name.replace("_deeper", "c")
    elif "RNN" in name:
        name = ""
    else:
        name = name + "a"

    if (
        ("Squeeze" in name)
        or ("VGG" in name)
        or ("ResNext" in name and "SE" not in name)
        or ("ResNet" in name and "SE" in name)
        or ("Dense" in name)
        or ("Alex" in name)
        or ("ResNet" in name and "SE" not in name and "X-" not in name)
    ):
        name = name[:-1]

    return name

def get_clean_names(m):
    return m.split("-")[1]

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
    df = df.assign(error=1.96*df['std']/np.sqrt(len(mazes)))
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

def plot_bars(df, metric):
    matplotlib.rcParams.update({'font.size': 12})
    
    full_names = list(df["Network"])
    clean_names = list(map(get_network_name, full_names))
    unique_names = list(pd.unique(clean_names))
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
    fig, ax = plt.subplots(figsize=(12, 12))
    y_lab = "Average Steps Needed Over Averaged Mazes"

    x = np.arange(len(clean_names))
    width = 0.65
    vals = list(df["mean"])
    
    if metric == "Completion":
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        y_lab = "Average Completion Over Averaged Mazes"
        ax.set_xticks(np.arange(0, 100, 10))
        ax.set_xlim(left = 0.0, right = 100.0)
    else:
        ax.set_xticks(np.arange(0, max(vals), 200))
        ax.set_xlim(left = 0.0, right = max(vals))
    
    ax.barh(
        x,
        vals,
        xerr=ci_bounds,
        color=color_labels,
        align="center",
        alpha=0.75,
        ecolor="grey",
        capsize=2,
    )
    ax.set_yticks(x)
#     ax.set_yticks(np.arange(0, max(vals) + max_error, increment))
    ax.set_yticklabels(labels=sparse_labels)
    # ax.legend()
    # Axis styling.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)

    ax.set_xlabel("Percentage Completed", labelpad=15)
    ax.set_ylabel("Architecture", labelpad=15)
    ax.set_title(f"{metric} Navigated per Network Replicate")
    return ax 

# Scatter Plot Code
def get_csv(file):
    return "csv" in file

def get_petrained(file):
    return "-pretrained" in file

def get_nonpetrained(file):
    return "-notpretrained" in file

def get_losses(csv_files, data_dir, loss_type):
    training_losses = []
    for c in csv_files:
        if c == "classification-resnet18-pretrained-trainlog-0.csv":
            training_losses.append(0.0)
            continue
        df = pandas.read_csv(data_dir + "/" + c)
        if len(df) == 0:
            training_losses.append(0.0)
        else:
            if loss_type == "time":
                training_losses.append(min(df[loss_type]))
            elif loss_type == "accuracy":
                training_losses.append(max(df[loss_type]))
            else:
                training_losses.append(min(df[loss_type]))
    return training_losses

def merge_loss_data(data_dir, df, loss_type, model_type, average=False):
    data_files = os.listdir(data_dir)
    data_files.sort()
    csvs = list(filter(get_csv, data_files))
    if model_type == "pretrained":
        csvs = list(filter(get_petrained, csvs))
    elif model_type == "rnn":
        csvs = csvs
    elif model_type == "cmd":
        csvs = csvs
    else: 
        csvs = list(filter(get_nonpetrained, csvs))
    losses = get_losses(csvs, data_dir, loss_type)
    means = df['mean']
    names = list(map(get_network_name, df['Network']))    
    
    if average:        
        losses = np.array(losses).reshape(-1,4).mean(axis=1)
        means = np.array(means).reshape(-1,4).mean(axis=1)
        names = list(pd.unique(names))
    
    df = pandas.DataFrame()
    df = df.assign(clean_names = names)
    df = df.assign(losses = losses)
    df = df.assign(mean_completion = means)
    return df

def plot_average_scatter(df):   
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='paper', style='whitegrid', font_scale=1, rc=custom_params)    
#     matplotlib.rcParams.update({'font.size': 12})
#     sns.set_theme()
#     sns.set_context("paper")
    fig = plt.gcf()
    fig.set_size_inches(12, 9)
    sns.scatterplot(data=df, x="losses", y="mean_completion", hue="clean_names", style="clean_names", s=200)
    plt.legend(bbox_to_anchor=(1.10, 1), borderaxespad=0)
    x = list(df['losses'])
    y = list(df['mean_completion'])
    for i, label in enumerate(list(df['clean_names'])):
        plt.annotate(label, (x[i], y[i]))
    return plt

def clean_maze_name(m):
    return m[:-4].capitalize().replace("_", " ")

def plot_boxplot(df, mazes):
    all_data = []
    for m in mazes:    
        x = list(df[m])
        all_data.append(x)
    all_data = pd.DataFrame(all_data).T
    all_data.columns = mazes
    all_data = pd.DataFrame(all_data.stack())
    all_data.reset_index(level=1, inplace=True)
    all_data.columns = ['maze', 'percentage']
    clean_mazes = list(map(clean_maze_name, list(all_data['maze'])))
    all_data = all_data.assign(maze=clean_mazes)
        
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='paper', style='whitegrid', font_scale=1, rc=custom_params)
    fig = plt.gcf()
    fig.set_size_inches(17, 9)
    sns.swarmplot(x="maze", y="percentage", data=all_data)
    plt.ylim=(0, None)
    return plt
