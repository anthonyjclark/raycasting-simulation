# ---
# jupyter:
#   jupytext:
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

# Import necessary libraries and packages
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.models as models
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import *
from fastai.vision.widgets import *
from RNN_classes_funcs_Marchese import *
import re

# Set device to run training on
torch.cuda.set_device(3)
torch.cuda.current_device()

# Get classes and filenames
path = Path("data_RNN")
classes = get_class_labels(path)
all_filenames = get_filenames(path)
all_filenames.sort()

# Get size of dataset and corresponding list of indices
dataset_size = len(all_filenames)
dataset_indices = list(range(dataset_size))

# Get index for where we want to split the data
val_split_index = int(np.floor(0.2 * dataset_size))

# Split list of indices into training and validation indices
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

# Get list of filenames for training and validation set
train_filenames = [all_filenames[i] for i in train_idx]
val_filenames = [all_filenames[i] for i in val_idx]
train_filenames

# Get data via custom dataset
train_data = ImageDataset(classes, train_filenames)
val_data = ImageDataset(classes, val_filenames)

# Create the DataLoader
dls = DataLoaders.from_dsets(train_data, val_data, bs=32, shuffle=False)
dls = dls.cuda()

# Initialize the network
net = ConvRNN()
net = net.cuda()

# Initialize FastAI learner
learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=CSVLogger(fname='test'))

# Find good learning rate
lr = learn.lr_find()

lr = float(re.split("=|\)", str(lr))[1])

lr

# Fit learner
learn.fine_tune(8, lr)

# Save trained model
PATH = 'fai_RNN.pth'
torch.save(net.state_dict(), PATH)


