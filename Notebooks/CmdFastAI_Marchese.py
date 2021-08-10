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

# Import necessary packages and libraries
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.models as models
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import *
from fastai.vision.widgets import *
from cmd_classes_funcs_Marchese import *

# Set device to run training on
torch.cuda.set_device(1)
torch.cuda.current_device()

# ## Use two functions and classes from cmd_classes_funcs_Marchese to make train/valid dataset

# Get classes and filenames
path = Path("data")
classes = get_class_labels(path)
all_filenames = get_filenames(path)

# Get size of dataset and corresponding list of indices
dataset_size = len(all_filenames)
dataset_indices = list(range(dataset_size))

# Shuffle the indices
np.random.shuffle(dataset_indices)

# Get the index for where we want to split the data
val_split_index = int(np.floor(0.2 * dataset_size))

# Split the list of indices into training and validation indices
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

# Get the list of filenames for the training and validation sets
train_filenames = [all_filenames[i] for i in train_idx]
val_filenames = [all_filenames[i] for i in val_idx]

# Create training and validation datasets
train_data = ImageWithCmdDataset(classes, train_filenames)
val_data = ImageWithCmdDataset(classes, val_filenames)

# Create the DataLoader
dls = DataLoaders.from_dsets(train_data, val_data)
dls = dls.cuda()

# Initialize the network
net = MyModel1()
net

# Create FastAI Learner
learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# Freeze model to train the head
learn.freeze()

# Find good learning rate
learn.lr_find()

# Train head of model
learn.fit_one_cycle(4, 0.00052)

# unfreeze to train the whole model
learn.unfreeze()
learn.lr_find()

# Fit the learner
learn.fit(10, lr=9.1e-08)

# Save the model to a given PATH
PATH = 'cmd_fai_next50.pth'
torch.save(net.state_dict(), PATH)


