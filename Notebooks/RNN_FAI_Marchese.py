# ---
# jupyter:
#   jupytext:
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

# +
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.models as models
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *
from RNN_classes_funcs_Marchese import *
# -

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device

# Get classes and filenames
path = Path("data_RNN")
classes = get_classes(path)
all_filenames = get_filenames(path)
all_filenames.sort()

# Getting size of dataset and corresponding list of indices
dataset_size = len(all_filenames)
dataset_indices = list(range(dataset_size))

# Getting index for where we want to split the data
val_split_index = int(np.floor(0.2 * dataset_size))

# Splitting list of indices into training and validation indices
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

# Getting list of filenames for training and validation set
train_filenames = [all_filenames[i] for i in train_idx]
val_filenames = [all_filenames[i] for i in val_idx]
train_filenames

# Getting data via custom dataset
train_data = ImageDataset(classes, train_filenames)
val_data = ImageDataset(classes, val_filenames)

# Creating DataLoader
dls = DataLoaders.from_dsets(train_data, val_data, bs=8, shuffle=False)
dls = dls.cuda()

net = ConvRNN()
net.to(device)

learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# Find good learning rate
learn.lr_find()

learn.fit(20, lr=4.36e-06)

PATH = 'fai_RNN.pth'
torch.save(net.state_dict(), PATH)


