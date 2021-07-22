# ---
# jupyter:
#   jupytext:
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

# +
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *
from cmd_classes_funcs_Marchese import *
# -

# ## Use two functions and classes from cmd_classes_funcs_Marchese to make train/valid dataset

# Get classes and filenames
path = Path("data")
classes = get_class_labels(path)
all_filenames = get_filenames(path)

# Getting size of dataset and corresponding list of indices
dataset_size = len(all_filenames)
dataset_indices = list(range(dataset_size))

# Shuffling the indices
np.random.shuffle(dataset_indices)

# Getting index for where we want to split the data
val_split_index = int(np.floor(0.2 * dataset_size))

# Splitting list of indices into training and validation indices
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

# Getting list of filenames for training and validation set
train_filenames = [all_filenames[i] for i in train_idx]
val_filenames = [all_filenames[i] for i in val_idx]

# Create training and validation datasets
train_data = ImageWithCmdDataset(classes, train_filenames)
val_data = ImageWithCmdDataset(classes, val_filenames)

# Creating DataLoader
dls = DataLoaders.from_dsets(train_data, val_data)
dls = dls.cuda()

net = MyModel_dnet169()
net

learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# Freeze model to train the head
learn.freeze()

# Find good learning rate
learn.lr_find()

# Train head of model
learn.fit_one_cycle(4, 0.0016)

# unfreeze to train the whole model
learn.unfreeze()
learn.lr_find()

learn.fit(50, lr=9.1e-08, cbs=TrackerCallback(monitor='valid_loss', reset_on_fit=True))

# +
# maybe try to get a confusion matrix...
# -

learn.export(os.path.abspath('cmd_fai.pkl'))

PATH = 'cmd_fai_dnet169.pth'
torch.save(net.state_dict(), PATH)

learn.model

net = models.resnet18()
net


