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
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *
from cmd_classes_funcs_Marchese import *
# -

# Make sure we're running on the server's GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device

# Get classes and filenames
path = Path("data")
classes = get_class_labels(path)
all_filenames = get_filenames(path)

# ## Splitting Data into train/validate sets

# Getting size of dataset and corresponding list of indices
dataset_size = len(all_filenames)
dataset_indices = list(range(dataset_size))

# Shuffling the indices
np.random.shuffle(dataset_indices)

# Getting index for where we want to split the data
val_split_index = int(np.floor(0.2 * dataset_size))

# Splitting list of indices into training and validation indices
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

# Creating samplers
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# Getting list of filenames for training and validation set
train_filenames = [all_filenames[i] for i in train_idx]
val_filenames = [all_filenames[i] for i in val_idx]

# Create training and validation datasets
train_data = ImageWithCmdDataset(classes, train_filenames)
val_data = ImageWithCmdDataset(classes, val_filenames)

train_loader = DataLoader(dataset=train_data, shuffle=False, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(dataset=val_data, shuffle=False, batch_size=16, sampler=val_sampler)

# Instantiate MyModel class
net = MyModel_dnet201()

# Send model to GPU
net.to(device)

n = dataset_size
w = torch.tensor([(n-138)/n,(n-211)/n,(n-786)/n])
w = w.to(device)
# defining loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=w)
optimizer = optim.Adam(net.parameters(), lr=0.0000095)

num_epochs = 40

from time import time

# +
# Model training

net.train()

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    
    start = time()
    
    for data in train_loader:
        # Get the inputs and labels
        inp_data, label = data
        
        # Break up the inputs
        img, cmd = inp_data
        
        # Putting data into the GPU
        img = img.to(device)
        cmd = cmd.to(device)
        label = label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        output = net((img, cmd))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print(f"Epoch:{epoch+1:}/{num_epochs}, Training Loss:{running_loss:0.1f}, Time:{time()-start:0.1f}s")

print('Finished Training')

# +
# Checking accuracy on validation set

correct = 0
total = 0

# Variables to keep track of accuracy for each class
class_correct = [0 for _ in classes]
class_total = [0 for _ in classes]

net.eval()

with torch.no_grad():

    for data in val_loader:

        # Get the inputs and label data
        inp_data, label = data
        
        # Break up the inputs
        img, cmd = inp_data
        
        # Putting data into the GPU
        img = img.to(device)
        cmd = cmd.to(device)
        label = label.to(device)


        # Predict
        output = net((img, cmd))
        
        # Assuming we always get batches
        for i in range(output.size()[0]):
                
            # Getting the predicted most probable move
            move = torch.argmax(output[i])
                
            if move == label[i]:
                class_correct[label[i]] += 1
                class_total[label[i]] += 1
                correct +=1
            else:
                class_total[label[i]] += 1
            total += 1
        
# Calculate and output total set accuracy 
accuracy = correct / total
print(f"Accuracy on validation set: {correct}/{total} = {accuracy*100:.2f}%")

# Calculate and show accuracy for each class
for i, cls in enumerate(classes):
    ccorrect = class_correct[i]
    ctotal = class_total[i]
    caccuracy = ccorrect / ctotal
    print(f"  Accuracy on {cls:>5} class: {ccorrect}/{ctotal} = {caccuracy*100:.2f}%")
# -

PATH = 'cmd_torch_dnet201.pth'
torch.save(net.state_dict(), PATH)

data, labels = next(iter(val_loader))
labels.shape

plt.imshow(data[0][0].permute(1,2,0))

labels[0]

net.eval()
img = data[0][0]
cmd = data[1][0]
img = img.to(device)
cmd = cmd.to(device)
net((img.unsqueeze(0), cmd.unsqueeze(0)))


