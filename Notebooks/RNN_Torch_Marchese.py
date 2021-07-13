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

# Import necessary libraries and packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import convLSTM as convLSTM
import numpy as np
from pathlib import Path
from time import time
from PIL import Image
from RNN_classes_funcs_Marchese import *

# Ensure correct GPU is being used if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device

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

# Get list of filenames for training and validation sets
train_filenames = [all_filenames[i] for i in train_idx]
val_filenames = [all_filenames[i] for i in val_idx]
train_filenames

# Get data via custom dataset
train_data = ImageDataset(classes, train_filenames)
val_data = ImageDataset(classes, val_filenames)

# Create a dataloader for training and validation dataset
train_loader = DataLoader(dataset=train_data, shuffle=False, batch_size=32)
val_loader = DataLoader(dataset=val_data, shuffle=False, batch_size=32)

# Initialize the network
net = ConvRNN()
net.to(device)

# Initialize a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0006)

num_epochs = 45

# +
# Training the network

net.train()

for epoch in range(num_epochs):

    running_loss = 0.0
    
    start = time()
    
    for data in train_loader:
        # Get the inputs and labels; currently ommitting cmd
        img, label = data
        
        # Send data to the GPU
        img = img.to(device)
        label = label.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        output = net(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Accumulate total loss for each epoch
        running_loss += loss.item()
        
    # Print statistics
    print(f"Epoch:{epoch+1:}/{num_epochs}, Training Loss:{running_loss:0.1f}, Time:{time()-start:0.1f}s")

print('Finished Training')

# +
# Check accuracy on validation set

correct = 0
total = 0

# Variables to keep track of accuracy for each class
class_correct = [0 for _ in classes]
class_total = [0 for _ in classes]

net.eval()

with torch.no_grad():

    for data in val_loader:

        # Get the inputs and labels; currently ommitting cmd
        img, label = data
        
        # Put data into the GPU
        img = img.to(device)
        label = label.to(device)


        # Predict
        output = net(img)
        
        # Assuming we always get batches
        for i in range(output.size()[0]):
                
            # Get the predicted most probable move
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

# Save trained model
PATH = 'torch_RNN.pth'
torch.save(net.state_dict(), PATH)


