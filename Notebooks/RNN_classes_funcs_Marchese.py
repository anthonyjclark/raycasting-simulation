# Import necessary libraries and packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import convLSTM as convLSTM
from pathlib import Path
from time import time
from PIL import Image
import numpy as np

class ConvRNN(nn.Module):
    def __init__(self):
        """
        Initializes the layers of the convolutional recurrent neural network.
        """
        super().__init__()
        self.convlstm = convLSTM.ConvLSTM(3, 15, (3,3), 
                                          6, True, True, False) 
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(15*224*224, 512)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(512, 3)
        
    def forward(self, img):
        """
        Does a forward pass of the given data through the layers of the neural network.
        
        :param img: (tensor) tensor of rgb values that represent an image
        """
        _, lstm_output = self.convlstm(img)
        x = self.flat(lstm_output[0][0])
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

def get_filenames(img_dir):
    """
    Gets all the filenames in the image directory.
    
    :param img_dir: ('PosixPath' object) path to image directory
    :return: (list) list of filenames in the image directory
    """
    all_filenames = []
    if img_dir.is_dir():
        all_filenames += [item for item in img_dir.iterdir() if ".ipynb_checkpoints" not in str(item)]
    return all_filenames

def get_class_labels(img_dir):
    """
    Creates labels for dataset: ["left", "right", "straight"]. Does this through iterating 
    through the directories in img_dir and splitting on the backslash in the path name 
    (as d is data/<label>), then taking the last item of the list produced by .split 
    
    :param img_dir: ('PosixPath' object) path to image directory
    :return: (list) list of labels for dataset
    """
    # change later
    return ["left", "right", "straight"]

class ImageDataset(Dataset):
    def __init__(self, class_labels, filenames):
        """
        Creates objects for class labels, class indices, and filenames.
        
        :param class_labels: (list) a list of labels denoting different classification categories
        :param filenames: (list) a list of filenames that make up the dataset
        """
        self.class_labels = class_labels
        self.class_indices = {lbl:i for i, lbl in enumerate(self.class_labels)} 
        self.all_filenames = filenames
        
    def __len__(self):
        """
        Gives length of dataset.
        
        :return: (int) the number of filenames in the dataset
        """
        return len(self.all_filenames)

    def __getitem__(self, index):
        """
        Gets the filename associated with the given index, opens the image at
        that index, then uses the image's filename to get information associated
        with the image such as its label.
        
        :param index: (int) number that represents the location of the desired data
        :return: (tuple) tuple of all the information associated with the desired data
        """
        # The filename of the image given a specific index
        img_filename = self.all_filenames[index]
        
        # Opens image file and ensures dimension of channels included
        img = Image.open(img_filename).convert('RGB')
        # Resizes the image
        img = img.resize((224, 224))
        # Converts the image to tensor and 
        img = torch.Tensor(np.array(img)/255)
        # changes the order of the dimensions and adds a dimension
        img = img.permute(2,0,1).unsqueeze(0)
        
        # Splits up filename into components to derive info from it
        split_filename = img_filename.name.replace("-", " ").replace(".", " ").split()
        
        # Uses split_filename to get the label
        label_name = split_filename[1]
        # Given the label, gets the class_indices
        label = self.class_indices[label_name]
        
        # Returns data and the label associated with that data
        return img, label


