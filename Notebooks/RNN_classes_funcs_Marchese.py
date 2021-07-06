import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import convLSTM as convLSTM
from pathlib import Path
from cmd_classes_funcs_Marchese import *
from time import time
from PIL import Image

class ConvRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = convLSTM.ConvLSTM(3, 10, (3,3), 1, True, True, False)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(10*240*320, 512)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(512, 3)
        
    def forward(self, img):
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

def get_classes(img_dir):
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

        self.class_labels = class_labels
        self.class_indices = {lbl:i for i, lbl in enumerate(self.class_labels)}
        
        self.all_filenames = filenames
        
    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, index):
        # img_filename looks like data/<label>/<number>-<previous_move>.png
        img_filename = self.all_filenames[index]
        
        # Opens image file and ensures dimension of channels included
        img = Image.open(img_filename).convert('RGB')
        #img = img.resize((128,128))
        # Converts image to tensor
        img = torch.Tensor(np.array(img)/255)
        img = img.permute(2,0,1).unsqueeze(0)
        
        split_filename = img_filename.name.replace("-", " ").replace(".", " ").split()
        
        # img_filename.parent takes the parent directory of img_filename
        # then that is made to be of type string, split on the backslashes
        # and the <label> in img_filename is taken
        label_name = split_filename[1]
        label = self.class_indices[label_name]
        
        # Data and the label associated with that data
        return img, label
