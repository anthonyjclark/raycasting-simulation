from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *


class ImageWithCmdDataset(Dataset):
    def __init__(self, class_labels, filenames):

        self.class_labels = class_labels
        self.class_indices = {lbl:i for i, lbl in enumerate(self.class_labels)}
        
        self.all_filenames = filenames
        
    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, index):
        # img_filename looks like data/<label>/<number>-<previous_move>.png
        img_filename = self.all_filenames[index]
        
        # Opens image file and converts it into a tensor
        img = Image.open(img_filename)
        #img = img.resize((128,128))
        img = torch.Tensor(np.array(img)/255)
        img = img.permute(2,0,1)
        
        # Replaces - and . with spaces then splits on the spaces
        # taking the item at index 1 which is the <previous_move>
        cmd_name = img_filename.name.replace("-", " ").replace(".", " ").split()[1]
        cmd = self.class_indices[cmd_name]
        
        # img_filename.parent takes the parent directory of img_filename
        # then that is made to be of type string, split on the backslashes
        # and the <label> in img_filename is taken
        label_name = str(img_filename.parent).split("/")[-1]
        label = self.class_indices[label_name]
        
        # Data and the label associated with that data
        return (img, cmd), label
    
class MyModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel, self).__init__()
        self.cnn = models.resnet18(pretrained=pretrained)
        
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_class_labels(img_dir):
    """
    Creates labels for dataset: ["left", "right", "straight"]. Does this through iterating 
    through the directories in img_dir and splitting on the backslash in the path name 
    (as d is data/<label>), then taking the last item of the list produced by .split 
    
    :param img_dir: ('PosixPath' object) path to image directory
    :return: (list) list of labels for dataset
    """
    return [str(d).split("/")[-1] for d in img_dir.iterdir() if d.is_dir() and ".ipynb_checkpoints" not in str(d)]

def get_filenames(img_dir):
    """
    Gets all the filenames in the image directory.
    
    :param img_dir: ('PosixPath' object) path to image directory
    :return: (list) list of filenames in the image directory
    """
    all_filenames = []
    for d in img_dir.iterdir():
        if d.is_dir():
            all_filenames += [item for item in d.iterdir() if ".ipynb_checkpoints" not in str(item)]
    return all_filenames
