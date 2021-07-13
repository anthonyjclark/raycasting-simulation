from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.models.xresnet import *
import torchvision.models as models


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
        self.r1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.fc2(x)
        return x

class MyModel1(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel1, self).__init__()
        self.cnn = models.resnet18(pretrained=pretrained)
        
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModel2(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel2, self).__init__()
        self.cnn = models.resnet18(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModel34(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel34, self).__init__()
        self.cnn = models.resnet34(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModel50(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel50, self).__init__()
        self.cnn = models.resnet50(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModel101(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel101, self).__init__()
        self.cnn = models.resnet101(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModelx18(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModelx18, self).__init__()
        self.cnn = xresnet18(pretrained=False)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModelx34(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModelx34, self).__init__()
        self.cnn = xresnet34(pretrained=False)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModelx50(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModelx50, self).__init__()
        self.cnn = xresnet50(pretrained=False)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x

class MyModelx101(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModelx101, self).__init__()
        self.cnn = xresnet101(pretrained=False)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x    

class MyModel_sq1(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_sq1, self).__init__()
        self.cnn = models.squeezenet1_0(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x    

class MyModel_sq1_1(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_sq1_1, self).__init__()
        self.cnn = models.squeezenet1_1(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x    

class MyModel_dnet121(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_dnet121, self).__init__()
        self.cnn = models.densenet121(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x    

class MyModel_dnet161(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_dnet161, self).__init__()
        self.cnn = models.densenet161(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x    

class MyModel_dnet169(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_dnet169, self).__init__()
        self.cnn = models.densenet169(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x        

class MyModel_dnet201(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_dnet201, self).__init__()
        self.cnn = models.densenet201(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x 

class MyModel_next50(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_next50, self).__init__()
        self.cnn = models.resnext50_32x4d(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
        x = self.fc2(x)
        return x    

class MyModel_next101(nn.Module):
    def __init__(self, pretrained=True):
        super(MyModel_next101, self).__init__()
        self.cnn = models.resnext101_32x8d(pretrained=pretrained)
        
        self.bn1 = nn.BatchNorm1d(1000)
        self.dr1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(1000 + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.dr2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        img, cmd = data
        x1 = self.cnn(img)
        x2 = cmd.unsqueeze(1)
        
        x1 = self.bn1(x1)
        x1 = self.dr1(x1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.r1(self.fc1(x))
        x = self.bn2(x)
        x = self.dr2(x)
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
