# ---
# jupyter:
#   jupytext:
#     formats: py:light
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
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import os.path
from os import path

from fastai.vision.all import *
from fastai.callback.progress import CSVLogger
from torch.utils.data import Dataset
# -

# Assign GPU
torch.cuda.set_device(2)
print("Running on GPU: " + str(torch.cuda.current_device()))

# Constants (same for all trials)
VALID_PCT = 0.05
NUM_REPLICATES = 1
NUM_EPOCHS = 1
DATASET_DIR = Path("/raid/clark/summer2021/datasets")
MODEL_PATH_REL_TO_DATASET = Path("cmd_models")
DATA_PATH_REL_TO_DATASET = Path("cmd_data")
VALID_MAZE_DIR = Path("../Mazes/validation_mazes8x8/")

compared_models = {
    "resnet18": resnet18
}


def filename_to_class(filename: str) -> str:
    angle = float(str(filename).split("_")[1].split(".")[0].replace("p", "."))
    if angle > 0:
        return "left"
    elif angle < 0:
        return "right"
    else:
        return "forward"


class ImageWithCmdDataset(Dataset):
    def __init__(self, filenames):
        """
        Creates objects for class labels, class indices, and filenames.
        
        :param filenames: (list) a list of filenames that make up the dataset
        """
        self.class_labels = ['left', 'forward', 'right']
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
        with the image such as its label and the label of the previous image.
        
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
        # changes the order of the dimensions
        img = img.permute(2,0,1)
        
        # Getting the current image's label
        label_name = filename_to_class(img_filename)
        label = self.class_indices[label_name]
        
        # Getting the previous image's label
        # The default is 'forward'
        cmd_name = 'forward'
        
        # If the index is not 0, the cmd is determined by the previous img_filename
        if index != 0:
            prev_img_filename = self.all_filenames[index-1]
            cmd_name = filename_to_class(prev_img_filename)            
        cmd = self.class_indices[cmd_name]
        
        # Data and the label associated with that data
        return (img, cmd), label


class cmd_model(nn.Module):
    def __init__(self, arch: str, pretrained: bool):
        super(cmd_model, self).__init__()
        self.cnn = arch(pretrained=pretrained)
        
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


def get_fig_filename(prefix: str, label: str, ext: str, rep: int) -> str:
    fig_filename = f"{prefix}-{label}-{rep}.{ext}"
    print(label, "filename :", fig_filename)
    return fig_filename


def prepare_dataloaders(dataset_name: str, prefix: str) -> DataLoaders:

    path = DATASET_DIR / dataset_name
    files = get_image_files(path)
    
    # Get size of dataset and corresponding list of indices
    dataset_size = len(files)
    dataset_indices = list(range(dataset_size))
    
    # Shuffle the indices
    np.random.shuffle(dataset_indices)

    # Get the index for where we want to split the data
    val_split_index = int(np.floor(VALID_PCT * dataset_size))
    
    # Split the list of indices into training and validation indices
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    
    # Get the list of filenames for the training and validation sets
    train_filenames = [files[i] for i in train_idx]
    val_filenames = [files[i] for i in val_idx]
    
    # Create training and validation datasets
    train_data = ImageWithCmdDataset(train_filenames)
    val_data = ImageWithCmdDataset(val_filenames)
    
    # Get DataLoader
    dls = DataLoaders.from_dsets(train_data, val_data)
    dls = dls.cuda()

    #dls.show_batch()  # type: ignore
    plt.savefig(get_fig_filename(prefix, "batch", "pdf", 0))

    return dls  # type: ignore


def train_model(
    dls: DataLoaders,
    model_arch: str,
    pretrained: bool,
    logname: Path,
    modelname: Path,
    prefix: str,
    rep: int,
):
    arch = compared_models[model_arch]
    net = cmd_model(arch, pretrained=pretrained)
    
    learn = Learner(
        dls,
        net,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        cbs=CSVLogger(fname=logname),
    )

    if pretrained:
        learn.fine_tune(NUM_EPOCHS)
    else:
        learn.fit_one_cycle(NUM_EPOCHS)

    # Save trained model
    torch.save(net.state_dict(), modelname)


def main():

    arg_parser = ArgumentParser("Train cmd classification networks.")
    arg_parser.add_argument(
        "model_arch", help="Model architecture (see code for options)"
    )
    arg_parser.add_argument(
        "dataset_name", help="Name of dataset to use (corrected-wander-full)"
    )
    arg_parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )

    args = arg_parser.parse_args()

    # TODO: not using this (would require replacing first layer)
    # rgb_instead_of_gray = True

    # Make dirs as needed
    model_dir = DATASET_DIR / args.dataset_name / MODEL_PATH_REL_TO_DATASET
    model_dir.mkdir(exist_ok=True)
    print(f"Created model dir (or it already exists) : '{model_dir}'")

    data_dir = DATASET_DIR / args.dataset_name / DATA_PATH_REL_TO_DATASET
    data_dir.mkdir(exist_ok=True)
    print(f"Created data dir (or it already exists)  : '{data_dir}'")

    file_prefix = "classification-" + args.model_arch
    # file_prefix += "-rgb" if rgb_instead_of_gray else "-gray"
    file_prefix += "-pretrained" if args.pretrained else "-notpretrained"
    fig_filename_prefix = data_dir / file_prefix

    dls = prepare_dataloaders(args.dataset_name, fig_filename_prefix)

    # Train NUM_REPLICATES separate instances of this model and dataset
    for rep in range(NUM_REPLICATES):
        
        model_filename = DATASET_DIR / args.dataset_name / MODEL_PATH_REL_TO_DATASET / f"{file_prefix}-{rep}.pth"
        print("Model relative filename :", model_filename)

        # Checks if model exists and skip if it does (helps if this crashes)
        if path.exists(model_filename):
            continue

        log_filename = DATASET_DIR / args.dataset_name / DATA_PATH_REL_TO_DATASET / f"{file_prefix}-trainlog-{rep}.csv"
        print("Log relative filename   :", log_filename)

        train_model(
            dls,
            args.model_arch,
            args.pretrained,
            log_filename,
            model_filename,
            fig_filename_prefix,
            rep,
        )


if __name__ == "__main__":
    main()
