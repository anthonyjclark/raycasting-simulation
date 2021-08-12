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

import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from os import path

from fastai.vision.all import *
from fastai.callback.progress import CSVLogger
from torch.utils.data import Dataset

sys.path.append("../Notebooks")
import convLSTM as convLSTM

# Assign GPU
torch.cuda.set_device(3)
print("Running on GPU: " + str(torch.cuda.current_device()))

# Constants (same for all trials)
VALID_PCT = 0.05
NUM_REPLICATES = 4
NUM_EPOCHS = 8
DATASET_DIR = Path("/raid/clark/summer2021/datasets")
MODEL_PATH_REL_TO_DATASET = Path("RNN_models")
DATA_PATH_REL_TO_DATASET = Path("RNN_data")
VALID_MAZE_DIR = Path("../Mazes/validation_mazes8x8/")


def get_fig_filename(prefix: str, label: str, ext: str, rep: int) -> str:
    fig_filename = f"{prefix}-{label}-{rep}.{ext}"
    print(label, "filename :", fig_filename)
    return fig_filename


def filename_to_class(filename: str) -> str:
    angle = float(filename.split("_")[1].split(".")[0].replace("p", "."))
    if angle > 0:
        return "left"
    elif angle < 0:
        return "right"
    else:
        return "forward"


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
        img = img.unsqueeze(1)
        _, lstm_output = self.convlstm(img)
        x = self.flat(lstm_output[0][0])
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


def prepare_dataloaders(dataset_name: str, prefix: str) -> DataLoaders:

    path = DATASET_DIR / dataset_name
    files = get_image_files(path)
    files.sort()

    dls = ImageDataLoaders.from_name_func(
        path, files, filename_to_class, valid_pct=VALID_PCT, shuffle=False, item_tfms=Resize(224)
    )

    dls.show_batch()  # type: ignore
    plt.savefig(get_fig_filename(prefix, "batch", "pdf", 0))

    return dls  # type: ignore


def train_model(
    dls: DataLoaders,
    model_arch: str,
    logname: Path,
    modelname: Path,
    prefix: str,
    rep: int,
):
    net = ConvRNN()
    
    learn = Learner(
        dls,
        net,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        cbs=CSVLogger(fname=logname),
    )

    learn.fit_one_cycle(NUM_EPOCHS)

    # The following line is necessary for pickling
    learn.remove_cb(CSVLogger)
    # Save trained model
    learn.export(modelname)
    
    learn.show_results()
    plt.savefig(get_fig_filename(prefix, "results", "pdf", rep))

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.savefig(get_fig_filename(prefix, "toplosses", "pdf", rep))

    interp.plot_confusion_matrix(figsize=(10, 10))
    plt.savefig(get_fig_filename(prefix, "confusion", "pdf", rep))


def main():

    dataset_name = 'corrected-wander-full'
    model_arch = "RNN"

    # Make dirs as needed
    model_dir = DATASET_DIR / dataset_name / MODEL_PATH_REL_TO_DATASET
    model_dir.mkdir(exist_ok=True)
    print(f"Created model dir (or it already exists) : '{model_dir}'")

    data_dir = DATASET_DIR / dataset_name / DATA_PATH_REL_TO_DATASET
    data_dir.mkdir(exist_ok=True)
    print(f"Created data dir (or it already exists)  : '{data_dir}'")

    file_prefix = "classification-" + model_arch
    # file_prefix += "-rgb" if rgb_instead_of_gray else "-gray"
    fig_filename_prefix = data_dir / file_prefix

    dls = prepare_dataloaders(dataset_name, fig_filename_prefix)

    # Train NUM_REPLICATES separate instances of this model and dataset
    for rep in range(NUM_REPLICATES):
        
        model_filename = DATASET_DIR / dataset_name / MODEL_PATH_REL_TO_DATASET / f"{file_prefix}-{rep}.plk"
        print("Model relative filename :", model_filename)

        # Checks if model exists and skip if it does (helps if this crashes)
        if path.exists(model_filename):
            continue

        log_filename = DATASET_DIR / dataset_name / DATA_PATH_REL_TO_DATASET / f"{file_prefix}-trainlog-{rep}.csv"
        print("Log relative filename   :", log_filename)

        train_model(
            dls,
            model_arch,
            log_filename,
            model_filename,
            fig_filename_prefix,
            rep,
        )


if __name__ == "__main__":
    main()
