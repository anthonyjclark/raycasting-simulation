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

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import os.path
from os import path

from fastai.vision.all import *
from fastai.callback.progress import CSVLogger
from torchvision import transforms
from math import pi

# Assign GPU
torch.cuda.set_device(0)
print("Running on GPU: " + str(torch.cuda.current_device()))

# Constants (same for all trials)
VALID_PCT = 0.05
NUM_REPLICATES = 4
NUM_EPOCHS = 8
DATASET_DIR = Path("/raid/clark/summer2021/datasets")
MODEL_PATH_REL_TO_DATASET = Path("regression_models")
DATA_PATH_REL_TO_DATASET = Path("regression_data")
VALID_MAZE_DIR = Path("../Mazes/validation_mazes8x8/")

compared_models = {
    "xresnext18": xresnext18,
    "alexnet": alexnet,
    "densenet121": densenet121,
}


def get_throttles(f):
    split_name = f.name.split('_')
    angle = float(split_name[1][:-4].replace("p", "."))
    if angle < 0:
        return tensor([2.5, -2.5])#torch.stack((tensor(0.),tensor(-angle)))
    elif angle > 0:
        return tensor([-2.5, 2.5])#torch.stack((tensor(angle),tensor(0.)))
    else:
        return tensor([2.5, 2.5])#torch.stack((tensor(2.5),tensor(2.5)))


def angle_metric(preds, targs):
    angle_true = targs[:, 1] - targs[:, 0]
    angle_pred = preds[:, 1] - preds[:, 0]
    return torch.where(torch.abs(angle_true - angle_pred) < 0.1, 1., 0.).mean()


def direction_metric(preds, targs):
    angle_true = targs[:, 1] - targs[:, 0]
    angle_pred = preds[:, 1] - preds[:, 0]
    return torch.where(
        torch.logical_or(
            torch.sign(angle_pred) == torch.sign(angle_true),
            torch.abs(angle_pred) < 0.1,
        ),
        1.0,
        0.0,
    ).mean()


def get_fig_filename(prefix: str, label: str, ext: str, rep: int) -> str:
    fig_filename = f"{prefix}-{label}-{rep}.{ext}"
    print(label, "filename :", fig_filename)
    return fig_filename


def prepare_dataloaders(dataset_name: str, prefix: str) -> DataLoaders:

    path = DATASET_DIR / dataset_name
    
    db = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_items=get_image_files,
    get_y=get_throttles,
    splitter=RandomSplitter(valid_pct=VALID_PCT),
    )

    dls = db.dataloaders(path, bs=64)

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
    learn = cnn_learner(
        dls,
        compared_models[model_arch],
        y_range=(-100, 100),
        metrics=[mse, angle_metric, direction_metric],
        pretrained=pretrained,
        cbs=CSVLogger(fname=logname),
    )
    
    if pretrained:
        learn.fine_tune(NUM_EPOCHS)
    else:
        learn.fit_one_cycle(NUM_EPOCHS)

    # The follwing line is necessary for pickling
    learn.remove_cb(CSVLogger)
    learn.export(modelname)

    learn.show_results()
    plt.savefig(get_fig_filename(prefix, "results", "pdf", rep))

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.savefig(get_fig_filename(prefix, "toplosses", "pdf", rep))

    interp.plot_confusion_matrix(figsize=(10, 10))
    plt.savefig(get_fig_filename(prefix, "confusion", "pdf", rep))


def main():

    arg_parser = ArgumentParser("Train regression networks.")
    arg_parser.add_argument(
        "model_arch", help="Model architecture (see code for options)"
    )
    arg_parser.add_argument(
        "dataset_name", help="Name of dataset to use (handmade-full | corrected-wander-full)"
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
