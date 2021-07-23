#!/usr/bin/env python

# Useful functions:
# - learn.path
# - learn.summary()

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from fastai.vision.all import *
from fastai.callback.progress import CSVLogger


# Constants (same for all trials)
VALID_PCT = 0.05
NUM_REPLICATES = 4
NUM_EPOCHS = 8
DATASET_DIR = Path("/raid/clark/summer2021/datasets")
MODEL_PATH_REL_TO_DATASET = Path("models")
DATA_PATH_REL_TO_DATASET = Path("data")
VALID_MAZE_DIR = Path("../Mazes/validation_mazes8x8/")


compared_models = {
    "resnet18": resnet18,
    "xresnet18": xresnet18,
    "xresnet18_deep": xresnet18_deep,
    "xresnet18_deeper": xresnet18_deeper,
    "xse_resnet18": xse_resnet18,
    "xresnext18": xresnext18,
    "xse_resnext18": xse_resnext18,
    "xse_resnext18_deep": xse_resnext18_deep,
    "xse_resnext18_deeper": xse_resnext18_deeper,
    "resnet50": resnet50,
    "xresnet50": xresnet50,
    "xresnet50_deep": xresnet50_deep,
    "xresnet50_deeper": xresnet50_deeper,
    "xse_resnet50": xse_resnet50,
    "xresnext50": xresnext50,
    "xse_resnext50": xse_resnext50,
    "xse_resnext50_deep": xse_resnext50_deep,
    "xse_resnext50_deeper": xse_resnext50_deeper,
    "squeezenet1_1": squeezenet1_1,
    "densenet121": densenet121,
    "densenet201": densenet201,
    "vgg11_bn": vgg11_bn,
    "vgg19_bn": vgg19_bn,
    "alexnet": alexnet,
}


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


def prepare_dataloaders(dataset_name: str, prefix: str) -> DataLoaders:

    path = DATASET_DIR / dataset_name
    files = get_image_files(path)

    dls = ImageDataLoaders.from_name_func(
        path, files, filename_to_class, valid_pct=VALID_PCT
    )

    dls.show_batch()  # type: ignore
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
    learn = cnn_learner(
        dls,
        compared_models[model_arch],
        metrics=accuracy,
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

    arg_parser = ArgumentParser("Train basic classification networks.")
    arg_parser.add_argument(
        "model_arch", help="Model architecture (see code for options)"
    )
    arg_parser.add_argument(
        "dataset_name", help="Name of dataset to use (uniform-full | wander-full)"
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

        model_filename = MODEL_PATH_REL_TO_DATASET / f"{file_prefix}-{rep}.pkl"
        print("Model relative filename :", model_filename)

        log_filename = DATA_PATH_REL_TO_DATASET / f"{file_prefix}-trainlog-{rep}.csv"
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
