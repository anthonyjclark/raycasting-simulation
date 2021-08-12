# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt

from fastai.vision.all import *
from fastai.callback.progress import CSVLogger

# %%
# Constants (same for all trials)
VALID_PCT = 0.05
NUM_REPLICATES = 1
NUM_EPOCHS = 2# TODO: change to 8
DATASET_DIR = Path("/raid/clark/summer2021/datasets")
MODEL_PATH_REL_TO_DATASET = Path("models")
DATA_PATH_REL_TO_DATASET = Path("data")
VALID_MAZE_DIR = Path("../Mazes/validation_mazes8x8/")

# Configurable (depend on trial)
model_name = "resnet18"
dataset_name = "uniform-full"
rgb_instead_of_gray = True # TODO: not using this (would require replacing first layer)
use_pretraining = True

# Make dirs as needed
model_dir = DATASET_DIR / dataset_name / MODEL_PATH_REL_TO_DATASET
model_dir.mkdir(exist_ok=True)
print(f"Created model dir (or it already exists) : '{model_dir}'")

data_dir = DATASET_DIR / dataset_name / DATA_PATH_REL_TO_DATASET
data_dir.mkdir(exist_ok=True)
print(f"Created data dir (or it already exists)  : '{data_dir}'")

# TODO: need to take into account the replicate
rep = 1

# Derived
file_prefix = "classification-" + model_name
file_prefix += '-rgb' if rgb_instead_of_gray else '-gray'
file_prefix += '-pretrained' if use_pretraining else '-notpretrained'

model_filename = MODEL_PATH_REL_TO_DATASET / f"{file_prefix}-{rep}.pkl"
print("Model relative filename :", model_filename)
log_filename = DATA_PATH_REL_TO_DATASET / f"{file_prefix}-trainlog-{rep}.csv"
print("Log relative filename   :", log_filename)
fig_filename_prefix = data_dir / file_prefix

def get_fig_filename(label: str, ext: str, rep: int) -> str:
    fig_filename = f"{fig_filename_prefix}-{label}-{rep}.{ext}"
    print(label, "filename :", fig_filename)
    return fig_filename


# %% [markdown]
# ### [torchvision](https://github.com/fastai/fastai/blob/master/fastai/vision/models/tvm.py)
#
# - resnet18, resnet34, resnet50, resnet101, resnet152
# - squeezenet1_0, squeezenet1_1
# - densenet121, densenet169, densenet201, densenet161
# - vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
# - alexnet
#
# ### [fastai: xresnet](https://github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py)
#
# - xresnet18, xresnet34, xresnet50, xresnet101, xresnet152
# - xresnet18_deep, xresnet34_deep, xresnet50_deep
# - xresnet18_deeper, xresnet34_deeper, xresnet50_deeper
# - xse_resnet18, xse_resnet34, xse_resnet50, xse_resnet101, xse_resnet152
# - xresnext18, xresnext34, xresnext50, xresnext101
# - xse_resnext18, xse_resnext34, xse_resnext50, xse_resnext101
# - xsenet154
# - xse_resnext18_deep, xse_resnext34_deep, xse_resnext50_deep
# - xse_resnext18_deeper, xse_resnext34_deeper, xse_resnext50_deeper

# %%
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

# %%
len(compared_models)

# %% [markdown]
# ### Datasets
#
# - uniform-full
# - wondering-full

# %%
path = DATASET_DIR / dataset_name
files = get_image_files(path)


# %%
def filename_to_class(filename: str) -> str:
    angle = float(filename.split("_")[1].split(".")[0].replace("p", "."))
    if angle > 0:
        return "left"
    elif angle < 0:
        return "right"
    else:
        return "forward"


# %%
dls = ImageDataLoaders.from_name_func(path, files, filename_to_class, valid_pct=VALID_PCT)

# %%
dls.show_batch()
plt.savefig(get_fig_filename("batch", "pdf", rep))

# %%
learn = cnn_learner(dls, compared_models[model_name], metrics=accuracy, pretrained=use_pretraining, cbs=CSVLogger(fname=log_filename))

# %%
learn.path

# %%
# learn.summary()

# %%
if use_pretraining:
    learn.fine_tune(NUM_EPOCHS)
else:
    learn.fit_one_cycle(NUM_EPOCHS)

# %%
# The remove line is necessary for pickling
learn.remove_cb(CSVLogger)
learn.export(model_filename)

# %%
try:
    learn
except NameError:
    print("Loading learner from file.")
    model_path = DATASET_DIR / dataset_name / model_filename
    learn = load_learner(model_path)

# %%
learn.show_results()
plt.savefig(get_fig_filename("results", "pdf", rep))

# %%
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 10))
plt.savefig(get_fig_filename("toplosses", "pdf", rep))

# %%
interp.plot_confusion_matrix(figsize=(10, 10))
plt.savefig(get_fig_filename("confusion", "pdf", rep))

# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import HTML

import sys

sys.path.append("../PycastWorld")
sys.path.append("../Gym")

from gym_pycastworld.PycastWorldEnv import PycastWorldEnv

# %%
dls.vocab

# %%
class_to_action = {
    "left": 0,
    "forward": 1,
    "right": 2,
}

num_steps = 1000

# TODO: loop over all
maze_filename = Path(VALID_MAZE_DIR / "maze_01.txt")
env = PycastWorldEnv(maze_filename, 224, 224)

# Grab the initial observation (not used here)
observation = env.reset()
frames = [observation.copy()]

prev_class = None

for t in range(num_steps):

    with learn.no_bar():
        class_name, _, _ = learn.predict(observation)
    if (
        class_name == "left"
        and prev_class == "right"
        or class_name == "right"
        and prev_class == "left"
    ):
        class_name = "forward"
    prev_class = class_name

    action = class_to_action[class_name]
    observation, _, done, _ = env.step(action)

    frames.append(observation.copy())

    # Check if we reached the end goal
    if done:
        print(f"  Found goal in {t+1} steps")
        break

print(f"  Ended at position {env.world.x()}, {env.world.y()}")
env.close()

# TODO: collect: percent through maze and number of steps taken

# %%
fig, ax = plt.subplots()
im = ax.imshow(frames[0])


def update(frame_index):
    """Update the animation axis with data from next frame."""
    im.set_data(frames[frame_index])
    return (im,)


anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
anim.save(get_fig_filename(maze_filename.stem, "mp4", rep), fps=60)

# %%
