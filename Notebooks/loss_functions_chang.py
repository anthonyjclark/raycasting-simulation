# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

from fastai.vision.all import *
from fastbook import *
import datetime;
from fastai.vision.widgets import *
import torch

torch.cuda.set_device(1)
torch.cuda.current_device()

path = Path('/raid/Images/40_mazes_05-06-2021_23-27')

name = "blah"
os.system(f"mkdir {name}-diagnostics")
save_path = os.path.abspath(f"{name}-diagnostics")

plt.figure(figsize=(15, 10)) 
plt.savefig(os.path.join(save_path, f"completionchart_test.png"))

# ns = !ls -l '/raid/Images/40_mazes_05-06-2021_23-27/straight' | wc -l
ns = int(ns[0])

ns

# nl = !ls -l '/raid/Images/40_mazes_05-06-2021_23-27/left' | wc -l
nl = int(nl[0])

nl

# nr = !ls -l '/raid/Images/40_mazes_05-06-2021_23-27/right' | wc -l
nr = int(nr[0])

nr

# # Make Proxy Dataset

# +
# calculate propotions
total = ns + nl + nr
# proportion of lefts images: 
left_proportion = nl/total
# proportion of right images:
right_proportion = nr/total

lefts = (path/'left').ls().sorted()
rights = (path/'right').ls().sorted()
straights = (path/'straight').ls().sorted()

# Left subset size 
left_subset_len = int(5000*left_proportion)
# Right subset size
right_subset_len = int(5000*right_proportion)
# Straight subsetsize
straight_subset_len = 5000 - (910 + 917)

lefts = np.random.choice(lefts, left_subset_len)
rights = np.random.choice(rights, right_subset_len)
straights = np.random.choice(straights, straight_subset_len)
# -

# !ls -l '/raid/Images/proxydata/right' | wc -l

# !ls -l '/raid/Images/proxydata/straight' | wc -l

# !ls -l '/raid/Images/proxydata/left' | wc -l

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

torch.cuda.current_device()

path = Path('/raid/Images/proxydata/')

dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(224)) 

dls.n

n_straight = 0
n_left = 0
n_right = 0
for p in dls.items:
    if "straight" in str(p):
        n_straight += 1
    if "left" in str(p):
        n_left += 1
    if "right" in str(p):
        n_right += 1
n_left, n_right, n_straight

dls.show_batch()

# ### Try Learning without Weights

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(5, cbs=SaveModelCallback())

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(15)

# ### With Weights

cuda = torch.device("cuda")
n_samples = 889 + 3081 + 889
weights = n_samples / (3 * np.array([889, 889, 3081]))
# weights = torch.tensor([n_left, n_right, n_straight], device=cuda)
weights = torch.tensor(weights, device=cuda)
weights = weights.type(torch.float)
weights

learn2 = cnn_learner(dls, resnet34, metrics=error_rate, loss_func = CrossEntropyLossFlat(weight = weights))
learn2.fine_tune(5, cbs=SaveModelCallback())

interp2 = ClassificationInterpretation.from_learner(learn2)
interp2.plot_confusion_matrix()

interp2.plot_top_losses(9)

interp2.print_classification_report()

interp2.most_confused()

# # Data Cleaning

# ### Before: Copy Directory in Terminal

# ### First Clean for Non-Weight Learner

rm_files = []
for indx in interp.top_losses(20).indices:
    print(dls.valid_ds.items[indx.item()])

path = Path('/raid/Images/proxy_clean')

import os
os.path.abspath("/raid/Images/proxy_clean")

os.system('mkdir test')

path = Path('/raid/Images')

for v in interp.top_losses(20).values:
    print(v)

cleaner = ImageClassifierCleaner(learn)
cleaner

for idx in cleaner.delete(): cleaner.fns[idx].unlink()

for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)

dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(224)) 

dls.show_batch()

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(5, cbs=SaveModelCallback())

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(15)

# +
# num_s = !ls -l '/raid/Images/proxydata/straight' | wc -l
# num_l = !ls -l '/raid/Images/proxydata/left' | wc -l
# num_r = !ls -l '/raid/Images/proxydata/right' | wc -l

num_s = int(num_s[0])
num_l = int(num_l[0])
num_r = int(num_r[0])

num_s, num_l, num_r
# -

cuda = torch.device("cuda")
n_samples = num_s + num_l + num_r
weights = n_samples / (3 * np.array([num_l, num_r, num_s]))
# weights = torch.tensor([n_left, n_right, n_straight], device=cuda)
weights = torch.tensor(weights, device=cuda)
weights = weights.type(torch.float)
weights

learn2 = cnn_learner(dls, resnet34, metrics=error_rate, loss_func = CrossEntropyLossFlat(weight = weights))
learn2.fine_tune(5, cbs=SaveModelCallback())

# ### Note: Scikit learn's weights on dataset did not produce better matrix

learn.export('/home/CAMPUS/eoca2018/raycasting-simulation/Models/proxy_model.pkl')

# # Clearning Larger Dataset

path = Path('/raid/Images/40_mazes_05-06-2021_23-27/')
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(224)) 

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, cbs=SaveModelCallback())

cleaner = ImageClassifierCleaner(learn)
cleaner

# # Applying Weights to Larger Dataset

path = Path('/raid/Images/40_mazes_05-06-2021_23-27/')

dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(224)) 

dls.n

n_straight = 0
n_left = 0
n_right = 0
for p in dls.items:
    if "straight" in str(p):
        n_straight += 1
    if "left" in str(p):
        n_left += 1
    if "right" in str(p):
        n_right += 1
n_left, n_right, n_straight

cuda = torch.device('cuda')   
n_samples = ns + nl + nr
weights = n_samples / (3 * np.array([nl, nr, ns]))
weights = torch.tensor([nl, nr, ns], device = cuda)
weights = weights.type(torch.float)
weights

cuda = torch.device('cuda')   
n_samples = ns + nl + nr
weights = n_samples / (3 * np.array([nl, nr, ns]))
weights = torch.tensor([nl, nr, ns], device = cuda)
weights = weights.type(torch.float)
weights

learn = cnn_learner(dls, resnet34, metrics=error_rate, loss_func = CrossEntropyLossFlat(weight = weights))
learn.fine_tune(1, cbs=SaveModelCallback())

learn = cnn_learner(dls, xresnet34, metrics=error_rate, loss_func = CrossEntropyLossFlat(weight = weights))
learn.fine_tune(5, cbs=SaveModelCallback())

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(15)

interp.print_classification_report()

interp.most_confused()

# # Without Weights

learn = cnn_learner(dls, xresnet34, metrics=error_rate)
learn.fine_tune(1, cbs=SaveModelCallback())

# # Experimenting Loss Functions

learn = unet_learner(dls, resnet18, metrics=error_rate, loss_func = FocalLossFlat(gamma = 2))
learn.fine_tune(1, cbs=SaveModelCallback())

import torch
torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


learn = cnn_learner(dls, resnet34, metrics=error_rate, loss_func = FocalLoss(gamma=2))
learn.fine_tune(1, cbs=SaveModelCallback())
