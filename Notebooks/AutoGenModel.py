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

# !pip install -Uqq fastai

from fastai.vision.all import *

import zipfile
with zipfile.ZipFile('../Images/AutoGen.zip', 'r') as zip_ref:
    zip_ref.extractall('../Images/')

# !ls ../Images

path = Path('/home/CAMPUS/jamb2018/arcs-j/Images/AutoGen')

path.ls()

(path/'left').ls()

(path/'right').ls()

(path/'straight').ls()

# # Classification

db = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=RandomSplitter(valid_pct=0.2, seed=47),
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(do_flip=False, flip_vert=False)
)
dls = db.dataloaders(path, bs=256)

dls.show_batch()

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(50, cbs=SaveModelCallback())

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(15)

learn.export('/home/CAMPUS/jamb2018/arcs-j/Models/auto-gen-c.pkl')


# # Regression

def parent_to_deg(f):
    parent = parent_label(f)
    if parent == 'left': return 90.
    elif parent == 'right': return -90.
    else: return 0.


# +
db_r = DataBlock(
        blocks=(ImageBlock, RegressionBlock(n_out=2)),
        get_items=get_image_files,
        get_y=parent_to_deg,
        splitter=RandomSplitter(valid_pct=0.2, seed=47),
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(do_flip=False, flip_vert=False)
)

dls_r = db_r.dataloaders(path, bs=128)
# -

dls_r.show_batch()


def sin_cos_loss(preds, targs):
    rad_targs = targs / 180 * np.pi
    x_targs = torch.cos(rad_targs)
    y_targs = torch.sin(rad_targs)
    x_preds = preds[:, 0]
    y_preds = preds[:, 1]
    return ((x_preds - x_targs)**2 + (y_preds - y_targs)**2).mean()


def within_angle(preds, targs, angle):
    rad_targs = targs / 180 * np.pi
    angle_pred = torch.atan2(preds[:,1], preds[:,0])
    abs_diff = torch.abs(rad_targs - angle_pred)
    angle_diff = torch.where(abs_diff > np.pi, np.pi*2. - abs_diff, abs_diff)
    return torch.where(angle_diff < angle, 1., 0.).mean()


def within_45_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 4)


def within_30_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 6)


def within_15_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 12)


learn_r = cnn_learner(dls_r, resnet34, loss_func=sin_cos_loss, y_range=(-1, 1),
                     metrics=[within_45_deg, within_30_deg, within_15_deg])
learn_r.fine_tune(40, cbs=SaveModelCallback(monitor='valid_loss'))

learn_r.export('/home/CAMPUS/jamb2018/arcs-j/Models/auto-gen-r.pkl')

learn_r.show_results(nrows=9)


