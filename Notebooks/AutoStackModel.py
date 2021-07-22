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
from torchvision import transforms

import zipfile
with zipfile.ZipFile('/home/CAMPUS/jamb2018/arcs-j/Images/AutoStack.zip', 'r') as zip_ref:
    zip_ref.extractall('/home/CAMPUS/jamb2018/arcs-j/Images')

path = Path('/home/CAMPUS/jamb2018/arcs-j/Images/AutoStack')

path.ls()


# # Stacking images vertically

def get_pair(o):
    curr_im_num = Path(o).name[:5]
    if not int(curr_im_num):
        prev_im_num = curr_im_num
    else:
        prev_im_num = int(curr_im_num)-1
    
    prev_im = None
    for item in Path(o).parent.ls():
        if int(item.name[:5]) == prev_im_num:
            prev_im = item
    if prev_im is None:
        prev_im = Path(o)
        
    assert prev_im != None
    
    img1 = Image.open(o).convert('RGB')
    img2 = Image.open(prev_im).convert('RGB')
    img1_t = transforms.ToTensor()(img1).unsqueeze_(0)
    img2_t = transforms.ToTensor()(img2).unsqueeze_(0)
    
    new_shape = list(img1_t.shape)
    new_shape[-2] = new_shape[-2] * 2
    img3_t = torch.zeros(new_shape)

    img3_t[:, :, :240, :] = img1_t
    img3_t[:, :, 240:, :] = img2_t
    
    img3 = transforms.ToPILImage()(img3_t.squeeze_(0))
    
    return np.array(img3)


get_pair(path/'00001_straight.png').shape

Image.fromarray(get_pair(path/'00001_straight.png'))

# ## Classification 1

db = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_x=get_pair,
        get_y=lambda x: x.name[6:-4],
        splitter=RandomSplitter(valid_pct=0.2, seed=47)
)

dls = db.dataloaders(path, bs=64)
dls.show_batch()

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(50, cbs=[SaveModelCallback(), 
                           EarlyStoppingCallback(monitor='valid_loss',
                                                 patience=10)])

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(15)

learn.export('/home/CAMPUS/jamb2018/arcs-j/Models/auto-stack-c1.pkl')


# ## Regression 1

def name_to_deg(f):
    label = f.name[6:-4]
    if label == 'left': return 90.
    elif label == 'right': return -90.
    else: return 0.


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


# +
db_r = DataBlock(
        blocks=(ImageBlock, RegressionBlock(n_out=2)),
        get_items=get_image_files,
        get_x=get_pair,
        get_y=name_to_deg,
        splitter=RandomSplitter(valid_pct=0.2, seed=47)
)

dls_r = db_r.dataloaders(path, bs=128)
# -

dls_r.show_batch()

learn_r = cnn_learner(dls_r, resnet34, loss_func=sin_cos_loss, y_range=(-1, 1),
                     metrics=[within_45_deg, within_30_deg, within_15_deg])
learn_r.fine_tune(50, cbs=[SaveModelCallback(), 
                           EarlyStoppingCallback(monitor='valid_loss',
                                                 patience=10)])


# # Stacking Images by Depth

def get_first(o):
    img1 = Image.open(o)
    return np.array(img1)


# +
def get_second(o):
    curr_im_num = Path(o).name[:5]
    if not int(curr_im_num):
        prev_im_num = curr_im_num
    else:
        prev_im_num = int(curr_im_num)-1
    
    prev_im = None
    for item in Path(o).parent.ls():
        if int(item.name[:5]) == prev_im_num:
            prev_im = item
    assert prev_im != None
    
#     img1 = Image.open(o).convert('RGB')
    img2 = Image.open(prev_im).convert('RGB')
#     img1_arr = np.array(img1)
    return np.array(img2)


# -

def get_pair_2(o):
    curr_im_num = Path(o).name[:5]
    if not int(curr_im_num):
        prev_im_num = curr_im_num
    else:
        prev_im_num = int(curr_im_num)-1
    
    prev_im = None
    for item in Path(o).parent.ls():
        if int(item.name[:5]) == prev_im_num:
            prev_im = item
    if prev_im is None:
        prev_im = Path(o)
    assert prev_im != None
    
    img1 = Image.open(o).convert('RGB')
    img2 = Image.open(prev_im).convert('RGB')
    img1_arr = np.array(img1, dtype=np.uint8)
    img2_arr = np.array(img2, dtype=np.uint8)
        
    new_shape = list(img1_arr.shape)
    new_shape[-1] = new_shape[-1] * 2
    img3_arr = np.zeros(new_shape, dtype=np.uint8)

    img3_arr[:, :, :3] = img1_arr
    img3_arr[:, :, 3:] = img2_arr
    
    return img3_arr.T.astype(np.float32)


test_out = get_pair_2(path/'00001_straight.png')
test_out.shape

# ## Classification 2

db_2 = DataBlock(
        blocks=((ImageBlock, ImageBlock), CategoryBlock),
        get_items=get_image_files,
        get_x=get_pair_2,
        get_y=lambda x: x.name[6:-4],
        splitter=RandomSplitter(valid_pct=0.2, seed=47)
)
dls_2 = db_2.dataloaders(path, bs=64)

learn_2 = cnn_learner(dls_2, resnet34, metrics=error_rate)

learn_2.model

learn_2.model[0][0] = nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

learn_2.model

learn_2.fine_tune(50, cbs=[SaveModelCallback(), 
                           EarlyStoppingCallback(monitor='valid_loss',
                                                 patience=10)])

interp_2 = ClassificationInterpretation.from_learner(learn_2)
interp.plot_confusion_matrix()
interp.plot_top_losses(15)


# ## Regression 2

def get_pair_2_int(o):
    curr_im_num = Path(o).name[:5]
    if not int(curr_im_num):
        prev_im_num = curr_im_num
    else:
        prev_im_num = int(curr_im_num)-1
    
    prev_im = None
    for item in Path(o).parent.ls():
        if int(item.name[:5]) == prev_im_num:
            prev_im = item
    if prev_im is None:
        prev_im = Path(o)
    assert prev_im != None
    
    img1 = Image.open(o).convert('RGB')
    img2 = Image.open(prev_im).convert('RGB')
    img1_arr = np.array(img1, dtype=np.uint8)
    img2_arr = np.array(img2, dtype=np.uint8)
        
    new_shape = list(img1_arr.shape)
    new_shape[-1] = new_shape[-1] * 2
    img3_arr = np.zeros(new_shape, dtype=np.uint8)

    img3_arr[:, :, :3] = img1_arr
    img3_arr[:, :, 3:] = img2_arr
    
    return img3_arr


# +
# functions defined for model required by fastai
def parent_to_deg(f):
    parent = parent_label(f)
    if parent == "left":
        return 90.0
    elif parent == "right":
        return -90.0
    else:
        return 0.0


def sin_cos_loss(preds, targs):
    rad_targs = targs / 180 * np.pi
    x_targs = torch.cos(rad_targs)
    y_targs = torch.sin(rad_targs)
    x_preds = preds[:, 0]
    y_preds = preds[:, 1]
    return ((x_preds - x_targs) ** 2 + (y_preds - y_targs) ** 2).mean()


def within_angle(preds, targs, angle):
    rad_targs = targs / 180 * np.pi
    angle_pred = torch.atan2(preds[:, 1], preds[:, 0])
    abs_diff = torch.abs(rad_targs - angle_pred)
    angle_diff = torch.where(abs_diff > np.pi, np.pi * 2.0 - abs_diff, abs_diff)
    return torch.where(angle_diff < angle, 1.0, 0.0).mean()


def within_45_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 4)


def within_30_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 6)


def within_15_deg(preds, targs):
    return within_angle(preds, targs, np.pi / 12)


def name_to_deg(f):
    label = f.name[6:-4]
    if label == "left":
        return 90.0
    elif label == "right":
        return -90.0
    else:
        return 0.0


def get_label(o):
    return o.name[6:-4]


def get_pair_2(o):
    curr_im_num = Path(o).name[:5]
    if not int(curr_im_num):
        prev_im_num = curr_im_num
    else:
        prev_im_num = int(curr_im_num) - 1

    prev_im = None
    for item in Path(o).parent.ls():
        if int(item.name[:5]) == prev_im_num:
            prev_im = item
    if prev_im is None:
        prev_im = Path(o)
    assert prev_im != None

    img1 = Image.open(o).convert("RGB")
    img2 = Image.open(prev_im).convert("RGB")
    img1_arr = np.array(img1, dtype=np.uint8)
    img2_arr = np.array(img2, dtype=np.uint8)

    new_shape = list(img1_arr.shape)
    new_shape[-1] = new_shape[-1] * 2
    img3_arr = np.zeros(new_shape, dtype=np.uint8)

    img3_arr[:, :, :3] = img1_arr
    img3_arr[:, :, 3:] = img2_arr

    return img3_arr.T.astype(np.float32)


# helper functions
def stacked_input(prev_im, curr_im):
    if prev_im is None:
        prev_im = curr_im

    new_shape = list(curr_im.shape)
    new_shape[-1] = new_shape[-1] * 2
    stacked_im = np.zeros(new_shape, dtype=np.uint8)

    stacked_im[:, :, :3] = curr_im
    stacked_im[:, :, 3:] = prev_im

    return stacked_im.T.astype(np.float32)


def reg_predict(pred_coords):
#     print(f"type: {type(pred_coords[1])} ")
#     print(f"pred_coord[1]: {pred_coords} ")
    pred_angle = np.arctan2(pred_coords[1], pred_coords[0]) / np.pi * 180
    pred_angle = pred_angle % (360)

    if pred_angle > 53 and pred_angle <= 180:
        return "left"
    elif pred_angle > 180 and pred_angle < 307:
        return "right"
    else:
        return "straight"



# -

l = load_learner('../Models/auto-stack-c3.pkl')

l.get_preds()

test_out = get_pair_2_int(path/'00001_straight.png')
test_out.shape

db_r_2 = DataBlock(
        blocks=((ImageBlock, ImageBlock), RegressionBlock(n_out=2)),
        get_items=get_image_files,
        get_x=get_pair_2,
        get_y=name_to_deg,
        splitter=RandomSplitter(valid_pct=0.2, seed=47)
)
dls_r_2 = db_r_2.dataloaders(path, bs=128)

learn_r_2 = cnn_learner(dls_r_2, resnet34, loss_func=sin_cos_loss, y_range=(-1, 1),
                     metrics=[within_45_deg, within_30_deg, within_15_deg])


learn_r_2.model[0][0] = nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

learn_r_2.fine_tune(50, cbs=[SaveModelCallback(monitor='valid_loss'), 
                            EarlyStoppingCallback(monitor='valid_loss',
                                                 patience=10)])

learn_r_2.export('/home/CAMPUS/jamb2018/arcs-j/Models/auto-stack-r.pkl')
