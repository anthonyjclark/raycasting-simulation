# ---
# jupyter:
#   jupytext:
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

# Import packages
from fastai.vision.all import *
from fastai.vision.widgets import *

# Specify output categories for NN and path to get training/validation images
move_type = 'straight', 'right', 'left'
path = Path("Images-Marchese")

# Getting images from specified path
fns = get_image_files(path)
fns

lefts = (path/'left').ls().sorted()
rights = (path/'right').ls().sorted()
straights = (path/'straight').ls().sorted()
print("Straights: " + str(len(straights)))
print("Lefts: " + str(len(lefts)))
print("Rights: " + str(len(rights)))

# Verify all images in path
failed = verify_images(fns)
failed

dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(128)) 
dls

"""# Creating DataLoader object for directions
movement = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))"""

"""# Checking the labeling of the images
dls = movement.dataloaders(path)
dls.valid.show_batch(max_n=10, nrows=2)"""

# Training the model
learn = cnn_learner(dls, resnet18, loss_func=CrossEntropyLossFlat(), metrics=error_rate)
learn.fine_tune(4)

# Seeing where the model may have gotten confused in classification
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Looking at the images that most confused the NN
interp.plot_top_losses(3, nrows=3)

# Loading in GUI that helps clean the data set
cleaner = ImageClassifierCleaner(learn)
cleaner

# +
# Delete specified images in cleaner
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
    
# Move specified images in cleaner
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
# -

# Export model for future use
learn.export(os.path.abspath('Move_NN.pkl'))

learn.model


