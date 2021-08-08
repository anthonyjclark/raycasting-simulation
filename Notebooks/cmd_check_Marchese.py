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

# ## Code to go through dataset and ensure there are no left-right/right-left movements

# Import necessary packages
from pathlib import Path

# Set directory paths to left and right images
img_dir_left = Path("data/left")
img_dir_right = Path("data/right")

# loop through each file in "left" and check if "right" is in the name
for file_l in img_dir_left.iterdir():
    if "right" in str(file_l):
        print(file_l)
        file_l.unlink()

# loop through each file in "right" and check if "left" is in the name
for file_r in img_dir_right.iterdir():
    if "left" in str(file_r):
        print(file_r)
        file_r.unlink()


