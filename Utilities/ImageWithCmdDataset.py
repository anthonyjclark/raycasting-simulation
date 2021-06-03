# TODO:
# - split validation and training
# - read images


"""
--- If these are the commands taken by AutoGen

1. straight
2. left
3. left
4. straight
5. left
6. straight
7. right
.
.
.


--- Then this is the resulting dataset

testdata
├── left
│   ├── 02-straight.png
│   ├── 03-left.png
│   └── 05-straight.png
├── right
│   └── 07-straight.png
└── straight
    ├── 01-straight.png
    ├── 04-left.png
    └── 06-left.png
"""


from pathlib import Path

from torch.utils.data import DataLoader, Dataset


class ImageWithCmdDataset(Dataset):
    def __init__(self, img_dir: Path):

        # Should be ["left", "right", "straight"]
        # TODO: this returns things like PosixPath('testdata/right') and
        # we want this to be 'right'
        self.class_labels = [d for d in img_dir.iterdir() if d.is_dir()]

        # Get all filenames
        self.all_filenames = []
        for d in img_dir.iterdir():
            if d.is_dir():
                self.all_filenames += list(d.iterdir())

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, index):
        img_filename = self.all_filenames[index]

        # TODO: don't know how to actually open the file
        # img = PIL.open(img_filename)

        print(img_filename)

        img = ""
        cmd = img_filename.name.replace("-", " ").replace(".", " ").split()[1]
        label = str(img_filename.parent).split("/")[-1]

        return (img, cmd), label


dataset = ImageWithCmdDataset(Path("testdata"))
# print(dataset.class_labels)
# print(dataset.all_filenames)
# print(len(dataset.all_filenames))

(img, cmd), label = dataset[0]

print(f"img: {img}")
print(f"cmd: {cmd}")
print(f"label: {label}")

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# 1. Create dataset
# 2. Use SubsetRandomSampler
# 3. Create dataloaders
# 4. Pass to fastai


# https://github.com/anthonyjclark/cs152spring2021/blob/main/l20-convolutional-neural-networks/l20-cnns.ipynb
