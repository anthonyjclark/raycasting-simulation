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

# Import libraries and packages
from fastai.vision.all import *
from fastai.vision.widgets import *
import torchvision.models as models
import re
import seaborn as sns
import sys

# Set GPU
torch.cuda.set_device(1)
torch.cuda.current_device()

# Specify possible move types
move_type = 'left', 'right', 'straight'

# Different model architectures to train
model_archs = [models.resnet18, models.resnet34, models.resnet50, models.resnet101, xresnet18, xresnet34, xresnet50, xresnet101,
              models.squeezenet1_0, models.squeezenet1_1, models.densenet121, models.densenet161, models.densenet169, 
              models.densenet201, models.resnext50_32x4d, models.resnext101_32x8d]

# Filenames for models to be saved to
model_names = ["resnet18.pth", "resnet34.pth", "resnet50.pth", "resnet101.pth", "xrnet18.pth", "xrnet34.pth", "xrnet50.pth", "xrnet101",
               "sqnet1_0.pth", "sqnet1_1.pth", "dnet121.pth", "dnet161.pth", "dnet169.pth", "dnet201.pth", "rnext50_2d.pth", 
               "rnext101_8d.pth"]

def label_func(filename):
    """
    Returns the corresponding label of a file given its filename.
    
    :param filename: (Path) path to file of interest
    :return: (str) label that corresponds to file/data of interest
    """
    if int(re.split('_|p', str(filename))[1]) == 0:
        return 'straight'
    elif '-' in str(filename):
        return 'right'
    else:
        return 'left'   

def get_data(img_dir):
    """
    Get data from a given directory and return a data loader.
    
    :param img_dir: (str) the name of the path to the given data directory
    :return: (DataLoader) a FastAI data loader of the data from the given directory
    """
    # Get images from specified path
    path = Path(img_dir)
    fns = get_image_files(path)

    # Load in data and labels
    dls = ImageDataLoaders.from_path_func(path, fns, label_func) 
    
    return dls

def confusion_matrix(trained_model):
    """
    Create and save confusion matrix as a png file.
    
    :param trained_model: FastAI model created by cnn_learner
    """
    # Getting the confusion matrix data
    interp = ClassificationInterpretation.from_learner(trained_model)
    cm = interp.confusion_matrix()
    
    # Plotting the confusion matrix data
    plt.figure(figsize=(3,3))
    plt.title('Confusion Matrix')
    plt.grid(False) 
    sns.heatmap(cm, cmap = 'Blues', annot = True, annot_kws={"size": 12}, cbar=False, 
                xticklabels=move_type, yticklabels=move_type, fmt='g')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.yticks(rotation=0)
    
    # Saving the plot as a png file
    plt.savefig(os.path.join(path, 'conf_matrix.png'), bbox_inches='tight')

def main():
    # Taking in parameters from script if given
    given_model = sys.argv[1] if len(sys.argv[1]) > 1 else "resnet18"
    img_dir = sys.argv[2] if len(sys.argv[2]) > 1 else "data"
    pretrained = bool(sys.argv[3]) if len(sys.argv[3]) > 1 else False
    file_name = sys.argv[4] if len(sys.arg[4]) > 1 else "resnet18.pth"
                                   
    #TODO: Load in data here, label it, create something that gets the architecture based on the given model name, add pretraining option to CNN learner, find a way to save training and validation loss, make a graph of loss overtime                                   
    if given_model not in str(model_archs):
        print("Invalid Architecture")
        return
    else:
        for model in model_archs:
            if given_model in str(model):
                given_model = model
                break
    
    dls = get_data(img_dir)
    
    # Create a path for each model and its corresponding info to be in
    sub_dir = str(file_name).split('.')[0]
    path = os.path.abspath(os.path.join("classification_models", sub_dir))
    os.mkdir(path)
    
    # Create and finetune model
    learn = cnn_learner(dls, given_model, pretrained=pretrained, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    learn.fine_tune(5)
    learn.recorder.plot_metrics()
    
    # Create and save confusion matrix
    confusion_matrix(learn)

    # Export Model
    learn.export(os.path.join(path, file_name))
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()




