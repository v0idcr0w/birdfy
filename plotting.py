import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import pathlib 
import random 

train_style_dict = {"label": "Train", "color": "indigo", "marker": "o"}
test_style_dict = {"label": "Test", "color": "deeppink", "marker": "o"}
def plot_results(results_dict):
    epochs = range(len(results_dict["train_loss"]))
    fig, axs = plt.subplots(1, 2)  
    
    axs[0].plot(epochs, results_dict["train_loss"], **train_style_dict)
    axs[0].plot(epochs, results_dict["test_loss"], **test_style_dict)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_box_aspect(1)
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)
    
    axs[1].plot(epochs, results_dict["train_acc"], **train_style_dict)
    axs[1].plot(epochs, results_dict["test_acc"], **test_style_dict)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_box_aspect(1)
    axs[1].legend()
    axs[1].grid(True, alpha=0.5)
    
    fig.tight_layout()
    
    
def plot_from_dataset(data, label_dict, index: int):
    """Given a dataset, plots an image"""
    
    # Fetch the tensor, permute shape, transfer to cpu and turn it into a numpy array. 
    if index > len(data) - 1:
        raise KeyError("Index out of bounds")
    image = data[index][0].permute(1,2,0).to("cpu").numpy()
    label = label_dict[data[index][1]]
    
    fig, ax = plt.subplots()
    
    ax.imshow(image) 
    ax.set_title(label)
    ax.axis(False)
    ax.set_box_aspect(1)
    
    return fig, ax 
    
def plot_random_image(path: str):
    """Given a string path of training/test images, plot a random image with no transformations"""
    
    image_path = pathlib.Path(path)
    image_list = list(image_path.glob("*/*/*.jpg")) 
    random_image = random.choice(image_list) 
    image_class = random_image.parent.stem
    
    image = Image.open(random_image)
    fig, ax = plt.subplots()
    
    ax.imshow(np.asarray(image))
    ax.set_title(image_class)
    ax.axis(False)
    ax.set_box_aspect(1)
    
    return fig, ax
    