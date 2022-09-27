#importing necessary packages for image processing
from __future__ import print_function, division
from PIL import Image
import os
import cv2 
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt

image_list=[]

img_folder="/home/tkdc/dog_cat_dataset/archive/dataset/training_set"
def create_dataset_PIL(img_folder):
    IMG_HEIGHT=500
    IMG_WIDTH=500
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= np.array(Image.open(image_path))
            image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array , class_name
PIL_img_data, class_name=create_dataset_PIL(img_folder)
plt.imshow(PIL_img_data[1], interpolation='nearest')
plt.show()