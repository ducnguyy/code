#importing necessary packages for image processing
from __future__ import print_function, division
import torchvision
import PIL
from PIL import Image
import os
import cv2 
#import glob
import torch
from torchvision import transforms
import torchvision.transforms as T
#import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#create dataloader
class dog_cat(Dataset):
    def __init__(self, root_dir,train=False):
        self.dataset_dir=root_dir 
        self.labels=np.array([])
        self.link=np.array([]) 
        if train==True:
            self.cat_dir=self.dataset_dir+"/training_set/cats"
            self.dog_dir=self.dataset_dir+"/training_set/dogs"
        else:
            self.cat_dir=self.dataset_dir+"/test_set/cats"
            self.dog_dir=self.dataset_dir+"/test_set/dogs"
        for filename in os.listdir(self.cat_dir):
            self.labels=np.append(self.labels,[0])
            self.link=np.append(self.link,filename)
        for filename in os.listdir(self.dog_dir):
            self.labels=np.append(self.labels,[1])
            self.link=np.append(self.link,filename)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,IDX):
        img=self.link[IDX]
        label_of_img = self.labels[IDX]
        if label_of_img==1:
            img_true=Image.open(os.path.join(self.dog_dir,img))
        else:
            img_true=Image.open(os.path.join(self.cat_dir,img))
        transform = T.Resize((250,250))
        img_true=transform(img_true)
        img_true=np.array(img_true)
        img_true=img_true.astype("float64")
        img_true/=255
        #plt.imshow(img_true)
        #plt.show()
        label_of_img=np.array(label_of_img)
        img_true=torch.from_numpy(img_true)
        img_true=img_true.permute(2,0,1)
        label_of_img=torch.from_numpy(label_of_img)
        print(img_true,img_true.shape,label_of_img.type())
        return img_true, label_of_img
root_dir_true="/home/tkdc/dog_cat_dataset/archive/dataset"
test=dog_cat(root_dir_true)
#img, label= test.get_item(0)

dataset_loader = torch.utils.data.DataLoader(test,batch_size=4, shuffle=True,num_workers=4)

#class doggocatter(Dataloader):
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

class Doggotrainer(torch.nn.Module):
    def __init__(self):
        super(Doggotrainer,self).__init__()
        self.flatten=torch.nn.Flatten()
        self.cnn1=torch.nn.Sequential(
            torch.nn.Conv2d(3,3,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.cnn2=torch.nn.Sequential(
            torch.nn.Conv2d(3,20,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc1=torch.nn.Sequential(
            #torch.nn.Linear(features),
            torch.nn.ReLU(),
        )
        self.fc2=torch.nn.Sequential(
            #torch.nn.Linear(features),
            torch.nn.Softmax(1),
        )
    def forward(self,x):
        #x=self.flatten(x)
        x=self.cnn1(x)
        x=self.cnn2(x)
        #x=self.fc1(x)
        #output=self.fc2(x)
        #return output
        return x

height,weight= 250,250
n_epochs=10
loss=torch.nn.BCELoss()
cattomodel=Doggotrainer()#.to(device)
print(cattomodel)
YE=torch.rand(1,3,250,250)
logits=cattomodel(YE)
pred_probab = torch.nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
