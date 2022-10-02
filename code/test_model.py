#importing necessary packages for image processing
import torchvision
import PIL
from PIL import Image
import os
import cv2 
#import glob
import torch 
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as T
#import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#create dataloader
class dog_cat(Dataset):
    #__init__ give back the complete root for operating based on true root provided by user&parameter Train
    def __init__(self, root_dir,train=True):
        self.dataset_dir=root_dir     
        self.labels=np.array([])
        self.link=np.array([]) 
        #making the training/test set's path
        if train==True:
            self.cat_dir=self.dataset_dir+"/training_set/cats"
            self.dog_dir=self.dataset_dir+"/training_set/dogs"
        else:
            self.cat_dir=self.dataset_dir+"/test_set/cats"
            self.dog_dir=self.dataset_dir+"/test_set/dogs"
        #start create labels tensor based on directions of each set: cat = 0, dog =1 
        #and making complete paths for each image
        for filename in os.listdir(self.cat_dir):
            self.labels=np.append(self.labels,[0])
            self.link=np.append(self.link,filename)
        for filename in os.listdir(self.dog_dir):
            self.labels=np.append(self.labels,[1])
            self.link=np.append(self.link,filename)
    #call __len__ give back length of Output - # of torch tensors in training/test set
    def __len__(self):
        return len(self.labels)
    #call getitem, it should give back 1 single torch type /double/ at the shape of (#channels,height,weight) - here is (3,250,250)
    def __getitem__(self,IDX):
        #take path of image & their label based on their idx
        img=self.link[IDX]
        label_of_img = self.labels[IDX]
        if label_of_img==1:
            img_true=Image.open(os.path.join(self.dog_dir,img))
        else:
            img_true=Image.open(os.path.join(self.cat_dir,img))
        #resize them to 250x250, #channel untouched
        transform = T.Resize((250,250))
        img_true=transform(img_true)
        img_true=np.array(img_true)
        #cast type float64 to match with requirement of tensor type
        img_true=img_true.astype("float64")
        #normalize image by divide by the largest pixel value = 255
        img_true/=255
        #test by showing the image's numpy array using plt
        #plt.imshow(img_true)
        #plt.show()
        label_of_img=np.array(label_of_img)
        #change type of img data from numpy to torch tensor
        img_true=torch.from_numpy(img_true)

        img_true=img_true.permute(2,0,1)
        label_of_img=torch.from_numpy(label_of_img)#.reshape(-1,1)
        #print(img_true,img_true.shape,label_of_img.type())
        return img_true, label_of_img
root_dir_true="/home/tkdc/dog_cat_dataset/archive/dataset"
#root_dir_true="/Users/nguyenduc/Downloads/cat_and_dog/dataset"
trainset=dog_cat(root_dir_true)
len_dataset=trainset.__len__()
print(len_dataset)
#img, label= test.__getitem__(2)
#create dataset loader which separate dataset OG to batches each contains 16 examples, therefore #batch = OG_len/len_batch, and set #multi processes to 0 since mac sucks
#sauce:https://github.com/pytorch/pytorch/issues/70344
def my_collate(batch):
    x_b,y_b=[],[]
    for X,y in batch:
        x_b.append(X)
        y_b.append(y)
    ge1=torch.stack((x_b))
    ge2=torch.stack((y_b))    
    return ge1,ge2
dataset_loader=torch.utils.data.DataLoader(trainset,batch_size=40,shuffle=True,num_workers=4,collate_fn=my_collate)
#check for dimension of each batch - as intended, each X_batch has dimension [16, 3, 250, 250], y_batch [16]


#i=0
#for X,y in dataset_loader:
#    i+=1
#    print("#",i,": X[i] has shape",X[i].shape,"y[i] has shape",y[i].shape)

#change to gpu if available, else device = cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


#now build the model 
class Doggotrainer(nn.Module):
    def __init__(self):
        super(Doggotrainer,self).__init__()
        self.flatten=nn.Flatten()
        self.cnn1=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cnn2=nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cnn3=nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(3*3*64,10),
            nn.Dropout(0.5),
            nn.Linear(10,2),
            nn.ReLU()
        )
    def forward(self,x):
        x=self.cnn1(x)
        x=self.cnn2(x)
        x=self.cnn3(x)
        #flattening tensor by x.view
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return x
        #return x
#define loss function and optimizing method:
TRAINER=Doggotrainer().float().to(device="cuda")
loss_criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(TRAINER.parameters(),lr=0.001)
#start training the model
total_correct = 0
total_predictions = 0
for epoch in range(23):    
    for X,y in dataset_loader:
        #print(X.shape)
        #move data to cuda for operating
        X=X.to(device="cuda")
        y=y.to(device="cuda")
        predict=TRAINER(X.float())
        #print("predict has shape:",predict.shape,"while y true has shape:",y.shape)
        #cast long type for y because error
        prediction = predict.max(1).indices
        total_correct += (y==prediction).float().sum()
        total_predictions +=  prediction.size(0)
        loss=loss_criterion(predict,y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss for current epoch",epoch,":",loss)
print(f"Accuracy on test set : {float(total_correct/total_predictions)*100:.2f}")
#now try it on the test set
testset=dog_cat(root_dir_true,train=False)
dataset_loader=torch.utils.data.DataLoader(testset,batch_size=40,shuffle=True,num_workers=4)
#TRAINER.eval()
with torch.no_grad():
    total_correct = 0
    total_predictions = 0
    for X_test,y_test in dataset_loader:
        X_test=X_test.to(device="cuda")
        y_test=y_test.to(device="cuda")
        predict=TRAINER(X_test.float())
        #take the maximum value's indices of the prediction
        prediction = predict.max(1).indices
        total_correct += (y_test==prediction).float().sum()
        total_predictions +=  prediction.size(0)
        #print(predict)
TRAINER.train()    
print(f"Accuracy on test set : {float(total_correct/total_predictions)*100:.2f}")