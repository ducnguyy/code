#code for data process is messy. torch has 2 data primitives:dataloader & dataset:
#dataset:store sample-label,dataloader:wrap iterable around dataset.
#torch lib has quite a lot pre-loaded dataset for use.

#load dataset
import torch
from torch.utils.data import Dataset
#import sub-class Dataset
from torchvision import datasets
#import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
#import transformation method
"""
we need to specify below parameters:

root: path of dataset
train: dataset is train/test dataset(true/false)
download:need to download from internet (true/false)
transform&target_transform: specify what transformation feature&label take
"""
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#iterating&visualizing dataset
#index dataset like a list [] & visualize = matplotlib
labels_map={
    0:"T-Shirt",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle Boot",
}
figure=plt.figure(figsize=(8,8))
cols,rows=3,3
for i in range(1,cols*rows+1):
    sample_idx=torch.randint(len(training_data),size=(1,)).item()
    img,label=training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()
#create custom dataset
#custom dataset must have 3 functs: init, len, getitem. 
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir, transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image=read_image(img_path)
        label=self.img_labels.iloc[idx,1]
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image, label
#init: run once when instantiating dataset: initialze dir(img, anno file, transforms)
'''label.csv:
tshirt1.jpg,0
tshirt2.jpg, 0
...
ankleboot999.jpg,9
'''
#len: return num(sample in dataset)
#getitem: load&return sample=dataset[idx]: identify img location->read-img:convert to a tensor->retrieve corresponding label in self.img_label
#->call transform->return {tensor img&label}

#prep train data w/dataloader: pass sample in minibatches->reshuffle data each epoch->use multiprocessing to speed up data retrieval.
from torch.utils.data import DataLoader

train_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)
#iterate through dataloader
'''
DataLoader loaded dataset and iterate through dataset. Each iter return batch(train_features, train_labels, size=64)
notice: shuffle=True => data shuffle after iterated.
'''
#display img&label
train_features,train_labels=next(iter(train_dataloader))
print(f"Feature batch shape:{train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img=train_features[0].squeeze()
label=train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label:{label}")
