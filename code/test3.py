'''since data can be not suitable for training -> transform to make it suitable
torchvision datasets has 2 param: transform - change features, target_transform - change labels.
'''
#fashionmnist:features-pil img, label-int.=>features:tensor(totensor()), label:1-hot encoded tensor(lambda).
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10,dtype=torch.float).scatter(0,torch.tensor(y),value=1))
)
#ToTensor():convert(pil img/np narray->floattensor&scale img pixel's intensity value to [0,1])
#lambda transform apply user-defined funct.
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
