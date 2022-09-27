#NN:old pal. 
# torch.nn: provides building blocks to build NN.
# nn.Module: every module in pytorch subclasses.
# NN = a module consists of modules(layers).
 
#ex: build nn: classify img in fashionMNIST
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#to train on GPU = use cuda()
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")

#define nn = nn.module -> initialize nn layers = init (all modules in nn.module use forward method)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

#create nn instance & move to device
model=NeuralNetwork().to(device)
print(model)

#pass input to use model.
X=torch.rand(1,28,28,device=device)
logits=model(X)
pred_probab=nn.Softmax(dim=1)(logits)
y_pred=pred_probab.argmax(1)
print(f"predicted class: {y_pred}")

#break sown layers in model.
#sample: minibatch(3 img 28x28)
input_image=torch.rand(3,28,28)
print(input_image.size())

#nn.flatten: covert 2d img -> pixel value array 
flatten=nn.Flatten()
flat_image=flatten(input_image)
print(flat_image.size())

#nn.linear: apply linear transformation to input using weights&bias
layer1=nn.Linear(in_features=28*28,out_features=20)
hidden1=layer1(flat_image)
print(hidden1.size())

#nn.relu:use for nonlinearity
print(f"before relu:{hidden1}\n\n")
hidden1=nn.ReLU()(hidden1)
print(f"After relu:{hidden1}")

#nn.sequential = ordered container(modules). data passes through in order. seq_modules to put tgh a quick network.
seq_modules=nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10),
)
input_image=torch.rand(3,28,38)

#nn.Softmax(logits=raw values [-infty,infty]). -> logits[0,1]=prob. dim:dim which value sum to 1.
softmax=nn.Softmax(dim=1)
pred_probab=softmax(logits)

#model parameters: nn.module auto make all params accessible.
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer:{name}|size:{param.size()}|values:{param[:2]}\n")


