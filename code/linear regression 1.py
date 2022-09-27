from tkinter import W
import torch

w=torch.tensor(3.0, requires_grad=True)
b=torch.tensor(1.0, requires_grad=True)

def forward(x):
    y=w*x+b 
    return y 

x = torch.tensor(2)
forward(x)








