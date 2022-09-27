#optimize params
'''
we have model&data.
lets train+validate+test it.
iteration=epoch, each epoch: model guess on output -> cal loss -> cal dl/d(params) -> optimize params.
'''

#load code from prev sessions
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda

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

train_dataloader=DataLoader(training_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

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
model=NeuralNetwork()
#the model - done.

#hyperparams: adjustable params ex: num(epochs), batch-size,learning_rate
'''
number of epochs: nums times iterate over dataset
batch size: nums data samples propagated through
learning rate:how much update params at each epoch.
'''
learning_rate=1e-3
batch_size=64
epochs=5
# 1 epoch has 2 parts: 
# train loop: iterate over train dataset, converge to optimal params
# test loop: ______________test dataset, check if model improve.

#brief familiarize:
#loss funct = dissimilarity btw result&target. need minimized.
loss_fn=nn.CrossEntropyLoss()

#optimization algo's using optimizer.
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#implement train&test loop:
def train_loop(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred=model(X)
        loss=loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100==0:
            loss,current=loss.item(),batch*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss,correct=0,0

    with torch.no_grad():
        for X,y in dataloader:
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
        
        test_loss/=num_batches
        correct/=size
        print(f"test error\n accuracy:{(100*correct):>0.1f}%, avg loss:{test_loss:>8f}\n")

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

epochs=10
for t in range(epochs):
    print(f"epoch{t+1}\n---")
    train_loop(train_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
print("done")

