import torch

x=torch.arange(18).view(3,2,3)

print(x)
print(x[1,1,1])
print(x[1,0:2,1])
#stop index is inclusive

print(x[1,:,:])
#omit ranges = all range)




