import torch
import numpy as np

#dimensional tensor
v=torch.tensor([1,2,3,4,5,6])
print(v)
print(v.dtype)
#dtype return type tensor
print(v[0])
#indexing same as list
print(v[1:])
print(v[1:4])
#the final index is exclusive

f=torch.FloatTensor([1,2,3,4,5,6])
print(f.dtype)


print(v.view(3,-1))
#view input must compatible to og input
# view(3,2)=view(3,-1) - use-1 to save confusion of size

#converse from numpy array to torch tensor, vice versa
a=np.array([1,2,3,4,5])
tensor_cnv=torch.from_numpy(a)
print(tensor_cnv,tensor_cnv.type())
numpy_cnv=tensor_cnv.numpy()
print(numpy_cnv)


