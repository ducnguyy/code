import torch
import numpy as np
print("it;s working right?")
data=[[1,2],[3,4]]
x_data=torch.tensor(data)
np_array=np.array(data)
x_np=torch.from_numpy(np_array)
x_ones=torch.ones_like(x_data)
print(f"Ones tensor:\n {x_ones}\n")
x_rand=torch.rand_like(x_data,dtype=torch.float)
print(f"Random tensor:\n{x_rand}\n")
shape=(2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(f"random tensor:\n{rand_tensor}\n")
print(f"1 tensor:\n{ones_tensor}\n")
print(f"0 tensor:\n{zeros_tensor}\n")
tensor=torch.rand(3,4)
print(f"tensor shape:{tensor.shape}\n")
print(f"tensor datatype:{tensor.dtype}\n")
print(f"tensor stored at:{tensor.device}\n")
'''gpu faster than cpu. tensor default is stored in cpu. to fasten the process, we move em to gpu using ".to"'''
if torch.cuda.is_available():
    tensor=tensor.to("cuda")
tensor=torch.ones(4,4)
print(f"First row:{tensor[0]}")#first row
print(f"First column:{tensor[:,0]}")#first column
print(f"Last column:{tensor[...,-1]}")#last column
tensor[:,1]=0#change 2nd column to 0's
print(tensor)
#concatenate multi tensor into 1
t1=torch.cat([tensor,tensor,tensor],dim=1)
print(t1)
#matrix multiplication btw 2 tensor. y1=y2=y3
y1=tensor@tensor.T
y2=tensor.matmul(tensor.T)
y3=torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(f"{y1}\n{y2}\n{y3}\n")
#element-wise product. z1=z2=z3
z1=tensor*tensor
z2=tensor.mul(tensor)
z3=torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)
print(f"{z1}\n{z2}\n{z3}\n")
#sum() all entries. item() to convert the data->python numerical data.
agg=tensor.sum()
agg_item=agg.item()
print(agg_item,type(agg_item))
print(f"{tensor}\n")
#.add() to add sth to all entries.
tensor.add_(5)
print(tensor)
#tensor in cpu and np array stored in same part->we can "bridge" btw 2.  
t=torch.ones(5)
print(f"t:{t}")
n=t.numpy()
print(f"n:{n}")
#change in 1 reflect in other.
t.add_(1)
print(f"t:{t}")
print(f"n:{n}")
#num to tensor
n=np.ones(5)
t=torch.from_numpy(n)
np.add(n,1,out=n)
print(f"t:{t}")
print(f"n:{n}")
