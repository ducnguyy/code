import torch 
#initialize 1 tensor = arange
one_d=torch.arange(0,9)
print(one_d)
#view one dim matrix as 2d mat 3 rows 3 cols
two_d=one_d.view(3,3)
print(two_d)
print(two_d.dim())

two_d[1,2]
#2x3x3 mat
x=torch.arange(18).view(3,3,2)
print(x)








