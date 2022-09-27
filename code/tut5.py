#mat multi

import torch

mat_a=torch.tensor([0,3,5,5,5,2]).view(2,3)
print(mat_a)
mat_b=torch.tensor([3,4,3,-2,4,-2]).view(3,2)
mat_m=torch.matmul(mat_a,mat_b)
print(mat_m)
#or
mat_m=mat_a@mat_b
print(mat_m)





