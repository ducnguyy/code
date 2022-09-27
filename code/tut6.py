import torch 
x=torch.tensor(1.0,requires_grad=True)
z=torch.tensor(2.0,requires_grad=True)
#y=9*x**4+2*x**3+3*x**2+6*x+1
#a=y.backward()
#b=x.grad
#print(a,b)

y=x**2+z**3
a=y.backward()
b=x.grad 
c=z.grad
print(b,c)