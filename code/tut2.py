#tut2
import torch
import matplotlib.pyplot as plt
t_one=torch.tensor([1,2,3])

t_two=torch.tensor([1,2,3])

#arithmetic operations
print(t_one*t_two)
dot_prod=torch.dot(t_one,t_two)
print(dot_prod)

#linspace(start,end range,#numbers)
x=torch.linspace(0,10,100)
y=torch.sin(x)

plt.plot(x.numpy(),y.numpy())
#show() to display
plt.show()













