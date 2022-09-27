#auto differential w/torch.autograd
'''
back prop: params adjusted according to grads of L() r.t given param.
we use back prop alot. 
torch.autograd auto computes grads.
'''

# define 1-layer NN, input=x, params=w,b, has L().
import torch 
x=torch.ones(5) #input tensor
y=torch.zeros(3) #E output
w=torch.randn(5,3,requires_grad=True)
b=torch.randn(3,requires_grad=True)
z=torch.matmul(x,w)+b
loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)

#optimize params: w&b = find grad(l r.t w,b) = requires_grad
'''
function = object(class(Function)). this obj know for-prop and back-prop = grad-fn
'''
print(f"grad function for z={z.grad_fn}")
print(f"grad funct of loss = {loss.grad_fn}")

#compute grads
'''
find dl/dw and dl/db under fixed(x&y) = loss.backward() -> w.grad&b.grad
'''
loss.backward()
print(w.grad)
print(b.grad)

#disable grad tracking
'''
tensors w/requiregrad=true tracking computation history. in case don't need<=>only forward: stop = torch.no_grad() 
'''
z=torch.matmul(x,w)+b
print(z.requires_grad)
with torch.no_grad():
    z=torch.matmul(x,w)+b
print(z.requires_grad)
#or use detach()
z=torch.matmul(x,w)+b
z_det=z.detach()
print(z_det.requires_grad)
#why stop tracking: frozen params&faster computations

#dag kinda like computational graph: leaves=input, root=output. tracking from root to leave = back prop


