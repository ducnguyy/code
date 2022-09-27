import numpy as np
import pandas as pd 
import math as math 

X=np.array([[1,0],[0,1]])
#print(X)
s0=np.transpose(np.array([0.0,0.0]))
print(np.shape(s0))
W_ss=np.array([[-1.0,0.0],[0.0,1.0]])
W_sx=np.array([[1.0,0.0],[0.0,1.0]])
shape_X_a, shape_X_b=np.shape(X)
#print(shape_X_a,shape_X_b)
for i in range(shape_X_a):
    print("xi:",X[i][:])
    infor_x=np.matmul(W_sx,np.transpose(X[i][:]))
    infor_s=np.transpose(np.matmul(W_ss,s0))
    print("s0",s0)
    sum=infor_x+infor_s
    print(np.shape(sum))
    print("x:",infor_x,"\n s:",infor_s,"\n sum:",sum)
    for f in range(2):
        if sum[f]>0:
            s0[f]=sum[f]
        else:
            s0[f]=0
print(s0)
    

