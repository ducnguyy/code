#linear perceptron algorithm
import numpy as np
import matplotlib.pyplot as plt 
import math as m 
theta=np.array([0.0,0.0])
theta_dot=0.0

#data : x-y:
X=np.array([[1,1],[2,3],[3,4],[-0.5,-0.5],[-1,-2],[-2,-3],[-3,-4],[-4,-3]])
Y=np.transpose(np.array([[1,1,1,-1,-1,-1,-1,-1]]))
print(X.shape,Y.shape)

#rotational matrix:
R=np.array([[0.5,-(m.sqrt(3))/2],[m.sqrt(3)/2, 0.5]])
R.reshape([2,2])
#rotate the X data value:
for t in range(X.shape[0]):
    print("Rotate:",X[t][:])
    X[t][:]=np.matmul(R,X[t][:])
    print("turned into:",X[t][:])
#Run perceptron
TRAINING_EP=4
for i in range(TRAINING_EP):
    for t in range(X.shape[0]):
        print("Y at step ",t," :",Y[t],"\n")
        print("X at step ",t," :",X[t][:],"\n")
        if Y[t]*(np.dot(theta,X[t][:])+theta_dot)<=0:
            print("Mistake found\n")
            theta+=(X[t][:]*Y[t])
            theta_dot+=Y[t]
            print(theta,"\n",theta_dot)
print(theta,theta_dot,"\n")

#rotate theta:
print(np.matmul(R,np.transpose(np.array([[1.5,1.5]]))))
#x_trans=np.transpose(X)
#x_1=x_trans[0][:]
#x_2=x_trans[1][:]
#plt.title("Line graph")
#plt.xlabel("X axis")
#plt.ylabel("Y axis")
#plt.scatter(x_1,x_2, color ="red")
#plt.show()