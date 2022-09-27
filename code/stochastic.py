import re
import numpy as np
from numpy.linalg import inv, norm

def gaussian_elim(matrix: list) -> list:
    length = len(matrix)
    for i in range(length-1):
        for j in range(i+1,length):
            matrix[i][j] = -matrix[i][j] / matrix[i][i]
        for j in range(i+1,length):
            for k in range(i+1,length):
                matrix[k][j] = matrix[k][j] + matrix[i][j]*matrix[k][i]
        print(matrix)
    vector = np.zeros(len(matrix))
    vector[len(matrix)-1] = 1
    for i in range(1, length):
        index = length - 1 - (i-1)
        for j in range(index+1,length+1):
            vector[index-1] = vector[index-1] + matrix[j-1][index-1]*vector[j-1]
        vector[index-1]=-vector[index-1]/matrix[index-1][index-1]
    vector = vector/norm(vector,1)
    return vector

def gauss_seidel(matrix: list, x0: list, b: list, itmax: int) -> tuple:
    matrix = np.array(matrix)
    soln = np.array(x0)
    b = np.array(b)
    D = np.diag(np.diag(matrix))
    L = -np.tril(matrix)+D
    U = -np.triu(matrix)+D

    # # Gauss-Seidel
    # M = inv(D - L)
    # B = M@U # Gauss-Seidel iteration matrix

    # # Jacobi
    # M = inv(D)
    # B = M@(matrix - D)

    # Successive Overrelaxation
    w = 1.1
    b = w*b
    M = inv(D - w*L)
    B = M@((1-w)*D + w*U)
    print(f"M:{M}\nD:{D}\nU:{U}\nL:{L}\nB:{B}")

    resid = []
    for i in range(itmax):
        soln = B@soln + M@b
        if norm(b, 2) == 0:
            soln = soln/norm(soln, 1)
        resid.append(norm(matrix@soln - b, 2))

    return(soln, resid)

def sor_alt(matrix: list, x0: list, itermax: int):
    a = []
    x = []
    y = []
    
    for i in range(len(matrix)):
        x.append(len(y))
        a.append(1)
        y.append(i)
        for j in range(len(matrix)):
            if matrix[i][j] != 0 and i != j:
                a.append(matrix[i][j]*(1/matrix[0][0]))
                y.append(j)
    x.append(len(y))

    for k in range(itermax):
        for i in range(len(x0)):
            s = 0
            initi = x[i]
            laste = x[i+1]
            for j in range(initi, laste):
                s += a[j] * x0[y[j]]
            x0[i] -= 1.1 * s
    
    x0 = np.array(x0) / norm(x0,1)
    return x0

# matrix = [
#  [1, 2, 0, 0],
#  [3, 1, 1, 0],
#  [1, 0, 1, 0],
#  [0, 1, 0, 1]
# ]
matrix = [
    [-0.5  ,  0.5  ,  0   ,  0  ],
    [ 0    , -0.5  ,  0.5 ,  0  ],
    [ 0    ,  0    , -0.5 ,  0.5],
    [ 0.125,  0.125,  0.25, -0.5]
]
x0 = [1, 2, 4, 4]
b = [0.0, 0.0, 0.0, 0.0]
itmax = int(input())

# print(gaussian_elim(matrix))
# print(gauss_seidel(matrix, x0, b, itmax)[0])
print(sor_alt(matrix, x0, itmax))
