#problem 19
import numpy as np
import time as t
def construct_sigma(singular_values, m, n):
    Sigma = np.zeros((m, n))
    min_dim = min(m, n)
    Sigma[:min_dim, :min_dim] = np.diag(singular_values)

    return Sigma
t1=t.time()
C1=np.array([[2,1],[1,0]])
U1,S1,V1=np.linalg.svd(C1)

Sigma1 = construct_sigma(S1,U1.shape[1],V1.shape[1])

D1=np.dot(np.dot(U1,Sigma1),V1)

print("Matrix(U):")
print(U1)

print("Matrix(S):")
print(Sigma1)

print("Matrix(V):")
print(V1)

print("Matrix Product(USV):")
print(D1)
print("-------------------------------------------------------")
C2=np.array([[2,1],[1,0],[0,1]])
U2,S2,V2=np.linalg.svd(C2)

Sigma2 = construct_sigma(S2,U2.shape[1],V2.shape[1])

D2=np.dot(np.dot(U2,Sigma2),V2)

print("Matrix(U):")
print(U2)

print("Matrix(S):")
print(Sigma2)

print("Matrix(V):")
print(V2)

print("Matrix Product(USV):")
print(D2)
print("------------------------------------------------------")
C3=np.array([[2,1],[-1,1],[1,1],[2,-1]])
U3,S3,V3=np.linalg.svd(C3)

Sigma3 = construct_sigma(S3,U3.shape[1],V3.shape[1])

D3=np.dot(np.dot(U3,Sigma3),V3)

print("Matrix(U):")
print(U3)

print("Matrix(S):")
print(Sigma3)

print("Matrix(V):")
print(V3)

print("Matrix Product(USV):")
print(D3)
print("-------------------------------------------------------")

C4=np.array([[1,1,0],[-1,0,1],[0,1,-1],[1,1,-1]])
U4,S4,V4=np.linalg.svd(C4)

Sigma4 = construct_sigma(S4,U4.shape[1],V4.shape[1])

D4=np.dot(np.dot(U4,Sigma4),V4)

print("Matrix(U):")
print(U4)

print("Matrix(S):")
print(Sigma4)

print("Matrix(V):")
print(V4)

print("Matrix Product(USV):")
print(D4)
print("-----------------------------------------------------")

C5=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,0],[1,0,1]])
U5,S5,V5=np.linalg.svd(C5)

Sigma5 = construct_sigma(S5,U5.shape[1],V5.shape[1])

D5=np.dot(np.dot(U5,Sigma5),V5)

print("Matrix(U):")
print(U5)

print("Matrix(S):")
print(Sigma5)

print("Matrix(V):")
print(V5)

print("Matrix Product(USV):")
print(D5)
print("---------------------------------------------------")