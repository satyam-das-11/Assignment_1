#problem 17
import numpy as np
def QR(A, max_iter=100, tol=1e-3):
    n = A.shape[0]
    V = np.eye(n)
    for i in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)
        V = np.dot(V, Q)
        if np.abs(A.diagonal(-1)).max() < tol:
            break
    eigenvalues = A.diagonal()
    return eigenvalues, V

# The given matrix
A=np.array([[5,-2],[-2,8]])
eigenvalues, eigenvectors = QR(A)
print("Eigenvalues using QR decomposition:")
eigenvalues=np.sort(eigenvalues)
print(eigenvalues)
eigenvalues1, eigenvectors1=np.linalg.eigh(A)
print("Eigenvalues using numpy.linalg.eigh:")
print(eigenvalues1)