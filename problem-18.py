#problem 18
import numpy as np
def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)  # Initial guess for the eigenvector
    x /= np.linalg.norm(x)  # Normalize the initial guess
    for _ in range(max_iter):
        x_new = np.dot(A, x)
        eigenvalue = np.dot(x_new, x)
        x_new /= np.linalg.norm(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return eigenvalue, x
# given matrix:
A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
eigenvalue, eigenvector = power_method(A)
print("Dominant eigenvalue:", eigenvalue)
print("Corresponding eigenvector:", eigenvector)