#problem 
import numpy as np
A=np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
b=np.array([1.,2.,3.,4.,5.])
x0=np.array([1.,1.,1.,1.,1.])
# Jacobi method
def jacobi(A, b, x0, tol=0.01, max_iter=1000):  # defining the jacobi function which givs the final solution and number of iterations
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    iterations = 0
    while iterations < max_iter:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new.copy()
        iterations=iterations+1
    return x, iterations
# Solving by Jacobi method
solution, iterations = jacobi(A, b, x0)
print("Solution using Jacobi method :", solution)
print("Number of iterations:", iterations)
print("---------------------------------------------")

# Gauss-Seidel method
def gauss_seidel(A, b, x0, tol=0.01, max_iter=1000):
    n = len(b)
    x = x0.copy()
    iterations = 0
    while iterations < max_iter:
        x_new = np.zeros_like(x)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        iterations=iterations+1
    return x, iterations

# Solving by Gauss-Seidel method
solution, iterations = gauss_seidel(A, b,x0)
print("Solution using Gauss-seidal method:", solution)
print("Number of iterations:", iterations)
print("---------------------------------------------")

# Relaxation method
def relaxation_method(A, b, x0, tolerance=0.01, max_iterations=1000, omega=1.25):
    n = len(b)
    solution = np.copy(x0)
    for iteration in range(max_iterations):
        old_solution = np.copy(solution)
        for i in range(n):
            summation = 0.0
            for j in range(n):
                if j != i:
                    summation=summation+A[i][j] * solution[j]
            solution[i] = (1 - omega) * old_solution[i] + (omega / A[i][i]) * (b[i] - summation)
        if np.linalg.norm(solution - old_solution) < tolerance:
            return solution, iteration + 1
    print("Warning: Maximum number of iterations reached without convergence.")
    return solution, max_iterations

# Solving by Relaxation method
solution, num_iterations = relaxation_method(A, b, x0)
print("Solution using Relaxation method:", solution)
print("Number of iterations:", num_iterations)
print("-------------------------------------------")

# Conjugate Gradient method
def conjugate_gradient(A, b, x0, tol=0.01, max_iter=1000):
    r = b - np.dot(A, x0)
    p = r
    x = x0
    rsold = np.dot(r, r)
    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, i+1

# Solving by Conjugate Gradient method
solution, iterations = conjugate_gradient(A, b, x0)
print("Solution:", solution)
print("Number of iterations:", iterations)
print("------------------------------------------")