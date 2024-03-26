#problem 11
import numpy as np
A1 = np.array([[3, -1,1], [3, 6,2],[3,3,7]])
b1 = np.array([1,0,4])
x1 = np.linalg.solve(A1, b1)

print("Solution(x) of 1st set of equations:")
print(x1)

A2 = np.array([[10, -1,0], [-1, 10,-2],[0,-2,10]])
b2 = np.array([9,7,6])
x2 = np.linalg.solve(A2, b2)

print("Solution(x) of 2nd set of equations:")
print(x2)

A3 = np.array([[10,5,0,0], [5,10,-4,0],[0,-2,8,-1],[0,0,-1,5]])
b3 = np.array([6,25,-11,-11])
x3 = np.linalg.solve(A3, b3)

print("Solution(x) of 3rd set of equations:")
print(x3)

A4 = np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
b4 = np.array([6,6,6,6,6])
x4 = np.linalg.solve(A4, b4)

print("Solution(x) of 4th set of equations:")
print(x4)