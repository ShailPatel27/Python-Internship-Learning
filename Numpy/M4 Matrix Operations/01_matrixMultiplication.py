import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[6, 5, 4], [3, 2, 1]])

print(np.dot(a, b.T))   #matrix multiplication
# or
print(a @ b.T)  #matrix multiplication
