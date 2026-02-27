import numpy as np

square = np.array([[2, 1],
                   [5, 3]])

print(np.linalg.det(square))   #determinant
print(np.linalg.inv(square))   #inverse
print(np.linalg.eig(square))   #eigenvalues and eigenvectors
