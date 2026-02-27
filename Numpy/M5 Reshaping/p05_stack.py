import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Creates a new dimension instead of joining along existing ones.

print(np.stack((arr1, arr2)) ,"\n")       # shape (2, 3)
print(np.stack((arr1, arr2), axis=1))  # shape (3, 2)
