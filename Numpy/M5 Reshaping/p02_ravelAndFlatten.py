import numpy as np

arr2d = np.array([[1,2,3], [4,5,6]])

#Both are used to convert multi dim arrays to 1D arrays

print(arr2d.ravel())    # returns a *view* (changes affect original)
print(f"Original:\n{arr2d}\n")

print(arr2d.flatten())  # returns a *copy* (changes do NOT affect original)
print(f"Original:\n{arr2d}")
