import numpy as np

arr = np.arange(10)
print(np.split(arr, 2), "\n")   # 2 equal parts

print(np.array_split(arr, 3), "\n")   # 3 parts (some uneven)

arr2d = np.array([[1,2,3,4], 
                  [5,6,7,8]])
print(np.hsplit(arr2d, 2), "\n")  # split into 2 column groups
print(np.vsplit(arr2d, 2))  # split into 2 row groups
