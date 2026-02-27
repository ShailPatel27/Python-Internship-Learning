import numpy as np

arr = np.arange(12)
print(arr)

reshaped = arr.reshape(3,4)
print(reshaped)

print(arr.reshape(6,-1))
# -1 is used to auto calculate the second value. You only need to provide 1 value