import numpy as np

print(np.random.randint(10))                   # single int(0-9)
print(np.random.randint(10, 100), "\n")        # single int (10-99)
print(np.random.randint(10, 100, 5), "\n")     # 1D array of 5 ints
print(np.random.randint(10, 100, (2,3)))       # 2D array
