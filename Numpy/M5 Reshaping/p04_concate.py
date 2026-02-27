import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c = np.array([[7], [8]])

# combine along rows (axis=0)
print(np.concat((a, b), axis=0))

# combine along columns (axis=1)
print(np.concatenate((a, c), axis=1))

# or
print("\n")

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(np.vstack((x, y)))  # vertical (2 rows)
print(np.hstack((x, y)))  # horizontal (1 long row)
