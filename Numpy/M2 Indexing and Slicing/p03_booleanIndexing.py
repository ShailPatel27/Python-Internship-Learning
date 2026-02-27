import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

# l = [1, 2, 3, 4, 5]
# python lists do not support boolean indexing
# print(l[l>3])

print(arr[arr>3])