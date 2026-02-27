import numpy as np

arr = np.arange(10)
print(arr)
np.random.shuffle(arr)        # shuffles the original array
print(arr)

print(np.random.permutation(arr))  # makes a new array and shuffles it (does not change original)

from numpy.strings import add as ad
ad(1, 2)