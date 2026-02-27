import numpy as nump
from numpy.random import shuffle


arr = nump.arange(10)
print(arr)
nump.random.shuffle(arr)
print(arr)

print("shuffle")
shuffle(arr)  

print(nump.random.permutation(arr))

arr.sort()

from numpy.strings import add as ad
ad(1, 2)