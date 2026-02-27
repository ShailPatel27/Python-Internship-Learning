import numpy as np

# python lists treat vector operations as string ones
l1 = [1, 2, 3]
l2 = [10, 20, 30]

print(l1 + l2)
print(l1 * 5)
# print(l1*l2)
# print(l1*l2)

#numpy arrays treat vectors as actual numbers
a1 = np.array([1, 2, 3])
a2 = np.array([10, 20, 30])

print(a1 + a2)
print(a1 * a2)
print(a1**2)
print(a1 * 5)