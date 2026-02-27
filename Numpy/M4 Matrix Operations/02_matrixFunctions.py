import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.T)  #Transpose
print(sum(a))   #Total sum of all elements
print(a.sum(axis=0))    #column wise sum: [5 7 9]
print(a.sum(axis=1))    #row wise sum: [6, 15]

