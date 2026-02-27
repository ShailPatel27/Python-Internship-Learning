import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr[0, 1])    #2
print(arr[1])       #[4,5,6]

print(arr[:2, 1:])      #[[2,3][5,6]]

print(arr[:2, 1:-1])    #[[2][5]]
#here [[2,3][5]] should have been the output but arrays can only be in matrix form so 3 was ignored