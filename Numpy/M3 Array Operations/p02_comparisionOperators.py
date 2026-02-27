import numpy as np

arr = np.array([10, 20, 30])

print(arr[arr>20]) #[30]

print(arr>20)   #[false false true]
print(arr==20)  #[false true false]