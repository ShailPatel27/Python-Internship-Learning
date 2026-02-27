import numpy as np

arr = np.array([10, 20, 30, 40, 50])
l = [10, 20, 30, 40, 50]
print(np.random.choice(arr))          # one element from array 
print(np.random.choice(l))            # one element from list
print(np.random.choice(arr, 3))       # 3 elements

print(np.random.choice(arr, 3, p=[0.1, 0.2, 0.3, 0.3, 0.1]))    # chance/probability
print(np.random.choice(arr, 6, p=[0.1, 0.2, 0.3, 0.3, 0.1]))    # chance/probability

