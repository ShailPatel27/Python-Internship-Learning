import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print(arr[0])   #10
print(arr[-1])  #50

print(arr[1:3]) #20,30
print(arr[:3])  #10,20,30
print(arr[::2]) #10,30,50

# ⚠️ NOTE: SLICING CREATES A POINTER IN NUMPMY ARRAYS
# IF WE SLICE A LIST AND TAKE IT TO ANOTHER VARIABLE, THEN CHANGING THAT VARIABLE WILL ALSO CHANGE THE ORIGINAL LIST

sliced_arr = arr[1:3]
sliced_arr[0] = 9000
print(arr)