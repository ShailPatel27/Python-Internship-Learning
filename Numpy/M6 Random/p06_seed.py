import numpy as np

#Sets a seed that will

np.random.seed(42)
print(np.random.randint(1, 100, 5))
print(np.random.randint(1, 100, 5))

np.random.seed(2321)
print(np.random.randint(1, 100, 5))
