import numpy as np

#Standard Deviation is used such that a curve is formed which can have numbers from -∞ to ∞
#There is a higher chance for numbers to be closer to 0 than away from it

# About 68% of values lie between -1 and +1
# About 95% between -2 and +2
# About 99.7% between -3 and +3

print(np.random.randn(5))     # mean=0, std=1 (normal dist)
print(np.random.randn(2,3))

x = np.random.randn(5000000)    #5 mil checks

for i in range(len(x)):
    if(x[i]>5):     #chance of 5 is 1 in 3.5 million
        print(x[i])
        
