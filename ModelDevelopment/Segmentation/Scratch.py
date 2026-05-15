import numpy as np

test = np.zeros(shape=(4, 4, 3))
test[2,2,:] = 1
test[2,2,2] = 3

#print(test)

points = np.reshape(test, (4 * 4, 3))
print(points)