import numpy as np
import copy

a = np.ones((4,4))
b= a.copy()
b[0,0] = 0
print(a)