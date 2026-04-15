import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 1, 100)
y = np.linspace(0.01, 1, 100)
X, Y = np.meshgrid(x, y)

f1 = np.divide(np.multiply(X, Y), X+Y)
plt.imshow(f1)
plt.colorbar()
plt.show()