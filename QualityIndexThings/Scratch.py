import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


smm = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/Lastarria_B.png", -1)
smm[62:72, 371:380] = 1
plt.imshow(smm)
plt.show()
cv2.imwrite("C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/Lastarria_B_V2.png", smm)