import cv2
import matplotlib.pyplot as plt
image = cv2.imread("Data/2022-05-28T180955_fltrA_1ag_499991ss_Plume.png",-1)
plt.imshow(image, cmap="gray")
plt.show()