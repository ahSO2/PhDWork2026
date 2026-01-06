import cv2
import matplotlib.pyplot as plt

image_A = cv2.imread("ToView/2023-10-04T224830_fltrA_1ag_760521ss_Plume.png", -1)
image_B = cv2.imread("ToView/2023-10-04T224830_fltrB_1ag_79991ss_Plume.png", -1)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(image_A, cmap='gray')
axs[1].imshow(image_B, cmap='gray')
plt.show()