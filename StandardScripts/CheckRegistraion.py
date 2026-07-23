import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualise_registration_points(image_A, image_B, points_A, points_B):
    for point in points_A:
        image_A = cv2.circle(image_A, center = (int(point[0]), int(point[1])), radius=3, color=(800,0,0), thickness=-1)
    for point in points_B:
        image_B = cv2.circle(image_B, center = (int(point[0]), int(point[1])), radius=3, color=(800,0,0), thickness=-1)

    fig, axs = plt.subplots(1, 2, figsize = (12, 8))
    A_plot = axs[0].imshow(image_A, cmap='gray')
    axs[0].set_title("Band A Image")
    B_plot = axs[1].imshow(image_B, cmap='gray')
    axs[1].set_title("Band B Image")
    #plt.savefig("C:/Users/ggp24ash/Documents/VolcanoData/Matched Corrected and Registered/TransformPoints.png")
    plt.show()

def check_registration(image_A, image_B):
    difference = np.array(image_A, dtype='float32') - np.array(image_B, dtype='float32')
    plt.imshow(difference[:,:])
    plt.colorbar()
    plt.show()

img_A = cv2.imread("DataToView/2023-09-21T022400_fltrA_1ag_760568ss_Plume.png", -1)
img_B = cv2.imread("DataToView/2023-09-21T022400_fltrB_1ag_79991ss_Plume.png", -1)

points_A = np.float32([[439,320], [359,266], [264,259], [116,310]])
points_B = np.float32([[455,324], [375,268], [277,260], [125,311]])

visualise_registration_points(img_A, img_B, points_A, points_B)

trans_matrix = cv2.getPerspectiveTransform(points_B, points_A)
transformed_B = cv2.warpPerspective(img_B, trans_matrix, dsize=(648,486))

plt.imshow(transformed_B)
plt.show()
plt.imshow(img_A)
plt.show()

check_registration(img_A, transformed_B)
