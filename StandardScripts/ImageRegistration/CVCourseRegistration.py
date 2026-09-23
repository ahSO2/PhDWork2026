import cv2
import matplotlib.pyplot as plt
import numpy as np

#Code from Computer Vision Module Week 8 Lab:
band_A_path = "DataToView/2024-04-20T133950_fltrA_1ag_1999900ss_Plume.png"
band_B_path = "DataToView/"
n=20
def convert_to_UINT8(image):
    if np.max(image) > 255:
        scale_factor = 255/np.max(image)
        scaled_image = image * scale_factor
        return scaled_image.astype(np.uint8)
    else:
        return image.astype(np.uint8)

def visualise_registration_points(image_A, image_B, points_A, points_B):
    for point in points_A:
        image_A = cv2.circle(image_A, center = (int(point[0][0]), int(point[0][1])), radius=3, color=(800,0,0), thickness=-1)
    for point in points_B:
        image_B = cv2.circle(image_B, center = (int(point[0][0]), int(point[0][1])), radius=3, color=(800,0,0), thickness=-1)

    fig, axs = plt.subplots(1, 2, figsize = (12, 8))
    A_plot = axs[0].imshow(image_A, cmap='gray')
    axs[0].set_title("Band A Image")
    B_plot = axs[1].imshow(image_B, cmap='gray')
    axs[1].set_title("Band B Image")
    #plt.savefig("C:/Users/ggp24ash/Documents/VolcanoData/Matched Corrected and Registered/TransformPoints.png")
    plt.show()

bandA = cv2.imread(band_A_path, -1)
bandA = convert_to_UINT8(bandA)
bandB = cv2.imread(band_B_path, -1)
bandB = convert_to_UINT8(bandB)

sift = cv2.SIFT_create()

#Detect keypoints and calculate the feature descriptor values
kp1, des1 = sift.detectAndCompute(bandA, None) #Input image, and optionally a mask for the area where you want to detect
kp2, des2 = sift.detectAndCompute(bandB, None)

m = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = m.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(bandA, kp1, bandB, kp2, matches[:n], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("SIFT Matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Now calculate the transform based on these points:
points_bandA = []
points_bandB = []
index = 0
for match in matches[:n]:
    index1 = match.queryIdx
    index2 = match.trainIdx
    point1 = kp1[index1].pt
    point2 = kp2[index2].pt
    p1 = np.array([point1[0], point1[1]])
    p2 = np.array([point2[0], point2[1]])
    points_bandA.append(p1)
    points_bandB.append(p2)
    index += 1

points_bandA = np.stack(points_bandA, axis=0)
points_bandB = np.stack(points_bandB, axis=0)
points_bandA = points_bandA.reshape(-1, 1, 2)
points_bandB = points_bandB.reshape(-1, 1, 2)

matrix, mask = cv2.findHomography(points_bandB, points_bandA, cv2.RANSAC)

trans_B = cv2.warpPerspective(bandB, matrix, dsize=(648,486))

visualise_registration_points(bandA, bandB, points_bandA, points_bandB)

plt.imshow(trans_B, cmap="gray")
plt.colorbar()
plt.show()

diff = np.abs(trans_B.astype(np.float32) - bandA.astype(np.float32))
plt.imshow(diff)
plt.colorbar()
plt.show()

#TODO note this is converting the images into 8-bit




