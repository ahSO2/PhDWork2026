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

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(mask_path, -1)
        plt.imshow(mask)
        plt.show()
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        plt.imshow(image)
        plt.show()
    return image
def calculate_diff(bandA, bandB):
    bandA_copy = bandA.copy()
    bandA_copy = np.ma.masked_where(bandB == 0, bandA_copy)
    diff = np.ma.abs(bandA_copy - bandB)
    bandB_zero = np.where(bandB == 0, 3, 0).astype(np.float32)
    edge_mask = cv2.blur(bandB_zero, (11, 11))
    edge_mask[:, 0] = 1
    edge_mask[:, -1] = 1
    edge_mask[0, :] = 1
    edge_mask[-1, :] = 1
    diff = np.ma.masked_where(edge_mask > 0, diff)
    return diff

def calculate_rough_AA(bandA, bandB):
    bandA_copy = bandA.copy()
    bandA_copy = np.ma.masked_where(bandB == 0, bandA_copy)
    ratio = np.ma.divide(bandA_copy.astype(np.float32), bandB.astype(np.float32))
    ratio = -1 * np.ma.log(ratio)
    bandB_zero = np.where(bandB == 0, 3, 0).astype(np.float32)
    edge_mask = cv2.blur(bandB_zero, (11, 11))
    edge_mask[:, 0] = 1
    edge_mask[:, -1] = 1
    edge_mask[0, :] = 1
    edge_mask[-1, :] = 1
    ratio = np.ma.masked_where(edge_mask > 0, ratio)
    return ratio
def check_registration(image_A, image_B):
    #difference = np.array(image_A, dtype='float32') - np.array(image_B, dtype='float32')
    difference = calculate_diff(np.array(image_A, dtype='float32'), np.array(image_B, dtype='float32'))
    plt.imshow(difference)
    plt.colorbar()
    plt.show()

    AA = calculate_rough_AA(image_A.astype(np.float32), image_B.astype(np.float32))
    plt.imshow(AA)
    plt.colorbar()
    plt.show()

img_A = cv2.imread("DataToView/2023-06-01T032305_fltrA_1ag_699997ss_Plume.png", -1)
img_B = cv2.imread("DataToView/2023-06-01T032305_fltrB_1ag_59979ss_Plume.png", -1)
clear_A = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/SelectedClears/Merapi/AveragedClearA_DarkCorrected.png", -1)
clear_B = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/SelectedClears/Merapi/AveragedClearB_DarkCorrected.png", -1)


#img_B = np.concatenate([np.ones(shape=(300, 648)), img_B[:-300, :]], axis=0)
#points_A = np.float32([[226, 373], [455,428], [220, 192], [555,242]])
#points_B = np.float32([[232, 300], [456,352], [226, 120], [554,169]])

plt.imshow(img_A)
plt.show()
plt.imshow(img_B)
plt.show()

vin_mask_A = clear_A.astype(np.float32)/np.max(clear_A)
img_A = np.divide(img_A, vin_mask_A)
vin_mask_B = clear_B.astype(np.float32)/np.max(clear_B)
img_B = np.divide(img_B, vin_mask_B)

img_A = mask_sensor_marks(img_A, "C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/Merapi_5A.png")
img_B = mask_sensor_marks(img_B, "C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/Merapi_0B_V2.png")


#visualise_registration_points(img_A, img_B, points_A, points_B)

#trans_matrix = cv2.getPerspectiveTransform(points_B, points_A)
trans_matrix = np.array([[ 9.8717630e-01, -3.2353610e-02, -4.5308247e+00],
 [ 5.5157226e-03,  9.7894287e-01,  5.5395761e+00],
 [ 6.2194173e-05, -9.1731679e-05,  1.0000000e+00]])
transformed_B = cv2.warpPerspective(img_B, trans_matrix, dsize=(648,486))


#plt.imshow(transformed_B)
#plt.show()


check_registration(img_A, transformed_B)
