import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyplis

'''Note this script is based on Tom's code in the PyCam Permanent software,
specifcally the ImageRegistration class. Calculate the registration transformation
based on the manually provided points, and also an optimised version computed 
with the cv2.findtransformECC function. For each transform, plot the difference 
in pixel value between the band A image and each warped version, and the resulting
absorbance image. '''

band_A_path = "DataToView/2023-05-07T210525_fltrA_1ag_400000ss_Plume.png"
band_B_path = "DataToView/2023-05-07T210525_fltrB_1ag_199994ss_Plume.png"

clear_A = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/SelectedClears/Kilauea/View1_NoClear.png", -1)
clear_B = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/SelectedClears/Kilauea/View1_NoClear.png", -1)

#Initial estimate of the registration transform
registration_points_A = np.float32([[206,375], [455,428], [230,221], [555,242]])
registration_points_B = np.float32([[207,304], [456,352], [230,147], [554,169]])

#Reventador June 2025 dictionary
#registration_points_A = np.float32([[88, 434], [327, 297], [434, 331], [499, 368]])
#registration_points_B = np.float32([[82, 433], [323, 296], [431, 331], [500, 369]])

def visualise_registration_points(image_A, image_B, points_A, points_B):
    A_to_plot = image_A.copy()
    B_to_plot = image_B.copy()
    for point in points_A:
        A_to_plot = cv2.circle(A_to_plot, center = (int(point[0]), int(point[1])), radius=3, color=(800,0,0), thickness=-1)
    for point in points_B:
        B_to_plot = cv2.circle(B_to_plot, center = (int(point[0]), int(point[1])), radius=3, color=(800,0,0), thickness=-1)

    fig, axs = plt.subplots(1, 2, figsize = (12, 8))
    A_plot = axs[0].imshow(A_to_plot, cmap='gray')
    axs[0].set_title("Band A Image")
    B_plot = axs[1].imshow(B_to_plot, cmap='gray')
    axs[1].set_title("Band B Image")
    #plt.savefig("C:/Users/ggp24ash/Documents/VolcanoData/Matched Corrected and Registered/TransformPoints.png")
    plt.show()
def convert_to_UINT8(image):
    if np.max(image) > 255:
        scale_factor = 255/np.max(image)
        scaled_image = image * scale_factor
        return scaled_image.astype(np.uint8)
    else:
        return image.astype(np.uint8)

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

n=500
eps=1e-10

bandA = cv2.imread(band_A_path, -1)
bandB = cv2.imread(band_B_path, -1)
plt.imshow(bandB)
plt.show()

if type(clear_A) == np.ndarray:
    vin_mask_A = clear_A.astype(np.float32)/np.max(clear_A)
    bandA = np.divide(bandA, vin_mask_A)
    vin_mask_B = clear_B.astype(np.float32)/np.max(clear_B)
    bandB = np.divide(bandB, vin_mask_B)
    plt.imshow(vin_mask_B)
    plt.show()
    plt.imshow(bandB)
    plt.show()

#bandA = mask_sensor_marks(bandA, "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Images/Data_Updated23rdJune26/SensorMarkMasks/Kilauea_1A.png")

visualise_registration_points(convert_to_UINT8(bandA), convert_to_UINT8(bandB), registration_points_A, registration_points_B)

bandA_f32 = bandA.astype(np.float32)
bandB_f32 = bandB.astype(np.float32)

#warp_matrix = np.eye(3,3, dtype=np.float32) #Initialise the transformation matrix
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n, eps)

initial_matrix = cv2.getPerspectiveTransform(registration_points_B, registration_points_A)
print("Manual matrix:")
print(initial_matrix)

value, optimal_matrix = cv2.findTransformECC(templateImage=bandB_f32, inputImage=bandA_f32, warpMatrix=initial_matrix.astype(np.float32),
                     motionType=cv2.MOTION_HOMOGRAPHY, criteria=criteria)


print("Optimised matrix")
print(optimal_matrix)
trans_B_manual = cv2.warpPerspective(bandB_f32, initial_matrix, dsize=(648,486))
trans_B_optim = cv2.warpPerspective(bandB_f32, optimal_matrix, dsize=(648,486))

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

diff_man = calculate_diff(bandA_f32, trans_B_manual)
diff_optim = calculate_diff(bandA_f32, trans_B_optim)
joined = np.concatenate([diff_man, diff_optim], axis=1)
joined_mask = np.concatenate([diff_man.mask, diff_optim.mask], axis=1)
plt.imshow(np.ma.masked_where(joined_mask, joined))
plt.colorbar()
plt.title("LHS: Manual points, RHS: Optimal Correlation")
plt.show()
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

ratio_man = calculate_rough_AA(bandA_f32, trans_B_manual)
ratio_optim = calculate_rough_AA(bandA_f32, trans_B_optim)

joined_ratio = np.concatenate([ratio_man, ratio_optim], axis=1)
joined_mask = np.concatenate([ratio_man.mask, ratio_optim.mask], axis=1)
plt.imshow(np.ma.masked_where(joined_mask, joined_ratio), cmap="YlGnBu_r")
plt.title("LHS: Manual Keypoints; RHS: Optimised Warp")
plt.colorbar()
plt.show()

