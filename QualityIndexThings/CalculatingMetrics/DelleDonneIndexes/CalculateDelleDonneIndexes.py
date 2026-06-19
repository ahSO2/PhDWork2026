import os
import cv2
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings")
import VolcanoesDictionaryForQualityModels

########################
#Read in a dataset, and corresponding region information defined in the volcano
#dictionary. Then calculate the visibility and fog indexes.
########################

####NOTE: I have updated the calculation of the correlation index so that the areas
#where AA is undefined (where bandB is zero) are masked out of the correlation calculation.
#I had originally applied this manually based on the registration transformation angle
#for Lastarria data, but have updated so this is implemented automatically for any location.
#Therefore the number of pixels included in the cross section will be very slightly altered
#and the correlation index values obtained might differ very slightly from those originally
#calculated (and included in the first submission to Frontiers before peer reviews).

file_type = "excel"
data_names_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs/Lastarria_AllSamples_Unbalanced.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data/Lastarria - All Unbalanced/"
sensor_mark_masks_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data/SensorMarkMasks/"
save_folder = "IndexValues/"
flank_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"
def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

def show(image, title=None, colormap=None):
    if colormap == None:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image, cmap=colormap)
    plt.colorbar()
    if title != None:
        plt.title(title)
    plt.show()
def calculate_optical_depth(measured_img, bg_val):
    ratio = measured_img.astype("float32")/bg_val
    return -1 * np.log(ratio)

def calculate_fog_index(image_B, sky_corners, ground_corners):
    sky_rectangle = image_B[sky_corners[0][1]: sky_corners[1][1], sky_corners[0][0]:sky_corners[1][0]]
    ground_rectangle = image_B[ground_corners[0][1]: ground_corners[1][1], ground_corners[0][0]:ground_corners[1][0]]

    sky_mean = np.mean(sky_rectangle)
    ground_mean = np.mean(ground_rectangle)
    infilled_image = image_B.copy()
    infilled_image[sky_corners[0][1]: sky_corners[1][1], sky_corners[0][0]:sky_corners[1][0]] = np.ones_like(
        sky_rectangle) * sky_mean
    infilled_image[ground_corners[0][1]: ground_corners[1][1],
    ground_corners[0][0]:ground_corners[1][0]] = np.ones_like(ground_rectangle) * ground_mean
    #show(infilled_image)
    # plt.imsave(folder_for_img_results + "/FogIndexRegionsBandB.png", infilled_image, cmap="gray")

    fog_index = sky_mean / ground_mean
    return fog_index

def calculate_correlation_index(image_A, image_B, sky_corners, flank_mask, cross_section_height):
    sky_rectangle_A = image_A[sky_corners[0][1]: sky_corners[1][1], sky_corners[0][0]:sky_corners[1][0]]

    # Select the background estimation region:
    # By selecting the point in the sky region with the lowest intensity (I have applied this in band A)
    max_val = sky_rectangle_A[0, 0]
    max_x = 0
    max_y = 0
    for y in range(0, sky_rectangle_A.shape[0]):
        for x in range(0, sky_rectangle_A.shape[1]):
            if sky_rectangle_A[y, x] > max_val:
                max_x = x
                max_y = y
                max_val = sky_rectangle_A[y, x]

    # Draw a circle on the point in sky region with max 310nm value, and show the image:
    image_to_show = (image_A.astype("float32") / 4)
    scale_factor = 255 / image_to_show.max()
    image_to_show = (image_to_show * scale_factor).astype("uint8")
    r = 20
    cv2.circle(image_to_show, center=(max_x, max_y), radius=r, color=255, thickness=3)
    #show(image_to_show, title="Selected BG point")

    bg_circle_mask = np.zeros_like(image_to_show)
    cv2.circle(bg_circle_mask, center=(max_x, max_y), radius=r, color=1, thickness=-1)

    bg_pixels_A = np.where(bg_circle_mask == 1, image_A, 0)
    bg_pixels_B = np.where(bg_circle_mask == 1, image_B, 0)

    bg_pixels_A = np.ma.masked_where(bg_circle_mask == 0, bg_pixels_A)
    bg_pixels_B = np.ma.masked_where(bg_circle_mask == 0, bg_pixels_B)
    bg_val_A = np.ma.mean(bg_pixels_A)
    bg_val_B = np.ma.mean(bg_pixels_B)

    # Calculate the Apparent Absorbance
    od_A = calculate_optical_depth(image_A, bg_val_A)
    od_B = calculate_optical_depth(image_B, bg_val_B)
    AA = od_A - od_B
    #show(AA, colormap="YlGnBu_r")

    #Shifting the flank mask upwards to make sure it covers the volcano
    #Only needed for visualisation
    #flank_mask = flank_mask[4:, :]
    #flank_mask = np.concatenate((flank_mask, np.zeros(shape=(4, 648))))
    #flank_mask[-10:, :] = np.zeros_like(flank_mask[-10:, :])

    # Extract the cross-section
    height = cross_section_height
    cross_section_image = AA.copy()
    cross_section_image[height - 2:height + 3, :] = np.zeros(shape=(5, 648))
    #cross_section_image = np.where(flank_mask == 1, cross_section_image, 0)
    #plt.imshow(cross_section_image[10:470, 0:625], cmap="YlGnBu_r")
    #plt.show()
    #plt.imshow(flank_mask[10:470, 0:625].astype('bool'),alpha=np.ones_like(flank_mask[10:470, 0:625]) - flank_mask[10:470, 0:625], cmap='gray')
    #plt.show()
    #absorbance_save_path = folder_for_img_results + "/AbsorbanceCrossSection.png"
    #plt.savefig(absorbance_save_path)
    #plt.close()

    edge_to_mask = np.where(image_B==0, 5, 0)
    edge_to_mask = cv2.blur(edge_to_mask, ksize=(5,5))
    #show(edge_to_mask)
    AA_masked = np.ma.masked_where(edge_to_mask>0, AA)
    image_A_masked = np.ma.masked_where(edge_to_mask>0, image_A)
    #show(AA_masked)
    cross_section_AA = AA_masked[height,:].compressed()
    cross_section_A = image_A_masked[height,:].compressed()


    # Calculate the correlation
    correlation = pearsonr(cross_section_AA, cross_section_A).statistic
    return correlation

if file_type == "excel":
    labels = pd.read_excel(data_names_path)
else:
    labels = pd.read_csv(data_names_path)
image_names_A = labels["image_name"].tolist()
image_names_B = labels["image_name_B"].tolist()
volcano_dictionaries_list = labels["volcano_dictionary_name"].tolist()
visibility_indexes = []
correlation_indexes = []
for index in range(0, len(image_names_A)):
    image_name_A = image_names_A[index]
    print("Calculating for image:" + image_name_A)
    image_A = cv2.imread(data_path + image_name_A, -1)

    #folder_for_img_results = save_folder + image_name_A[:-4]
    #os.mkdir(folder_for_img_results)

    image_name_B = image_names_B[index]
    image_B = cv2.imread(data_path + image_name_B, -1)

    volcano_dictionary = VolcanoesDictionaryForQualityModels.map_dictionary_name_to_dictionary(volcano_dictionaries_list[index])
    sensor_mask_path_A = volcano_dictionary["sensor_marks_mask_A"]
    sensor_mask_path_B = volcano_dictionary["sensor_marks_mask_B"]
    image_A = mask_sensor_marks(image_A, sensor_mask_path_A)
    image_B = mask_sensor_marks(image_B, sensor_mask_path_B)
    #plt.imsave(folder_for_img_results + "/ImageA.png", image_A, cmap="gray")
    #plt.imsave(folder_for_img_results + "/ImageB.png", image_B, cmap="gray")

    sky_corners = volcano_dictionary["sky_rectangle"]
    ground_corners = volcano_dictionary["ground_rectangle"]
    fog_index = calculate_fog_index(image_B, sky_corners, ground_corners)
    visibility_indexes.append(fog_index)

    #Next to calculate the correlation index
    flank_mask = cv2.imread(flank_masks_path + volcano_dictionary["flank_mask_name"], -1)
    cross_section_height = volcano_dictionary["DD_cross_section_height"]
    correlation_index = calculate_correlation_index(image_A, image_B, sky_corners, flank_mask, cross_section_height)
    correlation_indexes.append(correlation_index)


labels["visibility_index"] = visibility_indexes
labels["correlation_index"] = correlation_indexes
if file_type == "excel":
    labels.to_excel(save_folder + data_names_path.split("/")[-1][:-5] + "_QualityIndexes.xlsx")
else:
    labels.to_csv(save_folder + data_names_path.split("/")[-1][:-4] + "_QualityIndexes.csv")