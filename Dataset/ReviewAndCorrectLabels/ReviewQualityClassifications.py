#For ALL labelled images
#View bandA, bandB and the plume+exp pixels mask
#Scale whichever band image is darkest up
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/VolcDictionaryWithCorrectClears.py")
import VolcDictionaryWithCorrectClears


all_labelled_images = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList_UpdatedClassifications.xlsx")

image_names = all_labelled_images["image_name"].tolist()
image_names_B = all_labelled_images["image_name_B"].tolist()
batch_numers = all_labelled_images["labelling_batch_name"].tolist()
dictionary_names = all_labelled_images["volcano_dictionary_name"].tolist()

sensor_mark_masks_path = ("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/")

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + mask_path, -1)

        #smm_fig, smm_axs = plt.subplots(nrows=1, ncols=2)
        #smm_axs[0].imshow(image, cmap='gray')
        #smm_axs[1].imshow(mask, cmap='gray')
        #plt.show()

        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        #print(mask.dtype)
        #plt.imshow(mask, cmap = "gray")
        #plt.colorbar()
        #plt.show()
    return image

for image_index in range(0, len(image_names)):
    print(image_index)
    image_name = image_names[image_index]
    print(image_name)
    image_name_B = image_names_B[image_index]
    batch_name = batch_numers[image_index]

    image_A = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/" + image_name, -1)
    image_B = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/" + image_name_B, -1)
    updated_plume_mask = np.load("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/" + batch_name + "/PlumeAndExpPixels_" + image_name[:-4] + ".npy" )
    all_plume_and_exp_pixels_mask = updated_plume_mask[0,:,:] + updated_plume_mask[1,:,:]

    dictionary_name = dictionary_names[image_index]
    if dictionary_name == "MerapiTenthJune":
        dictionary_name = "MerapiTenthMay"
    elif dictionary_name == "MerapiSixteenthJune":
        dictionary_name = "MerapiSixteenthMay"

    volcano_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
    sensor_mask_path_A = volcano_dictionary["sensor_marks_mask_A"]
    sensor_mask_path_B = volcano_dictionary["sensor_marks_mask_B"]

    image_A = mask_sensor_marks(image_A, sensor_mask_path_A)
    image_B = mask_sensor_marks(image_B, sensor_mask_path_B)

    max_band_A = image_A.max()
    max_band_B = image_B.max()
    if max_band_A > max_band_B:
        scale_factor = max_band_A / max_band_B
        image_B = image_B * scale_factor
    elif max_band_B > max_band_A:
        scale_factor = max_band_B / max_band_A
        image_A = image_A * scale_factor

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    axs[0].imshow(image_A, cmap='gray')
    axs[0].set_title("BandA")
    axs[1].imshow(image_B, cmap='gray')
    axs[1].set_title("BandB")
    masked_for_plume = np.where(all_plume_and_exp_pixels_mask > 0, image_A.min(), image_A)
    axs[2].imshow(masked_for_plume, cmap='gray')
    axs[2].set_title("PlumeMask")
    fig.suptitle(image_name)
    plt.tight_layout()

    plt.show()

