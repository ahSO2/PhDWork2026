#For each location, go through all the samples that were
#selected to get their masks corrected.
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#If there is rain in band B, then add the image name to a list

updated_labels_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ReviewAndUpdateLabels/"
data_folder = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
all_labelled_images_list = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList.xlsx")

all_band_A_names = all_labelled_images_list["image_name"].tolist()
all_band_B_names = all_labelled_images_list["image_name_B"].tolist()

for location in ["Lascar"]:
    all_corrected_samples = pd.read_excel(updated_labels_path + location + "/LabelsToUpdate.xlsx")
    all_corrected_samples_names = all_corrected_samples["image_name"]

    for image_name in all_corrected_samples_names:
        print(image_name)
        #Match to the bandB name
        image_name_index = all_band_A_names.index(image_name)
        image_name_B = all_band_B_names[image_name_index]

        image_B = cv2.imread(data_folder + image_name_B, -1)

        #TODO show the band b image with plume and exp labels, then the plain band B image on the right

        matching_labels = np.load(updated_labels_path + location + "/ProcessedLabels/" + "PlumeAndExpPixels_" + image_name[:-4] + ".npy")

        plume_mask = matching_labels[0, :, :]
        exp_mask = matching_labels[1, :, :]
        plume_and_exp_mask = plume_mask + exp_mask
        #plt.imshow(plume_and_exp_mask)
        #plt.colorbar()
        #plt.show()
        masked_img = np.where(plume_and_exp_mask > 0, 0, image_B)

        fig, axs = plt.subplots(1, 2)
        masked_plot = axs[0].imshow(masked_img, cmap='gray')
        original_plot = axs[1].imshow(image_B, cmap='gray')
        axs[0].set_title("Masked")
        axs[1].set_title("Original")
        #plt.show()
        plt.savefig("C:/Users/ggp24ash/Documents/Scratch Data/Re-drawnSegmentationMasksBandB/" + image_name_B, dpi=fig.dpi)