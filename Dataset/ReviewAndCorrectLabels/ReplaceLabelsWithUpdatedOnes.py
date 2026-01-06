import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Copy the folder of labels to "ProcessedLabels_UpdatedAfterReview"

updated_labels = "C:/users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ReviewAndUpdateLabels/"
all_labelled_images_list = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList.xlsx")

all_image_names = all_labelled_images_list["image_name"].tolist()
all_batch_numbers = all_labelled_images_list["labelling_batch_name"].tolist()


#For each location
for location in os.listdir(updated_labels):
    location_updated_labels = pd.read_excel(updated_labels + location + "/LabelsToUpdate.xlsx")
    #For each updated label
    for image_name in location_updated_labels["image_name"]:
        #View the label
        print(updated_labels + location + "/ProcessedLabels/PlumeAndExpPixels_" + image_name[:-4] + ".npy")
        updated_label = np.load(updated_labels + location + "/ProcessedLabels/PlumeAndExpPixels_" + image_name[:-4] + ".npy")
        #plt.imshow(updated_label[0,:,:])
        #plt.show()

        image_name_index = all_image_names.index(image_name)
        batch_number = all_batch_numbers[image_name_index]
        original_label_path = "C:/users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/" + batch_number + "/PlumeAndExpPixels_" + image_name[:-4] + ".npy"
        original_label = np.load(original_label_path)
        print(original_label.shape)
        #plt.imshow(original_label[0, :, :])
        #plt.show()

        band_A_image = cv2.imread("C:/users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/" + image_name, -1)

        fig, axs = plt.subplots(nrows=1, ncols=2)
        masked_with_original = np.where(original_label[0,:,:] > 0, 0, band_A_image)
        axs[0].imshow(masked_with_original)
        masked_with_updated = np.where(updated_label[0,:,:]>0, 0, band_A_image)
        axs[1].imshow(masked_with_updated)
        plt.show()



        #Overwrite it in the updated processed labels folder
        #np.save(original_label_path, updated_label)
        #TODO now that I have run this, both subplots should look the same
        #TODO and they do! :)