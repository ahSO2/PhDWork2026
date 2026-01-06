#Read in the list of updated labels with rain in band B

#For each relevant image, go to the relevant updated label
#and remove rain from the updated masks
#then overwrite the updated mask
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()
def labelme_json_to_dataset(json_path):
    os.system("labelme_export_json "+json_path+" -o "+json_path.replace(".","_"))

def map_line_index_to_channel(line_index):
    #Map the line number in the text file to
    #corresponding colour channel number for the label
    if line_index == 2:
        #First label (second line, after background line)
        #Is red, which is third channel in the BGR label png
        channel = 2
    if line_index == 3:
        #Second label is green
        #Which is the middle channel in BGR label png
        channel = 1
    return channel

def extract_json_folders(folder_name):
    for file_name in os.listdir(folder_name):
        labelme_json_to_dataset(folder_name + "/" + file_name)

#extract_json_folders("RainyBandBLabels_ForUpdatedMasks")

def convert_json_labels(folder_name):
    #for each JSON label folder
    #NOTE I already ran the conversion of all the json files to png
    for json_folder in os.listdir(folder_name):
        print(json_folder)
        #print(folder_name + "/" + json_folder)
        if os.path.isdir(folder_name + "/" + json_folder):
        #Read the label image as numpy
            label_subfolder_name = json_folder.split(".")[0]
            label_png = cv2.imread(folder_name + "/" + label_subfolder_name + "/label.png")

            #print(label_subfolder_name)


            #Read each label name in the text file:
            label_file_path = folder_name + "/" + label_subfolder_name + "/label_names.txt"
            with open(label_file_path, 'r') as file:
                line_index = 1
                for line in file:
                    category = line.strip()
                    print(category)
                    if category == "_background_":
                        print("Ignoring background")
                        pass
                    elif category == "Precip":
                        print("Saving precip pixels")
                        #Take the channel that matches the line index
                        #and save it in the plume mask
                        channel_index = map_line_index_to_channel(line_index)
                        on_lens_pixels = label_png[:,:,channel_index]
                        on_lens_pixels = np.where(on_lens_pixels > 0, 1, 0)
                        #show(on_lens_pixels)
                        save_name = label_subfolder_name[:-5] + ".npy"
                        #print(save_name)
                        np.save(folder_name + "/" + label_subfolder_name + "/" + save_name, on_lens_pixels)
                    line_index += 1

#convert_json_labels("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/RainyBandBLabels_ForUpdatedMasks")


#Now for each folder of precipitation labels, read the precipitaion numpy file
#and read the corrected version of the plume and exp pixel masks
#remove the rainy pixels
#then overwrite the mask


precip_labels_folder = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/RainyBandBLabels_ForUpdatedMasks"
images_to_remove_precip = pd.read_excel("LabelsWithBandBRain.xlsx")
all_labelled_images = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList.xlsx")
corrected_plume_masks_folder = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ReviewAndUpdateLabels"


all_image_names_B = all_labelled_images["image_name_B"].tolist()
all_corr_volcano_names = all_labelled_images["volcano_name"].tolist()
all_corr_image_names_A = all_labelled_images["image_name"].tolist()

for image_name_B in images_to_remove_precip["image_name_B"]:

    json_folder_name = image_name_B[:-4] + "_json"
    #Need to check if the mask exists, because for some of the samples I decided not to remove any precip
    precip_mask_path = precip_labels_folder + "/" + json_folder_name + "/" + json_folder_name[:-5] + ".npy"
    if os.path.isfile(precip_mask_path):
        precip_mask = np.load(precip_mask_path)
        #show(precip_mask)

        #Load the corrected mask (this is in a folder separated by volcano, so I
        #need to map to the volcano name)
        image_name_index = all_image_names_B.index(image_name_B)
        volcano_name = all_corr_volcano_names[image_name_index]
        image_name_A = all_corr_image_names_A[image_name_index]

        reviewed_segmentation_labels = np.load(corrected_plume_masks_folder + "/" + volcano_name + "/ProcessedLabels/PlumeAndExpPixels_" + image_name_A[:-4] + ".npy")
        print(reviewed_segmentation_labels.shape)

        #show(reviewed_segmentation_labels[0,:,:])

        reviewed_segmentation_labels[0,:,:] = np.where(precip_mask>0, 0, reviewed_segmentation_labels[0,:,:])
        #show(reviewed_segmentation_labels[0, :, :])

        #show(reviewed_segmentation_labels[1, :, :])
        reviewed_segmentation_labels[1,:,:] = np.where(precip_mask>0, 0, reviewed_segmentation_labels[1,:,:])
        #show(reviewed_segmentation_labels[1, :, :])

        #TODO save the updated reviewed segmentaiton labels.
        np.save(corrected_plume_masks_folder + "/" + volcano_name + "/ProcessedLabels/PlumeAndExpPixels_" + image_name_A[:-4] + ".npy", reviewed_segmentation_labels)
