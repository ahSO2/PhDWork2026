import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sigfig import round
import matplotlib.pyplot as plt
import VolcDictionaryWithCorrectClears

def show(image):
    plt.imshow(image, cmap="gray")
    plt.show()

volcano_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary("MerapiView1")

#For the folder with all data samples
folder_to_correct = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal/"
shared_drive_path = "X:/pering_group/Shared/Merapi/2023-06-01/"

dark_path_B = volcano_dictionary['dark_path_B']
dark_names_B = os.listdir(dark_path_B)
dark_ss_list_B = []
sig_figs_for_dark_ss = volcano_dictionary['sig_figs_for_dark_ss']
for file_name in dark_names_B:
    shutter_speed = int(file_name.split("_")[3][:-2])
    dark_ss_list_B.append(round(shutter_speed, sigfigs=sig_figs_for_dark_ss))

clear_sky_image_B = cv2.imread(volcano_dictionary['clear_sky_path_B'], -1)
reg_trans_matrix = cv2.getPerspectiveTransform(volcano_dictionary['registration_points_B'],
                                                   volcano_dictionary['registration_points_A'])

#For every bandB image name containing "Merapi_2023-06-01"
for file_name in os.listdir(folder_to_correct):
    if "Merapi_2023-06-01" in file_name:
        if "fltrB" in file_name:
            # Read the image from the shared drive
            raw_img = cv2.imread(shared_drive_path + "_".join(file_name.split("_")[1:]), -1)

            #Correct it using the updated registration transform
            #In the same way as the original read-in

            shutter_speed = int(file_name.split("_")[4][:-2])
            shutter_speed = round(shutter_speed, sigfigs=sig_figs_for_dark_ss)

            #Dark correct
            matching_dark_image_index_B = dark_ss_list_B.index(shutter_speed)
            matching_dark_image_B = cv2.imread(dark_path_B + "/" + dark_names_B[matching_dark_image_index_B], -1)

            corr_image = raw_img - matching_dark_image_B

            # vignette correct

            vin_mask_B = clear_sky_image_B / clear_sky_image_B.max()
            #show(vin_mask_B)
            corr_image = np.divide(corr_image, vin_mask_B).astype('uint16')
            #show(band_B_image)

            # register band B
            corr_image = cv2.warpPerspective(corr_image, reg_trans_matrix, (648, 486))
            #show(corr_image)

            success = cv2.imwrite(folder_to_correct + file_name, corr_image)
            print(success)
