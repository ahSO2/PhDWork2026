import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

############################### Things to edit manually
#Read in a label file
label_path = "AllLabelledImagesList_UpdatedClassifications.xlsx"
mod = 1
folder_path_to_save = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026_Temporal/"
sensor_mark_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/"
############################## Full script
all_dictionary_names = []
for dictionary in VolcDictionaryWithCorrectClears.all_dictionaries:
    all_dictionary_names.append(dictionary['volcano_dictionary_name'])

def find_dictionary_index(dictionary_name):
    index = all_dictionary_names.index(dictionary_name)
    return index

def set_up_dark_lists(band):
    dark_ss_lists = {}
    dark_names_lists = {}
    for dictionary in VolcDictionaryWithCorrectClears.all_dictionaries:
        dark_ss_list = []
        #Create a list of the dark shutter speeds
        if band == "A":
            dark_path = dictionary["dark_path_A"]
        elif band == "B":
            dark_path = dictionary["dark_path_B"]
        sig_figs = dictionary["sig_figs_for_dark_ss"]
        dark_names = os.listdir(dark_path)
        for file_name in dark_names:
            shutter_speed = int(file_name.split("_")[3][:-2])
            dark_ss_list.append(round(shutter_speed, sig_figs))
        dictionary_name = dictionary["volcano_dictionary_name"]
        dark_ss_lists[dictionary_name + "_band" + band] = dark_ss_list
        dark_names_lists[dictionary_name + "_band" + band] = dark_names
    return dark_ss_lists, dark_names_lists

registration_transforms = {}
for dictionary in VolcDictionaryWithCorrectClears.all_dictionaries:
    dictionary_name = dictionary["volcano_dictionary_name"]
    reg_trans_matrix = dictionary["registration_matrix"]
    registration_transforms[dictionary_name] = reg_trans_matrix

def mask_sensor_marks(image, mask_path):
    if "None" in mask_path:
        pass
    else:
        mask = cv2.imread(mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

labels = pd.read_excel(label_path)

def plot_two_same_colorbar(im1, im2, title=None, cmap="gray"):
    fig, axs = plt.subplots(ncols=2)
    joined = np.concatenate([im1, im2])
    max = np.max(joined)
    min = np.min(joined)
    plot1 = axs[0].imshow(im1, cmap=cmap, vmax=max, vmin=min)
    plot2 = axs[1].imshow(im2, cmap=cmap, vmax=max, vmin=min)
    fig.colorbar(plot2, ax = [axs[0], axs[1]], location="bottom", shrink=0.7, pad=0.2)
    if title != None:
        fig.suptitle(title)
    plt.show()
def read_and_correct_column(column_name, band, correct = True, start_index=0, plot=False):
    dark_ss_lists, dark_names_lists = set_up_dark_lists(band)

    #For each labelled image:
    label_count = labels.shape[0]

    for image_index in range(start_index, label_count, mod):
        print("Image " + str(image_index) + " of " + str(label_count) + ":")
        image_name = labels[column_name][image_index]
        print("Reading " + image_name)
        # Read the corresponding volcano dictionary
        volcano_dictionary_name = labels["volcano_dictionary_name"][image_index]
        dictionary_index = find_dictionary_index(volcano_dictionary_name)
        volcano_dictionary = VolcDictionaryWithCorrectClears.all_dictionaries[dictionary_index]
        #Identify the correct file path:
        shared_drive_path = volcano_dictionary["shared_drive_folder_path"]
        include_year = volcano_dictionary["shared_drive_has_year_subfolders"]
        date = image_name.split("_")[1][:10]
        year = date[0:4]

        image_name_without_dictionary = "_".join(image_name.split("_")[1:])
        #Read the data from the shared drive.
        if include_year == "no":
            image_path = shared_drive_path + date
        if include_year == "yes":
            image_path = shared_drive_path + year + "/" + date

        #Check if the path exists
        #If it does read the data
        image_read_succesfully = False
        print(image_path + "/" + image_name_without_dictionary)
        if os.path.exists(image_path + "/" + image_name_without_dictionary) == True:
            image= cv2.imread(image_path + "/" + image_name_without_dictionary, -1)
            image_read_succesfully = True
        else:
            #If its not there, try looking in other directories
            for sub_dir in os.walk(image_path):
                if image_read_succesfully == False:
                    print("Checking if image is in subdir:" + sub_dir[0])
                    if os.path.exists(sub_dir[0] + "/" + image_name_without_dictionary):
                        image = cv2.imread(sub_dir[0] + "/" + image_name_without_dictionary, -1)
                        image_read_succesfully = True
                        print("Image read from: " + sub_dir[0] + "/" + image_name_without_dictionary)
        if image_read_succesfully == False:
            print("ERROR: couldn't find image to read.")
        #Correct the image
        if correct == True:
            ##################### Dark Correction
            if band == "A":
                dark_path = volcano_dictionary["dark_path_A"]
            elif band == "B":
                dark_path = volcano_dictionary["dark_path_B"]
            sig_figs = volcano_dictionary["sig_figs_for_dark_ss"]
            dark_names = dark_names_lists[volcano_dictionary_name + "_band" + band]
            dark_ss_list = dark_ss_lists[volcano_dictionary_name + "_band" + band]

            shutter_speed = int(image_name.split("_")[4][:-2])
            shutter_speed = round(shutter_speed, sig_figs)

            if shutter_speed in dark_ss_list:
                matching_dark_image_index = dark_ss_list.index(shutter_speed)
                matching_dark_image = cv2.imread(dark_path + "/" + dark_names[matching_dark_image_index], -1)

            if plot == True:
                copy_to_plot = image.copy()

            image = (image.astype("float32") - matching_dark_image.astype("float32"))
            image = np.where(image > 0, image, 0)

            if plot == True:
                plot_two_same_colorbar(copy_to_plot, image, title="Dark correction")

            ################Vignette correction
            if band == "A":
                clear_path = volcano_dictionary["clear_sky_path_A"]
            elif band == "B":
                clear_path = volcano_dictionary["clear_sky_path_B"]

            clear = cv2.imread(clear_path, -1)

            if plot == True:
                copy_to_plot = image.copy()
            vin_mask = clear / clear.max()
            image = np.divide(image, vin_mask).astype('uint16')
            if plot == True:
                plot_two_same_colorbar(copy_to_plot, image, title="Vignette correction")

            #Mask sensor marks:
            if band == "A":
                smmp = volcano_dictionary["sensor_marks_mask_A"]
            else:
                smmp = volcano_dictionary["sensor_marks_mask_B"]
            if plot == True:
                copy_to_plot = image.copy()
            image = mask_sensor_marks(image, sensor_mark_masks_path + smmp)
            if plot == True:
                plot_two_same_colorbar(copy_to_plot, image, title="Masking Sensor Marks")

            #If the resulting intensity is above 1023, scale it back down
            if image.max() > 1023:
                scale_factor = image.max()/1023
                if plot == True:
                    copy_to_plot = image.copy()
                image = (image/scale_factor).astype("uint16")
                if plot == True:
                    plot_two_same_colorbar(copy_to_plot, image, title="Scaling back to 10bit")

                print("Scaled image back to range: [0,1023]")

            ######################## If its a band B image, register it
            if band == "B":
                if plot == True:
                    copy_to_plot = image.copy()
                transform = registration_transforms[volcano_dictionary_name]
                image = cv2.warpPerspective(image, transform, (648, 486))
                if plot == True:
                    plot_two_same_colorbar(copy_to_plot, image, title="Registration")

        cv2.imwrite(folder_path_to_save +  image_name, image)


columns_dictionary = {"prev_min_name":"A",
                      "prev_min_name_B":"B"}

for column in columns_dictionary:
    column_to_read = column
    band = columns_dictionary[column]
    print("Correcting column: " + column)
    print("Band: " + band)
    read_and_correct_column(column_name=column_to_read, band = band, correct=True, start_index=0, plot=False)
