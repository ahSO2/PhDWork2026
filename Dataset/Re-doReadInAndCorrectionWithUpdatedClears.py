import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

sys.path.append("C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/VolcanoDictionaries")

# now we can import mod
# I have copied this dictionary directly into PhDProjectStep2>VolcDictionaryWithCorrectClears
import VolcDictionaryWithCorrectClears

############################### Things to edit manually
#Read in a label file
label_path = "AllLabelledImagesList.xlsx"
#label_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/ClusteringForNewLabels/AdditionalDataPool_AllLocations.xlsx"
mod = 1
folder_path_to_save = "AllData_CorrectedWithVolcDict2Temporal/"
#folder_path_to_save = "C:/Users/ggp24ash/Documents/Main Datasets/AdditionalDataPoolTemporal/"
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
    reg_trans_matrix = cv2.getPerspectiveTransform(dictionary['registration_points_B'],
                                                   dictionary['registration_points_A'])
    registration_transforms[dictionary_name] = reg_trans_matrix


labels = pd.read_excel(label_path)

def read_and_correct_column(column_name, band, correct = True, start_index=0):
    dark_ss_lists, dark_names_lists = set_up_dark_lists(band)

    #For each labelled image:
    label_count = labels.shape[0]
    #CotV2_indexes = labels[labels["volcano_dictionary_name"] == "CotopaxiView2"].index.tolist()
    #for image_index in CotV2_indexes:
        #print(labels["volcano_dictionary_name"][image_index])

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
        original_image = image
        #TODO correct the image
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

            image = (image.astype("float32") - matching_dark_image.astype("float32"))
            image = np.where(image > 0, image, 0)

            ################Vignette correction
            if band == "A":
                clear_path = volcano_dictionary["clear_sky_path_A"]
            elif band == "B":
                clear_path = volcano_dictionary["clear_sky_path_B"]

            clear = cv2.imread(clear_path, -1)

            vin_mask = clear / clear.max()
            image = np.divide(image, vin_mask).astype('uint16')

            #If the resulting intensity is above 1023, scale it back down
            if image.max() > 1023:
                scale_factor = image.max()/1023
                image = (image/scale_factor).astype("uint16")

                print("Scaled image back to range: [0,1023]")

            ######################## If its a band B image, register it
            if band == "B":
                transform = registration_transforms[volcano_dictionary_name]
                image = cv2.warpPerspective(image, transform, (648, 486))


        #plt.imshow(sensor_mark_mask)
        #plt.show()

        #fig, axs = plt.subplots(nrows=1, ncols=2)
        #original_plot = axs[0].imshow(original_image, cmap='gray')
        #axs[0].set_title("Original")
        #corrected_plot = axs[1].imshow(image, cmap='gray')
        #axs[1].set_title("Corrected")
        #fig.colorbar(original_plot, ax=axs[0], shrink=0.5)
        #fig.colorbar(corrected_plot, ax=axs[1], shrink=0.5)
        #plt.show()

        #if image.dtype != "uint16":
        #    print("Error in datatype")

        #if save_corrected_data = True:

        #Save the corrected data for use to make sensor mark masks:
        #UINT8_Img = (image/4).astype("uint8")
        #cv2.imwrite("DataToMakeNewSensorMarkMasks/BandB/" + image_name + ".png", UINT8_Img)
        cv2.imwrite(folder_path_to_save +  image_name, image)

        #TODO add a thing to read the image again and if its uint 8 then
        #view it and see whats going wrong
        #re_read = cv2.imread("C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/CorrectedLabelTest/" +  image_name, -1)
        #plt.imshow(re_read, cmap='gray')
        #plt.colorbar()
        #plt.show()



columns_dictionary = {"next_min_name":"A",
                      "next_min_name_B":"B"}

for column in columns_dictionary:
    column_to_read = column
    band = columns_dictionary[column]
    print("Correcting column: " + column)
    print("Band: " + band)
    read_and_correct_column(column_name=column_to_read, band = band, correct=True, start_index=0)
