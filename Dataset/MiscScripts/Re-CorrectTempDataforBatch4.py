import cv2
import matplotlib.pyplot as plt
import pandas as pd
import VolcanoDictionaries

all_labels = pd.read_excel("AllLabelledImagesList.xlsx")
batch_4_labels = all_labels[all_labels["labelling_batch_name"] == "Batch4"]
data_path_1 = "X:/volcano_cameras/Shared/Cotopaxi/2023/2023-06-19"
data_path_2 = "X:/volcano_cameras/Shared/Cotopaxi/2023/2023-07-27"

print(batch_4_labels.shape)

names_to_correct = batch_4_labels["image_name_B"]


def correct_image(image_name, data_path, volcano_dictionary):
    if "fltrA" in image_name:
        band = "A"
    else:
        band = "B"

    dark_path = volcano_dictionary['dark_path_' + band]
    dark_names = os.listdir(dark_path)
    sig_figs_for_dark_ss = volcano_dictionary['sig_figs_for_dark_ss']
    dark_ss_list = []
    for file_name in dark_names:
        shutter_speed = int(file_name.split("_")[3][:-2])
        dark_ss_list.append(round(shutter_speed, sigfigs=sig_figs_for_dark_ss))

    clear_sky_image = cv2.imread(volcano_dictionary['clear_sky_path_' + band], -1)

    if band == "B":
        reg_trans_matrix = cv2.getPerspectiveTransform(volcano_dictionary['registration_points_B'],
                                                   volcano_dictionary['registration_points_A'])

    image_shutter_speed = int(image_name.split("_")[3][:-2])
    image_shutter_speed = round(image_shutter_speed, sigfigs=sig_figs_for_dark_ss)

    if shutter_speed in dark_ss_list:
        matching_dark_image_index = dark_ss_list.index(shutter_speed)
        matching_dark_image = cv2.imread(dark_path + "/" + dark_names[matching_dark_image_index], -1)
    else:
        print("No matching dark image!")

    # If so, continue with corrections:
    image = cv2.imread(data_path + "/" + image_name[9:], -1)
    image = image - matching_dark_image

    # vignette correct
    vin_mask = clear_sky_image / clear_sky_image.max()
    plt.imshow(vin_mask_A)
    image = np.divide(image, vin_mask).astype('uint16')

    # register band B
    if band == "B":
        image = cv2.warpPerspective(image, reg_trans_matrix, (648, 486))

    plt.imshow(image, cmap='gray')

for image_name in names_to_correct:
    day = image_name.split("_")[1][0:10]
    if day == "2023-06-19":
        data_path = data_path_1
        dictionary = VolcanoDictionaries.Cotopaxi_View3_dictionary
    else:
        data_path = data_path_2
        dictionary = VolcanoDictionaries.Cotopaxi_View4_dictionary
    #image = cv2.imread(data_path + "/" + image_name[9:], -1)
    #plt.imshow(image, cmap="gray")
    #plt.show()
    #TODO correct the data, in the same way as the original read in func
    correct_image(image_name, data_path, )
    #TODO save the corrected data to the Temporal Data folder
