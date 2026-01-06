'''Aim of this script: to be able to step through data and if requested
correct and save specific images (along with timestep data).'''

import cv2
import os
import numpy as np
import pandas as pd

from sigfig import round
import matplotlib.pyplot as plt
import VolcanoDictionaries

def convert_hhmmss_to_s(time_in_hhmmss):
    time_in_s = int(time_in_hhmmss[-2:])
    time_in_s = time_in_s + int(time_in_hhmmss[-4:-2]) * 60
    time_in_s = time_in_s + int(time_in_hhmmss[:2]) * 3600
    return time_in_s

def find_match_and_correct(day_path, band_A_name, band_B_names, band_B_times, sig_figs_for_dark_ss, dark_path_A, dark_names_A, dark_ss_list_A, dark_path_B, dark_names_B, dark_ss_list_B, clear_sky_image_A, clear_sky_image_B, reg_trans_matrix):
    matched_B = False
    matching_dark_A = False
    matching_dark_B = False
    match_found_and_corrected = False
    corrected_A = None
    corrected_B = None

    # Find the matching band B
    band_A_time = int(band_A_name.split('_')[0][-6:])
    if band_A_time in band_B_times:
        matching_B_index = band_B_times.index(band_A_time)
        matching_B_image_name = band_B_names[matching_B_index]
        matched_B = True
    else:
        print("No matching band B image.")
        matching_B_image_name = None

    if matched_B == True:  # Then move onto the next step
        # Then correct the data (vignette, dark, registration)
        band_A_image = cv2.imread(day_path + "/" + band_A_name, -1)
        #show(band_A_image)
        #print(band_A_name)
        band_B_image = cv2.imread(day_path + "/" + matching_B_image_name, -1)
        #print(matching_B_image_name)
        #show(band_B_image)
        # Find if there are matching dark images:
        shutter_speed_A = int(band_A_name.split("_")[3][:-2])
        shutter_speed_B = int(matching_B_image_name.split("_")[3][:-2])
        shutter_speed_A = round(shutter_speed_A, sigfigs=sig_figs_for_dark_ss)
        shutter_speed_B = round(shutter_speed_B, sigfigs=sig_figs_for_dark_ss)

        if shutter_speed_A in dark_ss_list_A:
            matching_dark_image_index_A = dark_ss_list_A.index(shutter_speed_A)
            matching_dark_image_A = cv2.imread(dark_path_A + "/" + dark_names_A[matching_dark_image_index_A], -1)
            #print(dark_names_A[matching_dark_image_index_A])
            matching_dark_A = True
        else:
            print("No matching dark image for band A")

        if shutter_speed_B in dark_ss_list_B:
            matching_dark_image_index_B = dark_ss_list_B.index(shutter_speed_B)
            matching_dark_image_B = cv2.imread(dark_path_B + "/" + dark_names_B[matching_dark_image_index_B], -1)
            #print(dark_names_B[matching_dark_image_index_B])
            matching_dark_B = True
        else:
            print("No matching dark image for band B")

        #If so, continue with corrections:
        if matching_dark_A == True and matching_dark_B == True:
            band_A_image = band_A_image - matching_dark_image_A
            band_B_image = band_B_image - matching_dark_image_B

            # vignette correct
            vin_mask_A = clear_sky_image_A / clear_sky_image_A.max()
            #show(vin_mask_A)
            band_A_image = np.divide(band_A_image, vin_mask_A).astype('uint16')
            #show(band_A_image)
            vin_mask_B = clear_sky_image_B / clear_sky_image_B.max()
            #show(vin_mask_B)
            band_B_image = np.divide(band_B_image, vin_mask_B).astype('uint16')
            #show(band_B_image)

            # register band B
            band_B_image = cv2.warpPerspective(band_B_image, reg_trans_matrix, (648, 486))
            match_found_and_corrected = True
            #diff = band_A_image.astype('float32') - band_B_image.astype('float32')
            #show(diff)
            corrected_A = band_A_image
            corrected_B = band_B_image

    return match_found_and_corrected, corrected_A, corrected_B, matching_B_image_name
def view_and_optional_save_for_segmentation(data_folder_path, volcano_dictionary, start_time, end_time, mod, to_label_save_path, all_save_path, label_file_path, temp_steps_to_save, labelling_batch_name):
    #View the data from the folder in the time range specified
    all_files = os.listdir(data_folder_path)
    band_A_names = list(filter(lambda file_name: ("fltrA" in file_name and ".png" in file_name), all_files))
    band_B_names = list(filter(lambda file_name: ("fltrB" in file_name and ".png" in file_name), all_files))

    start_time_in_s = convert_hhmmss_to_s(start_time)
    end_time_in_s = convert_hhmmss_to_s(end_time)

    all_band_A_times_in_s = []
    for band_A_name in band_A_names:
        #print(band_A_name)
        time_in_hhmmss = band_A_name.split('_')[0][-6:]
        print(time_in_hhmmss)
        time_in_s = convert_hhmmss_to_s(time_in_hhmmss)
        all_band_A_times_in_s.append(time_in_s)

    all_band_B_times = []
    for band_B_name in band_B_names:
        all_band_B_times.append(int(band_B_name.split('_')[0][-6:]))

    dark_path_A = volcano_dictionary['dark_path_A']
    dark_path_B = volcano_dictionary['dark_path_B']
    dark_names_A = os.listdir(dark_path_A)
    dark_names_B = os.listdir(dark_path_B)
    sig_figs_for_dark_ss = volcano_dictionary['sig_figs_for_dark_ss']
    dark_ss_list_A = []
    dark_ss_list_B = []
    for file_name in dark_names_A:
        shutter_speed = int(file_name.split("_")[3][:-2])
        dark_ss_list_A.append(round(shutter_speed, sigfigs=sig_figs_for_dark_ss))
    for file_name in dark_names_B:
        shutter_speed = int(file_name.split("_")[3][:-2])
        dark_ss_list_B.append(round(shutter_speed, sigfigs=sig_figs_for_dark_ss))

    clear_sky_image_A = cv2.imread(volcano_dictionary['clear_sky_path_A'], -1)
    clear_sky_image_B = cv2.imread(volcano_dictionary['clear_sky_path_B'], -1)

    reg_trans_matrix = cv2.getPerspectiveTransform(volcano_dictionary['registration_points_B'],
                                                   volcano_dictionary['registration_points_A'])


    starting_band_A_index = all_band_A_times_in_s.index(start_time_in_s)
    ending_band_A_index = all_band_A_times_in_s.index(end_time_in_s)

    labels_df = pd.read_excel(label_file_path)

    for band_A_index in range(starting_band_A_index, ending_band_A_index, mod):
        #Show the data
        print(band_A_names[band_A_index])
        this_image = cv2.imread(data_folder_path + "/" + band_A_names[band_A_index], -1)
        plt.imshow(this_image, cmap="gray")
        plt.show()
        #Ask user for input
        save = input("Select image for labelling?")
        if save == "y":
            #If the user selects to save, attempt to correct and save the data
            matched_B = False
            band_A_name = band_A_names[band_A_index]
            match_found_and_corrected, corrected_band_A, corrected_band_B, matching_B_name = find_match_and_correct(
                day_path=day_path, band_A_name=band_A_name, band_B_names=band_B_names,
                band_B_times=all_band_B_times, sig_figs_for_dark_ss=sig_figs_for_dark_ss, dark_path_A=dark_path_A,
                dark_names_A=dark_names_A, dark_ss_list_A=dark_ss_list_A, dark_path_B=dark_path_B,
                dark_names_B=dark_names_B, dark_ss_list_B=dark_ss_list_B,
                clear_sky_image_A=clear_sky_image_A, clear_sky_image_B=clear_sky_image_B,
                reg_trans_matrix=reg_trans_matrix)
            if match_found_and_corrected == True:
                print("Image for current timestep sucessfully matched to band B and corrected.")
            else:
                print("Match and correction for current timestep unsucessful.")
            # Then if applicable, find and correct the temporal images
            continue_finding_timestep_images = True
            time_step_image_names_A = []
            time_step_images_A = []
            time_step_images_B = []
            time_step_image_names_B = []

            if match_found_and_corrected == True:
                # Then we want to find and correct images for each requested timestep
                for requested_time_step in temp_steps_to_save:
                    if continue_finding_timestep_images == True:
                        print("Requested time step:" + str(requested_time_step))
                        # Find the image index at the time closest to the requested time step
                        time_in_hhmmss = band_A_name.split('_')[0][-6:]
                        time_in_s = int(time_in_hhmmss[-2:])
                        time_in_s = time_in_s + int(time_in_hhmmss[-4:-2]) * 60
                        this_image_time_in_s = time_in_s + int(time_in_hhmmss[:2]) * 3600
                        time_differences = np.array(all_band_A_times_in_s) - this_image_time_in_s
                        difference_from_requested_time_step = np.abs(time_differences - np.ones(shape=time_differences.shape) * requested_time_step)
                        difference_from_requested_time_step = difference_from_requested_time_step.tolist()  # TODO plot the difference and see if this is going wrong
                        minimum_difference_index = difference_from_requested_time_step.index(min(difference_from_requested_time_step))
                        if difference_from_requested_time_step[minimum_difference_index] <= 10 and time_differences[
                            minimum_difference_index] != 0:
                            # A suitable timestep image has been found!
                            print("An image at time step:" + str(time_differences[minimum_difference_index]) + " has been found!")
                            time_step_image_name_A = band_A_names[minimum_difference_index]
                            time_step_image_names_A.append(time_step_image_name_A)

                            time_step_matched_and_corrected, time_step_corrected_band_A, time_step_corrected_band_B, time_step_matching_B_name = find_match_and_correct(
                                    day_path=day_path, band_A_name=time_step_image_name_A, band_B_names=band_B_names,
                                    band_B_times=all_band_B_times, sig_figs_for_dark_ss=sig_figs_for_dark_ss,
                                    dark_path_A=dark_path_A,
                                    dark_names_A=dark_names_A, dark_ss_list_A=dark_ss_list_A, dark_path_B=dark_path_B,
                                    dark_names_B=dark_names_B, dark_ss_list_B=dark_ss_list_B,
                                    clear_sky_image_A=clear_sky_image_A, clear_sky_image_B=clear_sky_image_B,
                                    reg_trans_matrix=reg_trans_matrix)

                            if time_step_matched_and_corrected == True:
                                print("The image has been sucessfully matched and corrected!")
                                time_step_images_A.append(time_step_corrected_band_A)
                                time_step_images_B.append(time_step_corrected_band_B)
                                time_step_image_names_B.append(time_step_matching_B_name)
                            else:
                                print("The timestep image was not sucessfully matched and corrected.")
                                continue_finding_timestep_images = False
                        else:
                            print("No image near this timestep was found in the given folder.")
                            continue_finding_timestep_images = False  # Stop finding matching timestep images, as there is at least one step with no match

                if continue_finding_timestep_images == True:  # If this has not been changed (bc at least one step had no suitable timestep image)
                # Then save the corrected image, plus the timestep images to the requested folder:
                # Save the corrected images for current time
                    cv2.imwrite(to_label_save_path + "/" + volcano_dictionary['volcano_name'] + "_" + band_A_name, corrected_band_A)
                    #cv2.imwrite(to_label_save_path + "/" + volcano_dictionary['volcano_name'] + "_" + matching_B_name, corrected_band_B)
                    cv2.imwrite(all_save_path + "/" + volcano_dictionary['volcano_name'] + "_" + band_A_name, corrected_band_A)
                    cv2.imwrite(all_save_path + "/" + volcano_dictionary['volcano_name'] + "_" + matching_B_name,
                            corrected_band_B)

                # Save the corrected timestep pairs:
                    for timestep_index in range(0, len(temp_steps_to_save)):
                        cv2.imwrite(all_save_path + "/" + volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[timestep_index], time_step_images_A[timestep_index])
                        cv2.imwrite(all_save_path + "/" + volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[timestep_index], time_step_images_B[timestep_index])
                    print("The image and all the relevant timestep data has been saved.")

                    new_row = {'image_name': volcano_dictionary['volcano_name'] + "_" + band_A_name,
                               'image_name_B': volcano_dictionary['volcano_name'] + "_" + matching_B_name,
                               'prev_min_name': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[0],
                               'prev_min_name_B': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[0],
                               'prev_thirtysec_name': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[1],
                               'prev_thirtysec_name_B': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[1],
                               'prev_tensec_name': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[2],
                               'prev_tensec_name_B': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[2],
                               'next_tensec_name': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[3],
                               'next_tensec_name_B': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[3],
                               'next_thirtysec_name': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[4],
                               'next_thirtysec_name_B': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[4],
                               'next_min_name': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_A[5],
                               'next_min_name_B': volcano_dictionary['volcano_name'] + "_" + time_step_image_names_B[5],
                               'volcano_dictionary_name': volcano_dictionary['volcano_dictionary_name'],
                               'labelling_batch_name': labelling_batch_name}
                    labels_df = pd.concat([labels_df, pd.DataFrame([new_row])], ignore_index=True)
                    labels_df.to_excel(label_file_path)

day_path = "X:/pering_group/Shared/Kilauea/2023-07-09"
#day_path = "X:/volcano_cameras/Shared/Cotopaxi/2023/2023-08-19"
volcano_dictionary = VolcanoDictionaries.Kilauea_View2_dictionary
start_time = "022515"
end_time = "024155"
mod = 30

labelling_batch_name = "Batch84"
to_label_save_path = "DataToLabel/" + labelling_batch_name
all_save_path = "TemporalData/" + labelling_batch_name
labels_file_path = "AllLabelledImagesList.xlsx"
temp_steps_to_save = [-60, -30, -10, 10, 30, 60]

#view_and_optional_save_for_segmentation(data_folder_path=day_path, volcano_dictionary=volcano_dictionary, start_time=start_time, end_time=end_time, mod=mod, to_label_save_path=to_label_save_path, all_save_path = all_save_path, label_file_path=labels_file_path, temp_steps_to_save=temp_steps_to_save, labelling_batch_name = labelling_batch_name)

def create_batch_folders(batch_name):
    os.makedirs("DataToLabel/" + batch_name)
    os.makedirs("ProcessedLabels/" + batch_name)
    os.makedirs("TemporalData/" + batch_name)

#create_batch_folders("Batch84")

def convert_folder_to_UINT8(folder_path):
    '''Adds converted versions of all pngs to the same folder, for labelling.'''
    folder_name = folder_path.split("/")[-1]
    os.makedirs("DataToLabelUINT8/"+ folder_name)
    for image_name in os.listdir(folder_path):
        image = cv2.imread(folder_path + "/" + image_name, -1)
        converted_img = (image/4).astype("uint8")
        cv2.imwrite("DataToLabelUINT8/"+ folder_name + "/" + image_name, converted_img)

#convert_folder_to_UINT8(folder_path="DataToLabel/Batch84")

def brighten_selected_day(date_in_text, folder_path):
    for image_name in os.listdir(folder_path):
        date = image_name.split("_")[1][0:10]
        print(date)
        if date == date_in_text:
            this_img = cv2.imread(folder_path + "/" + image_name, -1)
            this_img += 50
            cv2.imwrite(folder_path + "/" + image_name, this_img)

#brighten_selected_day("2023-04-23", "DataToLabelUINT8/Batch80")

def brighten_selected_image(image_name, folder_path):
    img = cv2.imread(folder_path + "/" + image_name, -1)
    img += 50
    cv2.imwrite(folder_path + "/" + image_name, img)

#brighten_selected_image("Reventador_2023-07-31T133450_fltrA_1ag_999904ss_Plume.png","DataToLabelUINT8/Batch80")

def read_match_save_pixel_labels(labels_folder_path, data_folder_path, save_folder_path, starting_task_no):
    '''Reads in numpy pixel-by-pixel labels, thresholds them to convert to binary,
    matches to the original file name, and saves.'''
    raw_labels_list = os.listdir(labels_folder_path)
    #number of samples x no of categories x image shape
    matched_labels_list = np.zeros((10,2,486,648))

    #go through each label
    for label_file_name in raw_labels_list:
        #read it
        raw_label = np.load(labels_folder_path + "/" + label_file_name)

        #convert to binary
        binary_label = np.where(raw_label>0, 1, 0)

        #check category
        category = label_file_name.split("-")[-2]

        #check sample number:
        task_no = int(label_file_name.split("-")[1])
        sample_no = task_no - starting_task_no

        #save in the relevant array space
        if category == "Plume":
            matched_labels_list[sample_no, 0, :,:] = binary_label
        elif category == "ExpPlume":
            matched_labels_list[sample_no, 1, :, :] = binary_label
        else:
            print("Unrecognised category!")

    #Visualise the labels
    index = 0
    for image_name in os.listdir(data_folder_path):
        image = cv2.imread(data_folder_path + "/" + image_name, -1)
        plume_mask = matched_labels_list[index, 0, :, :]
        plume_masked_img = np.where(plume_mask > 0, 0, image)
        exp_mask = matched_labels_list[index, 1, :, :]
        exp_masked_img = np.where(exp_mask > 0, 0, image)

        fig, axs = plt.subplots(1, 3)
        original_plot = axs[0].imshow(image, cmap='gray')
        plume_labels_plot = axs[1].imshow(plume_masked_img, cmap='gray')
        exp_labels_plot = axs[2].imshow(exp_masked_img, cmap='gray')
        axs[0].set_title("Original")
        axs[1].set_title("Plume")
        axs[2].set_title("Explosive")
        #fig.colorbar(original_plot, ax=axs[0], shrink=0.5)
        #fig.colorbar(plume_labels_plot, ax=axs[1], shrink=0.5)
        #fig.colorbar(exp_labels_plot, ax=axs[2], shrink=0.5)
        plt.show()

        index += 1

    #For each image_file name
    index = 0
    for image_name in os.listdir(data_folder_path):
        # Read the corresponding chunk of the numpy array
        labels_chunk = matched_labels_list[index, :, :, :]
        # Save it with the image file name
        np.save(save_folder_path + "/" + "PlumeAndExpPixels_" + image_name[:-4] + ".npy", labels_chunk)
        index += 1

'''
read_match_save_pixel_labels(labels_folder_path = "RawLabels/Batch11",
                             data_folder_path = "DataToLabel/Batch11",
                             save_folder_path = "ProcessedLabels/Batch11",
                             starting_task_no = 131)
'''

def labelme_json_to_dataset(json_path):
    os.system("labelme_export_json "+json_path+" -o "+json_path.replace(".","_"))

#json_path = "JSONSample/Reventador_2024-08-03T133350_fltrA_1ag_2999896ss_Plume.json"

#labelme_json_to_dataset(json_path)

#This works! Just need to find how the file structure is
#with a larger batch and then loop it.

def show(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()

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

def convert_json_labels(folder_name):
    #for each JSON label file
    for json_file in os.listdir(folder_name):
        #Convert to png
        labelme_json_to_dataset(folder_name + "/" + json_file)
        #Read the label image as numpy
        label_subfolder_name = json_file.split(".")[0] + "_json"
        label_png = cv2.imread(folder_name + "/" + label_subfolder_name + "/label.png")
        #plt.imshow(label_png)
        #plt.colorbar()
        #plt.show()
        #print(label_png.shape)
        #Match the label names to pixel values
        #It seems like each channel is being used for one category
        #B G R

        print(label_subfolder_name)
        #show(label_png[:,:,0])
        #show(label_png[:,:,1])
        #show(label_png[:,:,2])

        pixel_masks = np.zeros((2,486,648))


        #Read each label name in the text file:
        label_file_path = folder_name + "/" + label_subfolder_name + "/label_names.txt"
        with open(label_file_path, 'r') as file:
            line_index = 1
            for line in file:
                category = line.strip()
                if category == "_background_":
                    print("Ignoring background")
                    pass
                elif category == "Plume":
                    print("Saving plume pixels")
                    #Take the channel that matches the line index
                    #and save it in the plume mask
                    channel_index = map_line_index_to_channel(line_index)
                    pixel_masks[0,:,:] = label_png[:,:,channel_index]
                    #plt.imshow(pixel_masks[0,:,:])
                    #plt.colorbar()
                    #plt.show()
                elif category == "ExpPlume":
                    print("Saving exp plume pixels")
                    channel_index = map_line_index_to_channel(line_index)
                    pixel_masks[1,:,:] = label_png[:,:,channel_index]
                    #plt.imshow(pixel_masks[1, :, :])
                    #plt.colorbar()
                    #plt.show()
                line_index += 1

        #Convert the masks to binary
        pixel_masks = np.where(pixel_masks > 0, 1, 0)
        labelled_image = cv2.imread(folder_name + "/" + label_subfolder_name + "/img.png", -1)
        #print(labelled_image.shape)
        plume_masked_img = np.where(pixel_masks[0,:,:] > 0, 0, labelled_image)
        exp_masked_img = np.where(pixel_masks[1,:,:] > 0, 0, labelled_image)

        fig, axs = plt.subplots(1, 3)
        original_plot = axs[0].imshow(labelled_image, cmap='gray')
        plume_labels_plot = axs[1].imshow(plume_masked_img, cmap='gray')
        exp_labels_plot = axs[2].imshow(exp_masked_img, cmap='gray')
        axs[0].set_title("Original")
        axs[1].set_title("Plume")
        axs[2].set_title("Explosive")
        # fig.colorbar(original_plot, ax=axs[0], shrink=0.5)
        # fig.colorbar(plume_labels_plot, ax=axs[1], shrink=0.5)
        # fig.colorbar(exp_labels_plot, ax=axs[2], shrink=0.5)
        plt.show()


        #Save these masks in a 'ProcessedLabels/BatchName' folder
        batch_name = folder_name.split("/")[-1]
        save_path = "ProcessedLabels/" + batch_name + "/"
        np.save(save_path + "PlumeAndExpPixels_" + json_file.split(".")[0] + ".npy", pixel_masks)

#convert_json_labels("RawLabels/Batch84")

def view_saved_labels(data_folder, labels_folder):
    # Visualise the labels
    index = 0
    for image_name in os.listdir(data_folder):
        print(image_name)
        image = cv2.imread(data_folder + "/" + image_name, -1)

        matching_labels = np.load(labels_folder + "/" + "PlumeAndExpPixels_" + image_name[:-4] + ".npy")

        plume_mask = matching_labels[0, :, :]
        plume_masked_img = np.where(plume_mask > 0, 0, image)
        exp_mask = matching_labels[1, :, :]
        exp_masked_img = np.where(exp_mask > 0, 0, image)

        fig, axs = plt.subplots(1, 3)
        original_plot = axs[0].imshow(image, cmap='gray')
        plume_labels_plot = axs[1].imshow(plume_masked_img, cmap='gray')
        exp_labels_plot = axs[2].imshow(exp_masked_img, cmap='gray')
        axs[0].set_title("Original")
        axs[1].set_title("Plume")
        axs[2].set_title("Explosive")
        # fig.colorbar(original_plot, ax=axs[0], shrink=0.5)
        # fig.colorbar(plume_labels_plot, ax=axs[1], shrink=0.5)
        # fig.colorbar(exp_labels_plot, ax=axs[2], shrink=0.5)
        plt.show()

        index += 1

batch_to_view = "Batch69"
#view_saved_labels(data_folder="DataToLabel/" + batch_to_view, labels_folder="ProcessedLabels/" + batch_to_view)

def view_all_saved_labels(all_labels, mod):
    for index in range(0, all_labels.shape[0], mod):
        batch = all_labels["labelling_batch_name"][index]
        print(batch)

        image_name = all_labels["image_name"][index]
        print(image_name)
        image_name_B = all_labels["image_name_B"][index]
        image = cv2.imread("TemporalData/" + batch + "/" + image_name_B, -1)

        labels_folder = "ProcessedLabels/" + batch

        matching_labels = np.load(labels_folder + "/" + "PlumeAndExpPixels_" + image_name[:-4] + ".npy")

        plume_mask = matching_labels[0, :, :]
        plume_masked_img = np.where(plume_mask > 0, 0, image)
        exp_mask = matching_labels[1, :, :]
        exp_masked_img = np.where(exp_mask > 0, 0, image)

        fig, axs = plt.subplots(1, 3)
        original_plot = axs[0].imshow(image, cmap='gray')
        plume_labels_plot = axs[1].imshow(plume_masked_img, cmap='gray')
        exp_labels_plot = axs[2].imshow(exp_masked_img, cmap='gray')
        axs[0].set_title("Original")
        axs[1].set_title("Plume")
        axs[2].set_title("Explosive")
        plt.show()


#all_saved_labels = pd.read_excel(labels_file_path)
#view_all_saved_labels(all_saved_labels, mod=10)


all_dictionary_names = []
for dictionary in VolcanoDictionaries.all_dictionaries:
    all_dictionary_names.append(dictionary['volcano_dictionary_name'])

def find_dictionary_index(dictionary_name):
    index = all_dictionary_names.index(dictionary_name)
    return index

def decide_on_quality(on_lens, fg_cloud, image_name):
    #print(on_lens)
    #print(fg_cloud)
    if on_lens == "Yes":
        quality = "Low"
    elif fg_cloud == "Yes":
        quality = "Low"
    elif (on_lens == "No" and fg_cloud == "No"):
        quality = "Good"
    else:
        print("Error in quality labels.")
        print(on_lens)
        print(fg_cloud)
        print(image_name)
    return quality


def function_to_check_label_balance(labels_path):
    '''Count the number of explosive vs plume pixels.
    Count the number of images from each location which also have cloud.
    Count the number of images from each location which also have rain or ash.
    Count number of unique days. '''

    labels = pd.read_excel(labels_path)
    volcano_names_col = []
    image_date_col = []
    for row_index in range(0, labels.shape[0]):
        #Map the dictionary name to the volcano name
        dict_index = find_dictionary_index(labels["volcano_dictionary_name"][row_index])
        relevant_dictionary = VolcanoDictionaries.all_dictionaries[dict_index]
        volcano_names_col.append(relevant_dictionary["volcano_name"])

        #Isolate the date of the image
        date = labels["image_name"][row_index].split("_")[1][:10]
        image_date_col.append(date)

    labels["volcano_name"] = volcano_names_col
    labels["image_date"] = image_date_col
    labels["overall_quality"] = labels.apply(lambda x: decide_on_quality(x.on_lens, x.fg_cloud, x.image_name), axis=1)

    volcano_locations = ["Reventador", "Cotopaxi", "Lascar", "Lastarria", "Kilauea", "Merapi"]

    total_labels = 0
    total_good_qual = 0

    #For each location:
    for volcano_name in volcano_locations:
        location_labels = labels[labels["volcano_name"] == volcano_name]
        print("_" * 100)
        print("Location: " + volcano_name)
        print("Total labels: " + str(location_labels.shape[0]))
        total_labels += location_labels.shape[0]
        #Count how many good quality
        good_quality = location_labels[location_labels["overall_quality"] == "Good"]
        n_good_qual = good_quality.shape[0]
        print("-----")
        print("Good quality: " + str(n_good_qual))
        total_good_qual += n_good_qual
        good_w_bg_cl = good_quality[good_quality["any_cloud"] == "Yes"]
        print("Of these, " + str(good_w_bg_cl.shape[0]) + " have background cloud.")
        #From how many individual days
        print("From " + str(len(set(good_quality["image_date"]))) + " unique days.")
        print("-----")
        #Count how many low quality
        low_quality = location_labels[location_labels["overall_quality"] == "Low"]
        n_low_qual = low_quality.shape[0]
        print("Low quality: " + str(n_low_qual))
        #How many of these have cloud
        fg_cloud = low_quality[low_quality["fg_cloud"] == "Yes"]
        n_cloud = fg_cloud.shape[0]
        print("Count with fg cloud: " + str(n_cloud))
        on_lens = low_quality[low_quality["on_lens"] == "Yes"]
        n_lens = on_lens.shape[0]
        print("Count with stuff on lens: " + str(n_lens))
        #From how many unique days?
        print("From " + str(len(set(low_quality["image_date"]))) + " unique days.")

    labels.to_excel(labels_path)

    print("_" * 100)
    print("Total labels: " + str(total_labels))
    print("Total good quality data: " + str(total_good_qual))

function_to_check_label_balance("AllLabelledImagesList.xlsx")