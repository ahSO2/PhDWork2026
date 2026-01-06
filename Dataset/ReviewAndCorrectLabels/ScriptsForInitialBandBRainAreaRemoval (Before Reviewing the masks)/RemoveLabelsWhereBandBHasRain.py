import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

rainy_data = pd.read_excel("AllLabelledImagesWithRain.xlsx")
folder_path = "AllData_CorrectedWithVolcDict2/"

#Write the data to label to a folder
#for image_name in rainy_data["image_name_B"]:
#    image = cv2.imread(folder_path + "/" + image_name, -1)
#    converted_img = (image / 4).astype("uint8")
#    cv2.imwrite("RainyBandBDataUINT8/" + image_name, converted_img)


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

#extract_json_folders("RainyBandBLabels")

def convert_json_labels(folder_name):
    #for each JSON label folder
    #NOTE I already ran the conversion of all the json files to png
    for json_folder in os.listdir(folder_name):
        print(json_folder)
        print(folder_name + "/" + json_folder)
        if os.path.isdir(folder_name + "/" + json_folder):
        #Read the label image as numpy
            label_subfolder_name = json_folder.split(".")[0]
            label_png = cv2.imread(folder_name + "/" + label_subfolder_name + "/label.png")

            print(label_subfolder_name)


            #Read each label name in the text file:
            label_file_path = folder_name + "/" + label_subfolder_name + "/label_names.txt"
            with open(label_file_path, 'r') as file:
                line_index = 1
                for line in file:
                    category = line.strip()
                    if category == "_background_":
                        print("Ignoring background")
                        pass
                    elif category == "OnLensB":
                        print("Saving plume pixels")
                        #Take the channel that matches the line index
                        #and save it in the plume mask
                        channel_index = map_line_index_to_channel(line_index)
                        on_lens_pixels = label_png[:,:,channel_index]
                        on_lens_pixels = np.where(on_lens_pixels > 0, 1, 0)
                        #show(on_lens_pixels)
                        save_name = label_subfolder_name[:-5] + ".npy"
                        print(save_name)
                        np.save(folder_name + "/" + label_subfolder_name + "/" + save_name, on_lens_pixels)
                    line_index += 1


#convert_json_labels("RainyBandBLabels")


#Then go through the rainy days
for index in range(0, rainy_data.shape[0]):
    image_name_B = rainy_data["image_name_B"][index]
#If the corresponding correction .npy file exists
    correction_save_name = "RainyBandBLabels/" + image_name_B.split(".")[0] + "_json/" + image_name_B.split(".")[0] + ".npy"
    if os.path.exists(correction_save_name):
    #Find the original label file
        correction_mask = np.load(correction_save_name)
        label_batch = rainy_data["labelling_batch_name"][index]
        image_name_A = rainy_data["image_name"][index]
        og_label_path = "ProcessedLabels/" + label_batch + "/PlumeAndExpPixels_" + image_name_A.split(".")[0] + ".npy"
#Set relevant pixels to zero
        #zero both the plume and exp labels channels [0,:,:] and [1,:,:]
        og_labels = np.load(og_label_path)
        updated_labels = np.copy(og_labels)
        updated_labels[0,:,:] = np.where(correction_mask > 0, 0, og_labels[0,:,:])
        updated_labels[1, :, :] = np.where(correction_mask > 0, 0, og_labels[1, :, :])
        #TODO Display the result to check it
        channel_to_display = 0
        fig, axs = plt.subplots(ncols=2)
        image_B = cv2.imread(folder_path + image_name_B, -1)
        fill_color = np.percentile(image_B, 5)
        original_on_img = np.where(og_labels[channel_to_display, :, :] > 0, fill_color, image_B)
        axs[0].imshow(original_on_img, cmap='gray')
        axs[1].imshow(updated_labels[channel_to_display,:,:], cmap='gray')
        axs[0].set_title("Original")
        axs[1].set_title("With Band B OnLens Areas Rmvd")
        plt.title(image_name_B)
        #plt.show()

        #TODO save the updated labels
        np.save(og_label_path, updated_labels)


