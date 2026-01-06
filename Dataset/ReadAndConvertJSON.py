import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        save_path = "ProcessedLabels/JSONSample/"
        np.save(save_path + "PlumeAndExpPixels_" + json_file.split(".")[0] + ".npy", pixel_masks)

convert_json_labels("JSONSample")
