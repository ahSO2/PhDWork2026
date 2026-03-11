import os
import cv2
import numpy as np
#for image_name in os.listdir("10bitImgs"):
#    image = cv2.imread("10bitImgs/" + image_name, -1)
#    image = (image/4).astype("uint8")
#    image = image + 30
#    cv2.imwrite("UINT8/" + image_name, image)


#Next, convert the flank masks to numpy:

def labelme_json_to_dataset(json_path):
    os.system("labelme_export_json "+json_path+" -o "+json_path.replace(".","_"))

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

        pixel_masks = np.zeros((486,648))


        #Read each label name in the text file:
        label_file_path = folder_name + "/" + label_subfolder_name + "/label_names.txt"
        with open(label_file_path, 'r') as file:
            line_index = 1
            for line in file:
                category = line.strip()
                if category == "_background_":
                    print("Ignoring background")
                    pass
                elif category == "Flank":
                    print("Saving flank pixels")
                    #Take the channel that matches the line index
                    #and save it in the plume mask
                    channel_index = map_line_index_to_channel(line_index)
                    pixel_masks[:,:] = label_png[:,:,channel_index]
                    #plt.imshow(pixel_masks[0,:,:])
                    #plt.colorbar()
                    #plt.show()
                line_index += 1

        #Convert the masks to binary
        pixel_masks = np.where(pixel_masks > 0, 0, 1)




        #Save these masks in a 'ProcessedLabels/BatchName' folder
        save_path = "FlankMasks/"
        cv2.imwrite(save_path + "FlankPixels_" + json_file.split(".")[0] + ".png", pixel_masks)


convert_json_labels("RawMasks")

