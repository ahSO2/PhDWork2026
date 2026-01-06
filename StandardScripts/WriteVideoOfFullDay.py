#For a given folder containing corrected data from one day,
#write a video, scaled so that even dark conditions are visible
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data_paths = ["X:/volcano_cameras/Shared/Reventador/2024/2024-08-03/"]

def create_image_names_dataframe(data_path):
    '''Go through folder, list the matching pairs of
    band A and band B images.'''
    all_band_A_names = []
    all_band_B_names = []
    all_band_B_times = []
    matched_band_A_names = []
    matched_band_B_names = []
    for file_name in os.listdir(data_path):
        #print(file_name)
        if ".png" in file_name:
            if "fltrA" in file_name:
                all_band_A_names.append(file_name)
            elif "fltrB" in file_name:
                all_band_B_names.append(file_name)
                all_band_B_times.append(file_name[11:17])
                #print(all_band_B_times)
    for band_A_image in all_band_A_names:
        time = band_A_image[11:17]
        if time in all_band_B_times:
            index = all_band_B_times.index(time)
            matching_band_B_name = all_band_B_names[index]
            matched_band_A_names.append(band_A_image)
            matched_band_B_names.append(matching_band_B_name)
        else:
            print("No matching band B for this image.")

    outputs_df = pd.DataFrame()
    outputs_df["image_name"] = matched_band_A_names
    outputs_df["image_name_B"] = matched_band_B_names
    outputs_df.to_excel(dataframe_path)


def scale_brightness(image):
    image_max = np.percentile(image, 99)
    scale_factor = 255/image_max
    scaled_image = image.astype("float") * scale_factor
    scaled_image = np.where(scaled_image > 255, 255, scaled_image)
    return scaled_image.astype("uint8")
def create_video(data_path, dataframe_path, video_save_path):
    index = 0
    #output = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'XVID'), 5, (1296,486))
    output = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1296,486))

    dataframe = pd.read_excel(dataframe_path)
    #print(dataframe.shape)
    #dataframe = dataframe.tail(3295)
    #dataframe.reset_index(inplace=True)
    print("Pairs to write: " + str(dataframe.shape))
    for image_name in dataframe["image_name"]:
        print("Writing index " + str(index))
        print(image_name)
        image_name_B = dataframe["image_name_B"][index]

        image_A = cv2.imread(data_path + "/" + image_name, -1)
        image_B = cv2.imread(data_path + "/" + image_name_B, -1)
        #image_A = (image_A/4).astype("uint8")
        #image_B = (image_B/4).astype("uint8")

        # Scale each band so the 99th percentile value is 256
        # Then threshold to remove any values above that
        image_A = scale_brightness(image_A)
        image_B = scale_brightness(image_B)

        #prediction = round(float(results_df["model_predictions"][index]), 1)

        both_bands = np.concatenate([image_A, image_B], axis=1)

        #plt.imshow(both_bands)
        #plt.colorbar()
        #plt.show()

        rgb_both_bands = cv2.cvtColor(both_bands, cv2.COLOR_GRAY2BGR)
        #print(rgb_both_bands.shape)
        #max_val = int(rgb_both_bands.max())


        #max_val = 255
        #if prediction < 0.5:
        #    color = (0,max_val,0)
        #elif prediction <0.75:
        #    color = (0,max_val,max_val)
        #else:
        #    color = (0, 0, max_val)

        #both_bands_rgb = cv2.putText(rgb_both_bands, str(prediction), (100,100), cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=color, thickness=2)
        both_bands_rgb = cv2.rectangle(rgb_both_bands, (0,10), (560,40), (0,0,0), -1)
        both_bands_rgb = cv2.putText(both_bands_rgb, image_name, (20,30), cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=0.5, color=(255,255,255), thickness=1)

        #plt.imshow(both_bands_rgb)
        #plt.show()

        ##to_write.append(both_bands)
        output.write(both_bands_rgb)

        index += 1

    output.release()

for data_path in data_paths:
    print(data_path)
    dataframe_path = data_path + "/ImagePairNames.xlsx"
    print("Matching image pairs:")
    create_image_names_dataframe(data_path)
    create_video(data_path, dataframe_path, data_path + "/" + data_path.split("/")[-1] + "_BrightnessAdjusted_Uncorrected.mp4")
#os.system("shutdown /s /t 1")