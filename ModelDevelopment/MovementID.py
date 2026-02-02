#Goal: given an image pair, filter for areas in the first image which
#have moved going into the second image.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.restoration import denoise_bilateral


def show(img):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()

def convert_sequence_to_UINT8(sequence):
    converted_sequence = []
    for image in sequence:
        converted_image = (image/4).astype("uint8")
        converted_sequence.append(converted_image)
    return converted_sequence

def pixel_diff(current, next):
    #current = cv2.bilateralFilter(current,9, 20, 20)
    #next = cv2.bilateralFilter(next, 9, 20, 20)
    show(current)
    current = denoise_bilateral(current.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    next = denoise_bilateral(next.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    show(current)
    diff = np.abs(next.astype("float32") - current.astype("float32"))
    show(diff)

    counts, bins = np.histogram(diff.flatten(), 100, [-50, 50])
    plt.stairs(counts, bins)
    plt.show()





def apply_function_on_train_samples(samples_sheet, data_path, data_path_temporal, mod):
    timesteps = ["prev_tensec_name", "image_name"]
    #For each image in the specified training set
    dataset = pd.read_excel(samples_sheet)
    dataset = dataset[dataset["overall_obs"] == "No"]
    #dataset = dataset[dataset["volcano_name"]=="Reventador"]
    dataset.reset_index(inplace=True)
    for index in range(0, dataset.shape[0], mod):
        #Read the sequence
        sequence = []
        names = []
        for timestep_name in timesteps:
            if timestep_name == "image_name":
                folder_to_read = data_path
                print(dataset[timestep_name][index])
            elif timestep_name == "image_name_B":
                folder_to_read = data_path
            else:
                folder_to_read = data_path_temporal
            name_to_read = dataset[timestep_name][index]
            timestep_image = cv2.imread(folder_to_read + "/" + name_to_read, -1)
            #plt.imshow(timestep_image)
            #plt.show()
            sequence.append(timestep_image)
            names.append(name_to_read)
        #sequence = convert_sequence_to_UINT8(sequence)
        #print("Data type after converting sequence:")
        #print(sequence[0].dtype)
        #sequence = add_gauss_noise(sequence)
        for sequence_index in range(0, len(sequence) - 1):
            current_img = sequence[sequence_index]
            #show(current_img)
            next_img = sequence[sequence_index + 1]
            #show(next_img)
            pixel_diff(current_img, next_img)
            #MOG(current_img, next_img)

        #Save the results



samples_sheet = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/FinalSplit/Train.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
folder_to_save = "none"
mod = 20
apply_function_on_train_samples(samples_sheet, data_path, data_path_temporal, mod)
