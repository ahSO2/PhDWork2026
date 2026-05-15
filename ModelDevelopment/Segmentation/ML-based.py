import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import VolcDictionaryWithCorrectClears

samples_df = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/CotopaxiLeftOut_Train.xlsx")
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
extra_channels_path = "C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/InputChannels/woCotopaxi/"
sensor_mark_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/"
target_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/"

def show(image):
    plt.imshow(image, cmap="gray")
    plt.show()

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + "/" + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image
def read_data(df):
    #[Observation index, channel index, 486, 648]
    X = []
    Y = []
    for sample_index in range(0, df.shape[0]):
        image_name = df["image_name"][sample_index]
        dictionary_name = df["volcano_dictionary_name"][sample_index]
        dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
        batch = df["labelling_batch_name"][sample_index]

        image_A = cv2.imread(data_path + image_name, -1)
        smmn_A = dictionary["sensor_marks_mask_A"]
        image_A = mask_sensor_marks(image_A, smmn_A)
        diff = np.load(extra_channels_path + "D_" + image_name[:-3] + ".npy")
        rel_AA = np.load(extra_channels_path + "A_" + image_name[:-3] + ".npy")

        observation = #TODO stack the channels

        #TODO read the target mask:
        mask_path = target_masks_path + batch + "/PlumeAndExpPixels_" + image_name.split(".")[0] + ".npy"
        two_channel_mask = np.load(mask_path)
        all_plume_mask = two_channel_mask[0, :, :] + two_channel_mask[1, :, :]
        all_plume_mask = np.where(all_plume_mask > 0, 1, 0)


read_data(samples_df)

#TODO Read and normalise data
#Normalise using the mean and sd of the train set samples
#use torch.transforms.ToTensor()