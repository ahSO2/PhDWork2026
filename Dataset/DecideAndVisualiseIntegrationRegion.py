import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

import pandas as pd

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

#For a selected dictionary, view samples overlayed with

samples_sheet = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/FinalSplit/Train.xlsx")
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
views_to_consider = set(samples_sheet["volcano_dictionary_name"].tolist())
print(views_to_consider)

dictionary_name = "MerapiTenthJune"
center = (600, 100)
radius = 170
#TODO I can mask out any overlap with the flank

volcano_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
location_samples = samples_sheet[samples_sheet["volcano_dictionary_name"] == dictionary_name]
for image_name in location_samples["image_name"]:
    image = cv2.imread(data_path + image_name, -1)
    image = cv2.circle(image, center=center, radius=radius, color=(200), thickness=2)
    flank_mask = cv2.imread(volcano_dictionary["flank_mask_path"], -1)
    image = np.where(flank_mask==1, 0, image)
    #plt.imshow(flank_mask)
    #plt.colorbar()
    #plt.show()
    print(image_name)
    plt.imshow(image, cmap="gray")
    plt.show()


