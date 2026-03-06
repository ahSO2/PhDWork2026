import cv2
import matplotlib.pyplot as plt
import sys

import pandas as pd

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

#For a selected dictionary, view samples overlayed with

samples_sheet = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/FinalSplit/Train.xlsx")
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
views_to_consider = set(samples_sheet["volcano_dictionary_name"].tolist())
print(views_to_consider)

dictionary_name = "Reventador2024"
center = (348, 284)
radius =  100
#TODO I can mask out any overlap with the flank

location_samples = samples_sheet[samples_sheet["volcano_dictionary_name"] == dictionary_name]
for image_name in location_samples["image_name"]:
    image = cv2.imread(data_path + image_name, -1)
    image = cv2.circle(image, center=center, radius=radius, color=(200), thickness=2)
    plt.imshow(image, cmap="gray")
    plt.show()


