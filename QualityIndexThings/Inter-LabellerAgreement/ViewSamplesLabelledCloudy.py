#Read samples from given dataframe
#Save cloudy ones to given folder
import cv2
import matplotlib.pyplot as plt
import pandas as pd

df_to_read = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudFullSplit/Cloud_Full_Train.csv"
folder_to_write_to = "C:/Users/ggp24ash/Documents/Scratch Data/CloudLabelling/"
images_to_read = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2_UINT8/"

mod = 3
image_names = pd.read_csv(df_to_read)["image_name"].tolist()
cloud_level = pd.read_csv(df_to_read)["obs_cloud_level"].tolist()
for index in range(0, len(image_names), mod):
    print(index)
    name = image_names[index]
    image = cv2.imread(images_to_read + name, -1)
    this_cloud_level = cloud_level[index]
    cv2.imwrite(folder_to_write_to + this_cloud_level + " Cloud/" + name, image)
