#Following the issue described in "C:\Users\ggp24ash\Documents\Quality Index Write Up\Version for submission - R2\Note of additional changes"
#Loop through the given list of datasets
#Read the current timestep image from the original folder
#Read the other timestep images from the temporal folder
#Save in the new folders
import cv2
import os
import pandas as pd

main_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2/"
temporal_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2Temporal/"

main_destination_folder = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_Updated23rdJune26/MainDataset/"
temporal_destination_folder = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_Updated23rdJune26/MainDataset_AssociatedTemporal/"

directory = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudFullSplit/"
dataframe_paths = os.listdir(directory)

timesteps_to_read = ["minus_one_min_name", "image_name", "plus_one_min_name", "minus_ten_s_name", "plus_ten_s_name", "minus_one_min_name_B", "image_name_B", "plus_one_min_name_B", "minus_ten_s_name_B", "plus_ten_s_name_B"]

for path in dataframe_paths:
    print("Reading samples from: " + path)
    df = pd.read_csv(directory + path)
    if "Expanded" in path:
        pass #We read in the OG samples only using the unexpanded train df - the additional samples are already in a folder
    else:
        for image_index in range(0, len(df)):
            if image_index%100 == 0:
                print(image_index)
            for timestep_name in timesteps_to_read:
                name_to_read = df[timestep_name][image_index]

                #Read from original folders:
                if timestep_name == "image_name" or timestep_name == "image_name_B":
                    og_img = cv2.imread(main_data_path + name_to_read, -1)
                    cv2.imwrite(main_destination_folder + name_to_read, og_img)
                else:
                    og_img = cv2.imread(temporal_data_path + name_to_read, -1)
                    cv2.imwrite(temporal_destination_folder + name_to_read, og_img)



