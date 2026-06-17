#I want to add the full unbalanced Lastarria set to ORDA,
#to allow replication of my comparison with Delle Donne indexes.
import cv2
import pandas as pd

labels_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun12th2026/ForQualityIndexComparison/Lastarria_AllSamples_Unbalanced.csv"
timesteps = ["image_name", "minus_one_min_name", "minus_ten_s_name", "plus_ten_s_name", "plus_one_min_name"]
destination_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data/Lastarria - All Unbalanced/"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2/"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2Temporal/"

labels = pd.read_csv(labels_path)
labels.reset_index(inplace=True)
#For each row in labels dataframe:
for row in range(0, labels.shape[0]):
    #For each timestep
    for timestep in timesteps:
        #Read the corresponding image name
        name_A = labels[timestep][row]
        #Read the corresponding band B image name
        name_B = labels[timestep + "_B"][row]

        if timestep == "image_name":
            path_to_read = data_path
            print("Saving data for image: " + name_A)
        else:
            path_to_read = data_path_temporal

        image_A = cv2.imread(path_to_read + name_A, -1)
        image_B = cv2.imread(path_to_read + name_B, -1)

        #Save both to the destination folder
        cv2.imwrite(destination_path + name_A, image_A)
        cv2.imwrite(destination_path + name_B, image_B)