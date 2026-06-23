import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#I'm getting a different result when I calculate model predictions using
#the original temporal data path vs the combined data folder uploaded to ORDA.
#I'll loop through the temporal samples in the precip train set and see which
#if any are not equal.

old_folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2Temporal/"
new_folder_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data/Precipitation Full Split - Seen Locations/" #Folder storing image data

dataframe = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/ClusteringForNewLabels/AllLocations/OnLensExperiment_UpdatedPCA/ExpandedOnLensExpmtSeenLocationsTrainSet.xlsx"

timestep_names = ["minus_one_min_name", "plus_one_min_name"]

'''
df = pd.read_excel(dataframe)

mismatch_log = pd.DataFrame(columns=["image_name", "timestep_name", "timestep_image_name", "issue"])
for index in range(0, df.shape[0]):
    print("Checking data for image: " + df["image_name"][index])
    for timestep in timestep_names:
        name_to_read = df[timestep][index]
        img1 = cv2.imread(old_folder_path + name_to_read, -1)
        img2 = cv2.imread(new_folder_path + name_to_read, -1)
        if type(img1) == np.ndarray and type(img2) == np.ndarray:
            if np.array_equal(img1, img2) == False:
                print("Mismatch in image data")
                new_row = [df["image_name"][index], timestep, df[timestep][index], "mismatch"]
                mismatch_log.loc[len(mismatch_log)] = new_row
            else:
                print("Correct match.")
        else:
            new_row = [df["image_name"][index], timestep, df[timestep][index], "read-in-issue"]

mismatch_log.to_excel("MismatchedDataLog.xlsx")
'''

#For each mismatched sample, read the data from each folder and plot
mismatch_log = pd.read_excel("MismatchedDataLog.xlsx")
for index in range(0, mismatch_log.shape[0]):
    if mismatch_log["issue"][index] == "mismatch":
        name_to_read = mismatch_log["timestep_image_name"][index]
        img1 = cv2.imread(old_folder_path + name_to_read, -1)
        img2 = cv2.imread(new_folder_path + name_to_read, -1)
        fig, axs = plt.subplots(ncols=2)
        left_plot = axs[0].imshow(img1, cmap='gray')
        right_plot = axs[1].imshow(img2, cmap='gray')
        axs[0].set_title("Original Folder")
        axs[1].set_title("ORDA Folder")
        fig.colorbar(left_plot, ax=axs[0], shrink=0.5)
        fig.colorbar(right_plot, ax=axs[1], shrink=0.5)
        plt.show()

        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        plt.imshow(diff)
        plt.colorbar()
        plt.show()


