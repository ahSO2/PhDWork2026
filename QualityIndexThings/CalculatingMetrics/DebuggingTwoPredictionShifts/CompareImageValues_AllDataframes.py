#Loop through every image in the final full splits (ORDA sets)
#Read the image from the ORDA dataset and from the original dataset folder
#Check for any instances where they are different
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataframes_folder_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun12th2026/PrecipitationFullSplit/"
save_path = "CheckingImgValues_AllPrecipDFs.xlsx"

timesteps_to_check = ["minus_one_min_name", "image_name", "plus_one_min_name", "minus_ten_s_name", "plus_ten_s_name", "minus_one_min_name_B", "image_name_B", "plus_one_min_name_B", "minus_ten_s_name_B", "plus_ten_s_name_B"]
original_main_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2/"
original_temporal_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2Temporal/"
original_additional_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/AdditionalDataPool_SelectedForFGCloudExpmt/"
ORDA_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_OldVersionWithPotentialIssue/Precipitation Full Split - Seen Locations/"
ORDA_Lastarria_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_OldVersionWithPotentialIssue/Lastarria - Balanced for Precipitation Unseen Test/"
ORDA_additional_data_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_OldVersionWithPotentialIssue/FullFold Additional Training Samples For Precipitation/"

'''
outputs_log = pd.DataFrame(columns=["image_name", "timestep_w_issue", "timestep_image_name", "issue"])
for df_name in os.listdir(dataframes_folder_path):
    print("Checking " + df_name)
    df = pd.read_csv(dataframes_folder_path + df_name)
    #Loop over each sample:
    for image_index in range(0, df.shape[0]):
        if image_index%100 == 0:
            print(image_index)
        for timestep_name in timesteps_to_check:
            name_to_read = df[timestep_name][image_index]

            #Read from original folders:
            if timestep_name == "image_name" or timestep_name == "image_name_B":
                og_img = cv2.imread(original_main_data_path + name_to_read, -1)
            else:
                og_img = cv2.imread(original_temporal_data_path + name_to_read, -1)
            if type(og_img) != np.ndarray:
                og_img = cv2.imread(original_additional_data_path + name_to_read, -1)


            #Read from ORDA folder:
            if "Lastarria" in name_to_read:
                orda_img = cv2.imread(ORDA_Lastarria_path + name_to_read, -1)
            else:
                orda_img = cv2.imread(ORDA_path + name_to_read, -1)
            if type(orda_img) != np.ndarray:
                orda_img = cv2.imread(ORDA_additional_data_path + name_to_read, -1)


            #Check if the images are read correctly and that they match
            if type(og_img) == np.ndarray and type(orda_img) == np.ndarray:
                if np.array_equal(og_img, orda_img) == False:
                    new_row = [df["image_name"][image_index], timestep_name, name_to_read, "mismatch"]
                    outputs_log.loc[len(outputs_log)] = new_row
                    print("Mismatch issue")
            else:
                new_row = [df["image_name"][image_index], timestep_name, name_to_read, "read-in-issue"]
                outputs_log.loc[len(outputs_log)] = new_row
                print("Read-in issue")
    outputs_log.to_excel(save_path)

'''
#View the non-matching pairs:
mismatch_df = pd.read_excel(save_path)

for index in range(0, len(mismatch_df)):
    if mismatch_df["issue"][index] == "mismatch":
        name_to_read = mismatch_df["timestep_image_name"][index]
        timestep_name = mismatch_df["timestep_w_issue"][index]

        # Read from original folders:
        if timestep_name == "image_name" or timestep_name == "image_name_B":
            og_img = cv2.imread(original_main_data_path + name_to_read, -1)
        else:
            og_img = cv2.imread(original_temporal_data_path + name_to_read, -1)
        if type(og_img) != np.ndarray:
            og_img = cv2.imread(original_additional_data_path + name_to_read, -1)

        # Read from ORDA folder:
        if "Lastarria" in name_to_read:
            orda_img = cv2.imread(ORDA_Lastarria_path + name_to_read, -1)
        else:
            orda_img = cv2.imread(ORDA_path + name_to_read, -1)
        if type(orda_img) != np.ndarray:
            orda_img = cv2.imread(ORDA_additional_data_path + name_to_read, -1)

        diff = np.abs(orda_img.astype(np.float32) - og_img.astype(np.float32))
        fig, axs = plt.subplots(ncols=3)
        left_plot = axs[0].imshow(og_img, cmap='gray')
        center_plot = axs[1].imshow(orda_img, cmap='gray')
        right_plot = axs[2].imshow(diff, cmap="gray")
        axs[0].set_title("Original Folder")
        axs[1].set_title("ORDA Folder")
        axs[2].set_title("Difference")
        fig.colorbar(left_plot, ax=axs[0], shrink=0.5)
        fig.colorbar(center_plot, ax=axs[1], shrink=0.5)
        fig.colorbar(right_plot, ax=axs[2], shrink=0.5)
        plt.show()



