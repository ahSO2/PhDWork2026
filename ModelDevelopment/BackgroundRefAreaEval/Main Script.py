import matplotlib.pyplot as plt
import pandas as pd
import sys

from EvaluationFunctions import *
from ReadingFunctions import *
from RefAreaSelectionFunctions import *

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

#locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
locations = ["Cotopaxi"]

filter_for_quality = "Good"
mod = 1
timesteps = ["image_name", "next_tensec_name"]

paths_dictionary = {"df_path":"C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/",
                    "data_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2",
                    "data_path_temporal":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal",
                    "segmentation_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/",
                    "sensor_mark_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/",
                    "flank_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"}

##################### Script

for llo in locations: #Leaving out one location at a time
    print("Running tests on " + llo + "-left-out CV Fold.")
    train_df = pd.read_excel(paths_dictionary["df_path"] + llo + "LeftOut_Train.xlsx")

    if filter_for_quality == "Good":
        train_df = train_df[train_df["overall_obs"] == "No"]
        print("Selected " + str(train_df.shape[0]) + " good quality samples.")
        train_df.reset_index(inplace=True)
    elif filter_for_quality == "Low":
        train_df = train_df[train_df["overall_obs"] == "Yes"]
        print("Selected " + str(train_df.shape[0]) + " low quality samples.")
        train_df.reset_index(inplace=True)

    print("Reading Samples:")
    for sample_index in range(0, train_df.shape[0], mod):
        print(sample_index)
        # Read the timestep sequence
        volc_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(train_df["volcano_dictionary_name"][sample_index])
        sequence, names, plume_mask, flank_mask = read_sample(sample_index, train_df, timesteps, "A", volc_dictionary, paths_dictionary)
        sequence_B, names_B, NA, flank_mask = read_sample(sample_index, train_df, timesteps, "B", volc_dictionary, paths_dictionary)

        if sample_index == 1000:
            fig, axs = plt.subplots(nrows=2, ncols=3)
            axs[0,0].imshow(sequence[0], cmap="gray")
            axs[1,0].imshow(sequence_B[0], cmap="gray")
            axs[0,1].imshow(sequence[1], cmap="gray")
            axs[1,1].imshow(sequence_B[1], cmap="gray")
            axs[0,2].imshow(plume_mask, cmap="gray")
            axs[1,2].imshow(flank_mask, cmap="gray")
            plt.show()

        ref_areas = delledonne_min_ratio(sequence[0], sequence_B[0], volc_dictionary, plot=True)
        plt.imshow(ref_areas, cmap="gray")
        plt.show()






