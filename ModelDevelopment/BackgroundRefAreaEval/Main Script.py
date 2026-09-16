import matplotlib.pyplot as plt
import numpy as np
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
mod = 10
timesteps = ["image_name", "next_tensec_name"]

paths_dictionary = {"df_path":"C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/",
                    "data_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2",
                    "data_path_temporal":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal",
                    "segmentation_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/",
                    "sensor_mark_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/",
                    "flank_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"}

##################### Script
columns = ["image_name", "PP", "PN", "RP", "RN", "IOU", "F1_AD"]

for llo in locations: #Leaving out one location at a time

    metrics_df = pd.DataFrame(columns=columns)

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

        if sample_index < 1000:
            fig, axs = plt.subplots(nrows=2, ncols=3)
            axs[0,0].imshow(sequence[0], cmap="gray")
            axs[1,0].imshow(sequence_B[0], cmap="gray")
            axs[0,1].imshow(sequence[1], cmap="gray")
            axs[1,1].imshow(sequence_B[1], cmap="gray")
            axs[0,2].imshow(plume_mask, cmap="gray")
            axs[1,2].imshow(flank_mask, cmap="gray")
            plt.show()

        #ref_areas = thresholding(sequence[0], sequence_B[0], volc_dictionary, plot=True)
        ref_areas = smekens_repeated_fitting(sequence[0], flank_mask)
        #TODO Each method should include a masking out of the flank area
        #plt.imshow(ref_areas, cmap="gray")
        #plt.show()

        ground_truth_bg = np.where(plume_mask == 0, flank_mask, 0)
        plt.imshow(ground_truth_bg)
        plt.title("Ground truth background pixels")
        plt.colorbar()
        plt.show()
        TP, TN, FP, FN = calculate_conf_counts(ref_areas, ground_truth_bg)
        PP, PN = per_class_precision(TP, TN, FP, FN)
        print("Precisions")
        print(PP)
        print(PN)
        RP, RN = per_class_recall(TP, TN, FP, FN)
        print("Recalls")
        print(RP)
        print(RN)
        IOU = intersection_over_union(TP, TN, FP, FN)
        print("IOU")
        print(IOU)
        F1_modified = F1_score(RN, RP) #TODO I have modified the inputs so we have a harmonic mean of the two quantities that are most informative in this application
        print("F1")
        print(F1_modified)

        new_row = {"image_name": names[0],
                   "PP":PP,
                   "PN":PN,
                   "RP":RP,
                   "RN":RN,
                   "IOU":IOU,
                   "F1_AD":F1_modified}
        metrics_df.loc[len(metrics_df)] = new_row

    #TODO Calculate means and add as the last row of the dataframe
    #TODO Save dataframe for each location









