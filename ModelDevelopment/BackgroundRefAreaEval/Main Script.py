import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
import sys

from EvaluationFunctions import *
from ReadingFunctions import *
from RefAreaSelectionFunctions import *

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
#locations = ["Reventador"]
filter_for_quality = "Good"
set_to_consider = "Train"
mod = 7
timesteps = ["image_name", "next_tensec_name"]
save_results = False
save_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/"
rng = numpy.random.default_rng(42) #Create a random number generator with seed
paths_dictionary = {"df_path":"C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/",
                    "data_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026/",
                    "data_path_temporal":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026_Temporal/",
                    "segmentation_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/",
                    "sensor_mark_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/",
                    "flank_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"}

##################### Script
columns = ["image_name", "volcano_name", "PP", "PN", "RP", "RN", "IOU", "F1", "VR"] #Metrics to save per image
summary_columns = ["fold", "PP", "PP_L", "PP_U", "PN", "PN_L", "PN_U", "RP", "RP_L", "RP_U", "RN", "RN_L", "RN_U", "IOU", "IOU_L", "IOU_U", "F1_M", "F1_L", "F1_U", "VR", "VR_L", "VR_U"]
overall_CV_fold_results_df = pd.DataFrame(columns=summary_columns) #Dataframe to store summary results for each fold
all_samples_metrics = pd.DataFrame(columns=columns)
for llo in locations: #Leaving out one location at a time

    metrics_df = pd.DataFrame(columns=columns)

    print("Running tests on " + llo + "-left-out CV Fold.")
    samples_df = pd.read_excel(paths_dictionary["df_path"] + llo + "LeftOut_" + set_to_consider + ".xlsx")

    if filter_for_quality == "Good":
        samples_df = samples_df[samples_df["overall_obs"] == "No"]
        print("Selected " + str(samples_df.shape[0]) + " good quality samples.")
        samples_df.reset_index(inplace=True)
    elif filter_for_quality == "Low":
        samples_df = samples_df[samples_df["overall_obs"] == "Yes"]
        print("Selected " + str(samples_df.shape[0]) + " low quality samples.")
        samples_df.reset_index(inplace=True)

    print("Reading Samples:")
    for sample_index in range(0, samples_df.shape[0], mod):
        print(sample_index)
        # Read the timestep sequence
        volc_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(samples_df["volcano_dictionary_name"][sample_index])
        sequence, names, plume_mask, flank_mask = read_sample(sample_index, samples_df, timesteps, "A", volc_dictionary, paths_dictionary)
        sequence_B, names_B, NA, flank_mask = read_sample(sample_index, samples_df, timesteps, "B", volc_dictionary, paths_dictionary)

        #Blur my drawn flank masks slightly to ensure all flank pixels are excluded
        smoothed_flank_mask = cv2.blur(flank_mask * 5, (20, 20))
        flank_mask = np.where(smoothed_flank_mask < 5, 0, 1)

        #Calculate the reference areas with the chosen method
        #ref_areas, method_name = delledonne_max_bandA(sequence[0], sequence_B[0], plot=False)
        #ref_areas, method_name = delledonne_min_ratio(sequence[0], sequence_B[0], plot=False)
        #ref_areas, method_name = pyplis_rectangles_and_lines(sequence[0], plot=True, output="both")
        #ref_areas, method_name = pyplis_background_mask(sequence[0], sequence[1], plot=False)
        #ref_areas, method_name = kern_low_texture_and_ratio(sequence[0], sequence_B[0], flank_mask, plot=False)
        #ref_areas = thresholding(sequence[0], sequence_B[0], volc_dictionary, plot=True)
        #ref_areas, method_name = smekens_repeated_fitting(sequence[0], flank_mask, plot=False)
        ref_areas, method_name = custom_combined(sequence[0], sequence_B[0], sequence[1], sequence_B[1], flank_mask, plot=True)

        if sample_index < 0: #Plot the timesteps and masks
            fig, axs = plt.subplots(nrows=2, ncols=3)
            axs[0,0].imshow(sequence[0], cmap="gray")
            axs[0,0].set_title("310nm Frame 1", fontsize=10)
            axs[1,0].imshow(sequence_B[0], cmap="gray")
            axs[1, 0].set_title("330nm Frame 1", fontsize=10)
            axs[0,1].imshow(sequence[1], cmap="gray")
            axs[0, 1].set_title("310nm Frame 2", fontsize=10)
            axs[1,1].imshow(sequence_B[1], cmap="gray")
            axs[1, 1].set_title("320nm Frame 2", fontsize=10)
            axs[0,2].imshow(plume_mask, cmap="gray")
            axs[0, 2].set_title("Manual Plume Mask", fontsize=10)
            axs[1,2].imshow(flank_mask, cmap="gray")
            axs[1, 2].set_title("Flank Mask", fontsize=10)
            fig.subplots_adjust(wspace=0, hspace=0)
            for row in range(0, 2):
                for col in range(0, 3):
                    axs[row, col].set_xticklabels([])
                    axs[row, col].set_yticklabels([])
            plt.show()

        #Mask out the flank area, incase the method hasn't included this already
        ref_areas = np.where(flank_mask == 1, ref_areas, 0)
        ground_truth_bg = np.where(plume_mask == 0, flank_mask, 0)

        if sample_index < 0: #Plot the selected background regions
            fig, axs = plt.subplots(nrows=2, ncols=3)
            axs[0,0].imshow(sequence[0], cmap="gray")
            axs[1,0].imshow(sequence_B[0], cmap="gray")
            axs[0,1].imshow(sequence[1], cmap="gray")
            axs[1,1].imshow(sequence_B[1], cmap="gray")
            axs[0,2].imshow(ground_truth_bg, cmap="gray")
            axs[1,2].imshow(ref_areas, cmap="gray")
            plt.show()

        TP, TN, FP, FN = calculate_conf_counts(ref_areas, ground_truth_bg)
        PP, PN = per_class_precision(TP, TN, FP, FN)
        RP, RN = per_class_recall(TP, TN, FP, FN)
        IOU = intersection_over_union(TP, TN, FP, FN)
        F1 = F1_score(PP, RP)
        VR = variance_ratio(sequence[0], sequence_B[0], ref_areas, ground_truth_bg, flank_mask)

        new_row = {"image_name": names[0],
                   "volcano_name": samples_df["volcano_name"][sample_index],
                   "PP":PP,
                   "PN":PN,
                   "RP":RP,
                   "RN":RN,
                   "IOU":IOU,
                   "F1":F1,
                   "VR":VR}
        metrics_df.loc[len(metrics_df)] = new_row
        all_samples_metrics.loc[len(all_samples_metrics)] = new_row

    #Save the per-image metrics df
    per_image_metrics_save_name = method_name + "_wo" + llo + "_" + set_to_consider + "_mod" + str(
        mod) + "_quality-" + filter_for_quality + ".csv"
    if save_results == True:
        metrics_df.to_csv(save_path + per_image_metrics_save_name)

    #For this fold, calculate the mean of each metric over the samples, and the 95% bootstrap confidence interval
    PP_M, PP_L, PP_U, rng = mean_with_95p_bootstrap(metrics_df["PP"].to_numpy(), rng)
    PN_M, PN_L, PN_U, rng = mean_with_95p_bootstrap(metrics_df["PN"].to_numpy(), rng)
    RP_M, RP_L, RP_U, rng = mean_with_95p_bootstrap(metrics_df["RP"].to_numpy(), rng)
    RN_M, RN_L, RN_U, rng = mean_with_95p_bootstrap(metrics_df["RN"].to_numpy(), rng)
    IOU_M, IOU_L, IOU_U, rng = mean_with_95p_bootstrap(metrics_df["IOU"].to_numpy(), rng)
    F1_M, F1_L, F1_U, rng = mean_with_95p_bootstrap(metrics_df["F1"].to_numpy(), rng)
    VR_M, VR_L, VR_U, rng = mean_with_95p_bootstrap(metrics_df["VR"].to_numpy(), rng)
    summary_row = {"fold": llo,
                   "PP":PP_M,
                   "PP_L":PP_L,
                   "PP_U":PP_U,
                   "PN":PN_M,
                   "PN_L":PN_L,
                   "PN_U":PN_U,
                   "RP":RP_M,
                   "RP_L":RP_L,
                   "RP_U":RP_U,
                   "RN":RN_M,
                   "RN_L":RN_L,
                   "RN_U":RN_U,
                   "IOU":IOU_M,
                   "IOU_L":IOU_L,
                   "IOU_U":IOU_U,
                   "F1_M":F1_M,
                   "F1_L":F1_L,
                   "F1_U":F1_U,
                   "VR":VR_M,
                   "VR_L":VR_L,
                   "VR_U":VR_U}
    overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = summary_row

#Calculate the location-balanced mean metrics (with bootstrap CIs)
PP_M, PP_L, PP_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["PP"].to_numpy(), all_samples_metrics["volcano_name"], rng)
PN_M, PN_L, PN_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["PN"].to_numpy(), all_samples_metrics["volcano_name"], rng)
RP_M, RP_L, RP_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["RP"].to_numpy(), all_samples_metrics["volcano_name"], rng)
RN_M, RN_L, RN_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["RN"].to_numpy(), all_samples_metrics["volcano_name"], rng)
IOU_M, IOU_L, IOU_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["IOU"].to_numpy(), all_samples_metrics["volcano_name"], rng)
F1_M, F1_L, F1_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["F1"].to_numpy(), all_samples_metrics["volcano_name"], rng)
VR_M, VR_L , VR_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["VR"].to_numpy(), all_samples_metrics["volcano_name"], rng)
mean_over_folds_row = {"fold": "location-balanced",
                   "PP":PP_M,
                   "PP_L":PP_L,
                   "PP_U":PP_U,
                   "PN":PN_M,
                   "PN_L":PN_L,
                   "PN_U":PN_U,
                   "RP":RP_M,
                   "RP_L":RP_L,
                   "RP_U":RP_U,
                   "RN":RN_M,
                   "RN_L":RN_L,
                   "RN_U":RN_U,
                   "IOU":IOU_M,
                   "IOU_L":IOU_L,
                   "IOU_U":IOU_U,
                   "F1_M":F1_M,
                   "F1_L":F1_L,
                   "F1_U":F1_U,
                   "VR":VR_M,
                   "VR_L":VR_L,
                   "VR_U":VR_U}
overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = mean_over_folds_row

#Save the summary df
summary_save_name = method_name + "_mean_" + set_to_consider + "_mod" + str(
        mod) + "_quality-" + filter_for_quality + ".csv"
if save_results == True:
    overall_CV_fold_results_df.to_csv(save_path + summary_save_name)











