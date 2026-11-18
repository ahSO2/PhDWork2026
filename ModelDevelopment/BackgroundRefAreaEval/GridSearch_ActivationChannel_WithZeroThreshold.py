'''Running a grid search for the coefficients used to combine the activation channels.'''
import os

import numpy as np
import pandas as pd
import sys

from EvaluationFunctions import *
from ReadingFunctions import *
from RefAreaActivationFunctions import *
from RefAreaSelectionFunctions import *

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
#locations = ["Kilauea"]
filter_for_quality = "Good"
set_to_consider = "UnseenTest"
mod = 1
grid_size = 30
timesteps = ["image_name", "next_tensec_name", "prev_tensec_name"]
to_predict = "Background"
save_results = True
save_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Combination_Params_Grid_Search_Zero_Threshold/"
random_seed = 42
paths_dictionary = {"df_path":"C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/",
                    "data_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026/",
                    "data_path_temporal":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026_Temporal/",
                    "segmentation_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/",
                    "sensor_mark_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/",
                    "flank_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"}

########################################### Params

a_range = [1, 0.5]
b_range = [-1, -0.5, 0]
c_range = [-1, -0.5, 0]
d_range = [-1, -0.5, 0]
e_range = [-1, -0.5, 0]

########################################## Grid Search

def run_with_params(a, b, c, d, e, comb_no):
    '''Create and evaluate the activation channel using the given parameters.'''
    rng = np.random.default_rng(random_seed)  # Create a random number generator with seed
    new_save_folder_path = save_path + "/param_comb_" + str(comb_no)

    if os.path.exists(new_save_folder_path):
        pass
    else:
        os.mkdir(new_save_folder_path)

    all_samples_metrics = pd.DataFrame(columns=["image_name", "volcano_name", "PP", "PN", "RP", "RN", "IOU", "F1", "PR"]) #Save result for each image to allow easy overall evals
    overall_CV_fold_results_df = pd.DataFrame(columns=["comb_no", "a", "b", "c", "d", "e", "fold", "PP", "PP_L", "PP_U", "PN", "PN_L", "PN_U", "RP", "RP_L", "RP_U", "RN", "RN_L", "RN_U", "IOU", "IOU_L", "IOU_U", "F1_M", "F1_L", "F1_U", "PR", "PR_L", "PR_U"])

    for llo in locations:  #Effectively selecting one location as I'm using the unseen reserved location for each fold
        metrics_df = pd.DataFrame(columns=["image_name", "volcano_name", "PP", "PN", "RP", "RN", "IOU", "F1", "PR"]) #Save per-image results for this location

        print("Running tests on " + llo + "-left-out CV Fold.")
        samples_df = pd.read_excel(paths_dictionary["df_path"] + llo + "_" + set_to_consider + ".xlsx")

        if filter_for_quality == "Good":
            samples_df = samples_df[samples_df["overall_obs"] == "No"]
            print("Selected " + str(samples_df.shape[0]) + " good quality samples.")
            samples_df.reset_index(inplace=True)
        elif filter_for_quality == "Low":
            samples_df = samples_df[samples_df["overall_obs"] == "Yes"]
            print("Selected " + str(samples_df.shape[0]) + " low quality samples.")
            samples_df.reset_index(inplace=True)

        for sample_index in range(0, samples_df.shape[0], mod):
            #print(sample_index)
            # Read the timestep sequence
            volc_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(samples_df["volcano_dictionary_name"][sample_index])
            sequence, names, plume_mask, flank_mask = read_sample(sample_index, samples_df, timesteps, "A",volc_dictionary, paths_dictionary)
            sequence_B, names_B, NA, flank_mask = read_sample(sample_index, samples_df, timesteps, "B", volc_dictionary, paths_dictionary)

            '''Each method may include its own masking of pixels e.g. when absorbance is 
               calculated. The flank is masked out of the selected ref-areas below, incase this
               hasn't been done already in the method. '''

            # Blur my drawn flank masks slightly to ensure all flank pixels are excluded
            smoothed_flank_mask = cv2.blur(flank_mask * 5, (20, 20))
            flank_mask = np.where(smoothed_flank_mask < 5, 0, 1)

            bandB_zero = np.where(sequence_B[0] == 0, 5, 0)
            edge_mask = cv2.blur(bandB_zero, (11, 11))
            edge_mask[:, 0:6] = 1
            edge_mask[:, -5:] = 1
            edge_mask[0:6, :] = 1
            edge_mask[-5:, :] = 1
            to_mask = np.where(flank_mask == 0, 1, 0) + edge_mask

            #Calculate the channels and combine as ref_activ
            channel_a, method_name = relative_brightness_99p(sequence[0])
            channel_b, method_name = relative_absorbance(sequence[0], sequence_B[0], flank_mask)
            channel_c, method_name = mvmt_above_noise(sequence[0], sequence[1], sequence[2], flank_mask)
            channel_d, method_name = abs_var(sequence[0], sequence_B[0], grid_size=grid_size, flank_mask=flank_mask, to_mask=to_mask)
            channel_e, method_name = abs_diff_var(sequence, sequence_B, grid_size, edge_mask, to_mask)
            ref_activ = (a * channel_a) + (b * channel_b) + (c * channel_c) + (d * channel_d) + (e * channel_e)
            #show(ref_activ)

            ground_truth_bg = np.where(plume_mask == 0, flank_mask, 0)
            ground_truth_plume = np.where(flank_mask == 1, plume_mask, 0)
            if to_predict == "Plume":
                ground_truth = ground_truth_plume
            else:
                ground_truth = ground_truth_bg

            if to_predict == "Background":
                if np.sum(ground_truth_bg) < 600:
                    ground_truth_bg = np.zeros_like(ground_truth_bg)

            #Create predicted background mask by thresholding at 0
            predicted_bg = np.where(ref_activ > 0, 1, 0)
            #TODO Evaluate the same params as used in the main script evals
            TP, TN, FP, FN = calculate_conf_counts(predicted_bg, ground_truth_bg, exclude=np.where(flank_mask > 0, 0, 1))
            PP, PN = per_class_precision(TP, TN, FP, FN)
            RP, RN = per_class_recall(TP, TN, FP, FN)
            IOU = intersection_over_union(TP, TN, FP, FN)
            F1 = F1_score(PP, RP)
            PR = percentile_ratio(sequence[0], sequence_B[0], predicted_bg, ground_truth_bg, flank_mask)

            new_row = {"image_name": names[0],
                       "volcano_name": samples_df["volcano_name"][sample_index],
                       "PP":PP,
                       "PN":PN,
                       "RP":RP,
                       "RN":RN,
                       "IOU":IOU,
                       "F1":F1,
                       "PR":PR}
            metrics_df.loc[len(metrics_df)] = new_row
            all_samples_metrics.loc[len(all_samples_metrics)] = new_row

        # Save the per-image metrics df
        per_image_metrics_save_name = llo + "_" + set_to_consider + "_mod" + str(
                mod) + "_quality-" + filter_for_quality + ".csv"
        if save_results == True:
            metrics_df.to_csv(new_save_folder_path + "/" + per_image_metrics_save_name)

        # For this fold, calculate the mean of each metric over the samples, and the 95% bootstrap confidence interval
        PP_M, PP_L, PP_U, rng = mean_with_95p_bootstrap(metrics_df["PP"].to_numpy(), rng)
        PN_M, PN_L, PN_U, rng = mean_with_95p_bootstrap(metrics_df["PN"].to_numpy(), rng)
        RP_M, RP_L, RP_U, rng = mean_with_95p_bootstrap(metrics_df["RP"].to_numpy(), rng)
        RN_M, RN_L, RN_U, rng = mean_with_95p_bootstrap(metrics_df["RN"].to_numpy(), rng)
        IOU_M, IOU_L, IOU_U, rng = mean_with_95p_bootstrap(metrics_df["IOU"].to_numpy(), rng)
        F1_M, F1_L, F1_U, rng = mean_with_95p_bootstrap(metrics_df["F1"].to_numpy(), rng)
        PR_M, PR_L, PR_U, rng = mean_with_95p_bootstrap(metrics_df["PR"].to_numpy(), rng)

        summary_row = {"fold": llo,
                       "comb_no":comb_no,
                       "a":a,
                       "b":b,
                       "c":c,
                       "d":d,
                       "e":e,
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
                   "PR":PR_M,
                   "PR_L":PR_L,
                   "PR_U":PR_U}

        overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = summary_row

    print("Calculating summary results:")
    # Calculate the location-balanced mean metrics (with bootstrap CIs)
    PP_M, PP_L, PP_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["PP"].to_numpy(),
                                                                      all_samples_metrics["volcano_name"], rng)
    PN_M, PN_L, PN_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["PN"].to_numpy(),
                                                                      all_samples_metrics["volcano_name"], rng)
    RP_M, RP_L, RP_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["RP"].to_numpy(),
                                                                      all_samples_metrics["volcano_name"], rng)
    RN_M, RN_L, RN_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["RN"].to_numpy(),
                                                                      all_samples_metrics["volcano_name"], rng)
    IOU_M, IOU_L, IOU_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["IOU"].to_numpy(),
                                                                         all_samples_metrics["volcano_name"], rng)
    F1_M, F1_L, F1_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["F1"].to_numpy(),
                                                                      all_samples_metrics["volcano_name"], rng)
    PR_M, PR_L, PR_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["PR"].to_numpy(),
                                                                      all_samples_metrics["volcano_name"], rng)

    mean_over_folds_row = {"fold": "location-balanced",
                           "comb_no":comb_no,
                           "a": a,
                           "b": b,
                           "c": c,
                           "d": d,
                           "e": e,
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
                   "PR":PR_M,
                   "PR_L":PR_L,
                   "PR_U":PR_U}
    overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = mean_over_folds_row

    # Save the summary df
    summary_save_name = "Mean_" + set_to_consider + "_mod" + str(
        mod) + "_quality-" + filter_for_quality + ".csv"
    if save_results == True:
        overall_CV_fold_results_df.to_csv(new_save_folder_path + "/" + summary_save_name)

comb_no = 1
for a in a_range:
    for b in b_range:
        for c in c_range:
            for d in d_range:
                for e in e_range:

                    #Function to run grid search
                    print("Running comb_no " + str(comb_no) + "#################################")
                    result = run_with_params(a, b, c, d, e, comb_no)
                    comb_no += 1
