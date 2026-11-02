'''Running a grid search for the coefficients of the SLIC Superpixels algorithm.'''
import os

import numpy as np
import pandas as pd
import sys

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/ModelDevelopment/BackgroundRefAreaEval/")
from EvaluationFunctions import *
from ReadingFunctions import *
from RefAreaActivationFunctions import *
from RefAreaSelectionFunctions import *

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
#locations = ["Merapi"]
filter_for_quality = "Good"
set_to_consider = "UnseenTest"
mod = 1
timesteps = ["image_name", "next_tensec_name", "prev_tensec_name"]
to_predict = "Background"
save_results = True
save_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/Superpixels/"
random_seed = 42
paths_dictionary = {"df_path":"C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/",
                    "data_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026/",
                    "data_path_temporal":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026_Temporal/",
                    "segmentation_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/",
                    "sensor_mark_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/",
                    "flank_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"}

########################################### Params

bandA_mult = [1, 0]
bandB_mult = [1, 0]
activ_mult = [1, 0]
n_segments = [80]
compactness = [0.01, 0.1, 1, 10, 100]

########################################## Grid Search

def run_with_params(bandA_mult, bandB_mult, activ_mult, n_segments, compactness, comb_no):
    '''Create and evaluate the activation channel using the given parameters.'''
    rng = np.random.default_rng(random_seed)  # Create a random number generator with seed
    new_save_folder_path = save_path + "/param_comb_" + str(comb_no)

    if os.path.exists(new_save_folder_path):
        pass
    else:
        os.mkdir(new_save_folder_path)

    all_samples_metrics = pd.DataFrame(columns=["image_name", "volcano_name", "AUC_PR", "R_AP99", "R_AP95", "R_AP90", "P_AR99"]) #Save result for each image to allow easy overall evals
    overall_CV_fold_results_df = pd.DataFrame(columns=["band_A_mult", "band_B_mult", "activ_mult", "n_segments", "compactness", "fold", "AUC_PR", "AUC_PR_L", "AUC_PR_U", "R_AP99", "R_AP99_L", "R_AP99_U", "R_AP95", "R_AP95_L", "R_AP95_U", "R_AP90", "R_AP90_L", "R_AP90_U", "P_AR99", "P_AR99_L", "P_AR99_U"])

    for llo in locations:  #Effectively selecting one location as I'm using the unseen reserved location for each fold
        metrics_df = pd.DataFrame(columns=["image_name", "volcano_name", "AUC_PR", "R_AP99", "R_AP95", "R_AP90", "P_AR99"]) #Save per-image results for this location

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
            #show(sequence[0], colormap="gray")
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
            channel_d, method_name = abs_var(sequence[0], sequence_B[0], grid_size=30, flank_mask=flank_mask, to_mask=to_mask)
            channel_e, method_name = abs_diff_var(sequence, sequence_B, 30, edge_mask, to_mask)
            ref_activ = (0.5 * channel_a) + (-0.5 * channel_b) + (-1 * channel_c) + (-1 * channel_d) + (-1 * channel_e)

            ch_1 = bandA_mult * channel_a
            scaled_band_B, method_name_B = relative_brightness_99p(sequence_B[0])
            ch_2 = bandB_mult * scaled_band_B
            scaled_activation = min_max_scale(ref_activ)
            inv_activation = 1 - scaled_activation
            ch_3 = activ_mult * inv_activation

            sp = slic_superpixels(ch_1, ch_2, ch_3, k=n, comp=c, to_mask=to_mask,
                                  plot=False)

            ref_activ = function_over_superpixels(sp, ref_activ, plot=False)

            ground_truth_bg = np.where(plume_mask == 0, flank_mask, 0)
            ground_truth_plume = np.where(flank_mask == 1, plume_mask, 0)
            if to_predict == "Plume":
                ground_truth = ground_truth_plume
            else:
                ground_truth = ground_truth_bg

            if to_predict == "Background":  # If we are trying to recall background, return AUC values of nan for cases with only a very small number of (probably erroneous) background pixels
                min_true_pos = 600
            else:
                min_true_pos = 1

            auc_pr, r_ap99, r_ap95, r_ap90, p_ar99 = AUC_PR(ref_activ, ground_truth, flank_mask=flank_mask,
                                                            exclude_edges=True, bandB=sequence_B[0],
                                                            plot=False, min_true_pos=min_true_pos)

            new_row = {"image_name": names[0],
                       "volcano_name": samples_df["volcano_name"][sample_index],
                       "AUC_PR": auc_pr,
                       "R_AP99": r_ap99,  # Recall - max at precision >= 0.99
                       "R_AP95": r_ap95,  # Recall - max at precision >= 0.95
                       "R_AP90": r_ap90,  # Recall - max where precision >= 0.90
                       "P_AR99": p_ar99}  # Precision - max where recall > 0.99 (Useful for plume ID channels)

            metrics_df.loc[len(metrics_df)] = new_row
            all_samples_metrics.loc[len(all_samples_metrics)] = new_row

        # Save the per-image metrics df
        per_image_metrics_save_name = llo + "_" + set_to_consider + "_mod" + str(
                mod) + "_quality-" + filter_for_quality + ".csv"
        if save_results == True:
            metrics_df.to_csv(new_save_folder_path + "/" + per_image_metrics_save_name)

        # For this fold, calculate the mean of each metric over the samples, and the 95% bootstrap confidence interval
        AUC_M, AUC_L, AUC_U, rng = mean_with_95p_bootstrap(metrics_df["AUC_PR"].to_numpy(), rng)
        R_AP99, R_AP99_L, R_AP99_U, rng = mean_with_95p_bootstrap(metrics_df["R_AP99"].to_numpy(), rng)
        R_AP95, R_AP95_L, R_AP95_U, rng = mean_with_95p_bootstrap(metrics_df["R_AP95"].to_numpy(), rng)
        R_AP90, R_AP90_L, R_AP90_U, rng = mean_with_95p_bootstrap(metrics_df["R_AP90"].to_numpy(), rng)
        P_AR99, P_AR99_L, P_AR99_U, rng = mean_with_95p_bootstrap(metrics_df["P_AR99"].to_numpy(), rng)

        summary_row = {"fold": llo,
                       "band_A_mult":bandA_mult,
                       "band_B_mult":bandB_mult,
                       "activ_mult":activ_mult,
                       "n_segments":n_segments,
                       "compactness":compactness,
                       "AUC_PR": AUC_M,
                       "AUC_PR_L": AUC_L,
                       "AUC_PR_U": AUC_U,
                       "R_AP99": R_AP99,
                       "R_AP99_L": R_AP99_L,
                       "R_AP99_U": R_AP99_U,
                       "R_AP95": R_AP95,
                       "R_AP95_L": R_AP95_L,
                       "R_AP95_U": R_AP95_U,
                       "R_AP90": R_AP90,
                       "R_AP90_L": R_AP90_L,
                       "R_AP90_U": R_AP90_U,
                       "P_AR99": P_AR99,
                       "P_AR99_L": P_AR99_L,
                       "P_AR99_U": P_AR99_U}

        overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = summary_row

    print("Calculating summary results:")
    # Calculate the location-balanced mean metrics (with bootstrap CIs)
    AUC_M, AUC_L, AUC_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["AUC_PR"].to_numpy(),
                                                                         all_samples_metrics["volcano_name"], rng)
    R_AP99, R_AP99_L, R_AP99_U, rng = location_balanced_mean_with_95p_bootstrap(
        all_samples_metrics["R_AP99"].to_numpy(), all_samples_metrics["volcano_name"], rng)
    R_AP95, R_AP95_L, R_AP95_U, rng = location_balanced_mean_with_95p_bootstrap(
        all_samples_metrics["R_AP95"].to_numpy(), all_samples_metrics["volcano_name"], rng)
    R_AP90, R_AP90_L, R_AP90_U, rng = location_balanced_mean_with_95p_bootstrap(
        all_samples_metrics["R_AP90"].to_numpy(), all_samples_metrics["volcano_name"], rng)
    P_AR99, P_AR99_L, P_AR99_U, rng = location_balanced_mean_with_95p_bootstrap(
        all_samples_metrics["P_AR99"].to_numpy(), all_samples_metrics["volcano_name"], rng)
    mean_over_folds_row = {"fold": "location-balanced",
                           "band_A_mult": bandA_mult,
                           "band_B_mult": bandB_mult,
                           "activ_mult": activ_mult,
                           "n_segments": n_segments,
                           "compactness": compactness,
                           "AUC_PR": AUC_M,
                           "AUC_PR_L": AUC_L,
                           "AUC_PR_U": AUC_U,
                           "R_AP99": R_AP99,
                           "R_AP99_L": R_AP99_L,
                           "R_AP99_U": R_AP99_U,
                           "R_AP95": R_AP95,
                           "R_AP95_L": R_AP95_L,
                           "R_AP95_U": R_AP95_U,
                           "R_AP90": R_AP90,
                           "R_AP90_L": R_AP90_L,
                           "R_AP90_U": R_AP90_U,
                           "P_AR99": P_AR99,
                           "P_AR99_L": P_AR99_L,
                           "P_AR99_U": P_AR99_U}
    overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = mean_over_folds_row

    # Save the summary df
    summary_save_name = "Mean_" + set_to_consider + "_mod" + str(
        mod) + "_quality-" + filter_for_quality + ".csv"
    if save_results == True:
        overall_CV_fold_results_df.to_csv(new_save_folder_path + "/" + summary_save_name)

comb_no = 121
for m_1 in bandA_mult:
    for m_2 in bandB_mult:
        for m_3 in activ_mult:
            for n in n_segments:
                for c in compactness:

                    #Function to run grid search
                    print("Running comb_no " + str(comb_no) + "#################################")
                    result = run_with_params(m_1, m_2, m_3, n, c, comb_no)
                    comb_no += 1
