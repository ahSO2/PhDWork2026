import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
import sys

from EvaluationFunctions import *
from ReadingFunctions import *
from RefAreaActivationFunctions import *
from RefAreaSelectionFunctions import *

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
#locations = ["Reventador"]
filter_for_quality = "Good"
set_to_consider = "UnseenTest"
mod = 11
grid_size = 30
timesteps = ["image_name", "next_tensec_name", "prev_tensec_name"]
to_predict = "Background"
save_results = False
save_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/"
rng = numpy.random.default_rng(42) #Create a random number generator with seed
paths_dictionary = {"df_path":"C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/",
                    "data_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026/",
                    "data_path_temporal":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026_Temporal/",
                    "segmentation_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/",
                    "sensor_mark_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/SensorMarkMasks/",
                    "flank_masks_path":"C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"}

############ Script
columns = ["image_name", "volcano_name", "AUC_PR", "R_AP99", "R_AP95", "R_AP90", "P_AR99"] #Metrics to save per image
summary_columns = ["fold", "AUC_PR", "AUC_PR_L", "AUC_PR_U", "R_AP99", "R_AP99_L", "R_AP99_U", "R_AP95", "R_AP95_L", "R_AP95_U", "R_AP90", "R_AP90_L", "R_AP90_U", "P_AR99", "P_AR99_L", "P_AR99_U"]
overall_CV_fold_results_df = pd.DataFrame(columns=summary_columns) #Dataframe to store summary results for each fold
all_samples_metrics = pd.DataFrame(columns=columns)
for llo in locations: #Leaving out one location at a time

    metrics_df = pd.DataFrame(columns=columns)

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

    print("Reading Samples:")
    for sample_index in range(0, samples_df.shape[0], mod):
        print(sample_index)
        # Read the timestep sequence
        volc_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(samples_df["volcano_dictionary_name"][sample_index])
        sequence, names, plume_mask, flank_mask = read_sample(sample_index, samples_df, timesteps, "A", volc_dictionary, paths_dictionary)
        sequence_B, names_B, NA, flank_mask = read_sample(sample_index, samples_df, timesteps, "B", volc_dictionary, paths_dictionary)
        show(sequence[0])
        '''Each method may include its own masking of pixels e.g. when absorbance is 
        calculated. The flank is masked out of the selected ref-areas below, incase this
        hasn't been done already in the method. '''

        #Blur my drawn flank masks slightly to ensure all flank pixels are excluded
        smoothed_flank_mask = cv2.blur(flank_mask * 5, (20, 20))
        flank_mask = np.where(smoothed_flank_mask < 5, 0, 1)

        bandB_zero = np.where(sequence_B[0] == 0, 5, 0)
        edge_mask = cv2.blur(bandB_zero, (11, 11))
        edge_mask[:, 0:6] = 1
        edge_mask[:, -5:] = 1
        edge_mask[0:6, :] = 1
        edge_mask[-5:, :] = 1
        to_mask = np.where(flank_mask == 0, 1, 0) + edge_mask

        #Methods which instead return an activation
        #brightness, method_name = relative_brightness(sequence[0])
        #scaled_brightness, method_name = relative_brightness_99p(sequence[0])
        #ref_activ, method_name = relative_darkness(sequence[0])
        #inv_absorbance, method_name = inv_relative_absorbance(sequence[0], sequence_B[0], flank_mask)
        absorbance, method_name = relative_absorbance(sequence[0], sequence_B[0], flank_mask)
        #ref_activ, method_name = inv_diff_mvmt(sequence[0], sequence[1])
        #movement, method_name = mvmt_above_noise(sequence[0], sequence[1], sequence[2], flank_mask)
        #ref_activ, method_name = diff_mvmt(sequence[0], sequence[1], sequence[2])
        #abs_diff, method_name = rel_abs_temp_diff(sequence, sequence_B, edge_mask)
        #abv, method_name = abs_var(sequence[0], sequence_B[0], grid_size=grid_size, flank_mask=flank_mask, to_mask=to_mask)
        #adv, method_name = abs_diff_var(sequence, sequence_B, grid_size, edge_mask, to_mask)


        #### Combine the channels to form the activation
        ref_activ = absorbance
        #ref_activ = ero(ref_activ, 50, mask_out=to_mask, plot=False)
        #ref_activ = dil(ref_activ, 30, mask_out=to_mask, plot=True)

        #ref_activ = brightness - absorbance - 0.25 * movement - 5 * abs_activ_1 - 20 * abs_activ_2
        #ref_activ = np.ma.masked_where(to_mask > 0, ref_activ)

        #### Apply gap-filling/smoothing
        #ref_activ = function_over_grid(ref_activ, grid_size=50, function="mean", nan_fill=0, min_box_prop=0.75, plot=False)
        #show(ref_activ, colormap="gray")

        #ref_activ = ws(ref_activ, plot=True)
        #method_name = "AV"

        #to_superpixel = np.where(flank_mask > 0, brightness, 0)
        #to_superpixel = min_max_scale(to_superpixel)
        #sp = superpixels(to_superpixel, to_superpixel, to_superpixel, k=30, comp=7, plot=True)

        #show(brightness)
        #ref_smoothed = cross_bilateral_filter_fast(image=ref_activ, ref_img=brightness * 100, sigma_s=25, sigma_r=10, sa_s=10, sa_r=4)
        #fig, axs = plt.subplots(ncols=2)
        #axs[0].imshow(ref_activ, cmap="gray")
        #axs[1].imshow(ref_smoothed, cmap="gray", vmax=np.max(ref_activ), vmin=np.min(ref_activ))
        #axs[0].set_axis_off()
        #axs[1].set_axis_off()
        #plt.show()
        #show(ref_activ)

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
        #show(plume_mask)
        ground_truth_bg = np.where(plume_mask == 0, flank_mask, 0)
        ground_truth_plume = np.where(flank_mask == 1, plume_mask, 0)
        if to_predict == "Plume":
            ground_truth = ground_truth_plume
        else:
            ground_truth = ground_truth_bg

        if to_predict == "Background": #If we are trying to recall background, return AUC values of nan for cases with only a very small number of (probably erroneous) background pixels
            min_true_pos = 600
        else:
            min_true_pos = 1
        auc_pr, r_ap99, r_ap95, r_ap90, p_ar99 = AUC_PR(ref_activ, ground_truth, flank_mask=flank_mask, exclude_edges=True, bandB=sequence_B[0],
                        plot=False, min_true_pos=min_true_pos)

        if sample_index < 0: #Plot the selected background regions
            fig, axs = plt.subplots(nrows=2, ncols=3)
            axs[0,0].imshow(brightness, cmap="gray")
            axs[1,0].imshow(absorbance, cmap="gray")
            axs[0,1].imshow(movement, cmap="gray")
            axs[1,1].imshow(abs_activ_1, cmap="gray")
            axs[0,2].imshow(ground_truth, cmap="gray")
            axs[1,2].imshow(ref_activ, cmap="gray")
            plt.show()

        new_row = {"image_name": names[0],
                   "volcano_name": samples_df["volcano_name"][sample_index],
                   "AUC_PR": auc_pr,
                   "R_AP99":r_ap99, #Recall - max at precision >= 0.99
                   "R_AP95":r_ap95, #Recall - max at precision >= 0.95
                   "R_AP90":r_ap90, #Recall - max where precision >= 0.90
                   "P_AR99":p_ar99} #Precision - max where recall > 0.99 (Useful for plume ID channels)
        metrics_df.loc[len(metrics_df)] = new_row
        all_samples_metrics.loc[len(all_samples_metrics)] = new_row

    #Save the per-image metrics df #"_gs-" + str(grid_size) +
    per_image_metrics_save_name = method_name + "_wo" + llo + "_" + set_to_consider + "_mod" + str(
        mod) + "_quality-" + filter_for_quality + ".csv"
    if save_results == True:
        metrics_df.to_csv(save_path + per_image_metrics_save_name)

    #For this fold, calculate the mean of each metric over the samples, and the 95% bootstrap confidence interval
    AUC_M, AUC_L, AUC_U, rng = mean_with_95p_bootstrap(metrics_df["AUC_PR"].to_numpy(), rng)
    R_AP99, R_AP99_L, R_AP99_U, rng = mean_with_95p_bootstrap(metrics_df["R_AP99"].to_numpy(), rng)
    R_AP95, R_AP95_L, R_AP95_U, rng = mean_with_95p_bootstrap(metrics_df["R_AP95"].to_numpy(), rng)
    R_AP90, R_AP90_L, R_AP90_U, rng = mean_with_95p_bootstrap(metrics_df["R_AP90"].to_numpy(), rng)
    P_AR99, P_AR99_L, P_AR99_U, rng = mean_with_95p_bootstrap(metrics_df["P_AR99"].to_numpy(), rng)

    summary_row = {"fold": llo,
                   "AUC_PR":AUC_M,
                   "AUC_PR_L":AUC_L,
                   "AUC_PR_U":AUC_U,
                   "R_AP99":R_AP99,
                   "R_AP99_L":R_AP99_L,
                   "R_AP99_U": R_AP99_U,
                   "R_AP95": R_AP95,
                   "R_AP95_L": R_AP95_L,
                   "R_AP95_U": R_AP95_U,
                   "R_AP90":R_AP90,
                   "R_AP90_L":R_AP90_L,
                   "R_AP90_U": R_AP90_U,
                   "P_AR99":P_AR99,
                   "P_AR99_L":P_AR99_L,
                   "P_AR99_U":P_AR99_U}

    overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = summary_row

#Calculate the location-balanced mean metrics (with bootstrap CIs)
AUC_M, AUC_L, AUC_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["AUC_PR"].to_numpy(), all_samples_metrics["volcano_name"], rng)
R_AP99, R_AP99_L, R_AP99_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["R_AP99"].to_numpy(), all_samples_metrics["volcano_name"], rng)
R_AP95, R_AP95_L, R_AP95_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["R_AP95"].to_numpy(), all_samples_metrics["volcano_name"], rng)
R_AP90, R_AP90_L, R_AP90_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["R_AP90"].to_numpy(), all_samples_metrics["volcano_name"], rng)
P_AR99, P_AR99_L, P_AR99_U, rng = location_balanced_mean_with_95p_bootstrap(all_samples_metrics["P_AR99"].to_numpy(), all_samples_metrics["volcano_name"], rng)
mean_over_folds_row = {"fold": "location-balanced",
                       "AUC_PR":AUC_M,
                       "AUC_PR_L":AUC_L,
                       "AUC_PR_U":AUC_U,
                       "R_AP99": R_AP99,
                       "R_AP99_L": R_AP99_L,
                       "R_AP99_U": R_AP99_U,
                       "R_AP95": R_AP95,
                       "R_AP95_L": R_AP95_L,
                       "R_AP95_U": R_AP95_U,
                       "R_AP90": R_AP90,
                       "R_AP90_L": R_AP90_L,
                       "R_AP90_U": R_AP90_U,
                       "P_AR99":P_AR99,
                       "P_AR99_L":P_AR99_L,
                       "P_AR99_U":P_AR99_U}
overall_CV_fold_results_df.loc[len(overall_CV_fold_results_df)] = mean_over_folds_row

#Save the summary df
#"_gs-" + str(grid_size)
summary_save_name = method_name + "_mean_" + set_to_consider + "_mod" + str(
        mod) + "_quality-" + filter_for_quality + ".csv"
if save_results == True:
    overall_CV_fold_results_df.to_csv(save_path + summary_save_name)











