import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
import sys

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/ModelDevelopment/BackgroundRefAreaEval/")
from EvaluationFunctions import *
from ReadingFunctions import *
from RefAreaSelectionFunctions import *
from Show import *

sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
filter_for_quality = "Good"
set_to_consider = "UnseenTest"
mod = 3
n_bins = 20
plot = True
timesteps = ["image_name", "next_tensec_name", "prev_tensec_name"]
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
for llo in locations: #Leaving out one location at a time

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

        #Masking out the same values as in the background estimation functions
        #Then I can evaluate the distribution of the remaining background pixels

        inv_plume_mask = np.where(plume_mask == 0, 1, 0)
        ground_truth_bg = np.where(to_mask == 0, inv_plume_mask, 0)

        #TODO if the ground truth background contains a tiny number of pixels, omit it?

        #TODO I actually want to calcualte the distribution of the ratio
        bandA_copy = copy.deepcopy(sequence[0])
        bandA_copy = np.ma.masked_where(edge_mask > 0, bandA_copy)
        # Then calculate the absorbance (without accounting for backgrounds)
        ratio = np.ma.divide(bandA_copy.astype(np.float32), sequence_B[0].astype(np.float32))
        ratio = -1 * np.ma.log(ratio)

        background_pixel_vals = np.ma.masked_where(ground_truth_bg==0, ratio)
        show(background_pixel_vals)

        values = background_pixel_vals.compressed()
        counts, bins = np.histogram(values, n_bins, [np.nanmin(values), np.nanmax(values)])
        bin_length = np.abs(np.nanmax(values) - np.nanmin(values)) / n_bins

        if plot == True:
            bin_centers = np.nanmin(values) + np.linspace(bin_length / 2, bin_length * (n_bins - 1 + 0.5), n_bins)
            plt.stairs(counts, bins)
            plt.scatter(bin_centers, np.ones_like(bin_centers))
            plt.show()

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
