import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from skimage.restoration import denoise_bilateral
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries
import VolcDictionaryWithCorrectClears

def show(image):
    if plot_stuff == True:
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.show()
    else:
        pass

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + "/" + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

def read_sample(sample_index, df, timesteps):
    # Create a sequence of timestep images
    sequence = []
    names = []
    dictionary_name = df["volcano_dictionary_name"][sample_index]
    batch = df["labelling_batch_name"][sample_index]
    dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
    smmn_A = dictionary["sensor_marks_mask_A"]
    smmn_B = dictionary["sensor_marks_mask_B"]
    all_plume_mask = None

    for timestep_name in timesteps:
        if timestep_name == "image_name":
                folder_to_read = data_path
        elif timestep_name == "image_name_B":
            folder_to_read = data_path
        else:
            folder_to_read = data_path_temporal
        name_to_read = df[timestep_name][sample_index]
        timestep_image = cv2.imread(folder_to_read + "/" + name_to_read, -1)


        if "fltrB" in name_to_read:
            timestep_image = mask_sensor_marks(timestep_image, smmn_B)
        else:
            timestep_image = mask_sensor_marks(timestep_image, smmn_A)


        sequence.append(timestep_image)
        names.append(name_to_read)
        if timestep_name == "image_name":
            print(name_to_read)
            mask_path = segmentation_masks_path + batch + "/PlumeAndExpPixels_" + name_to_read.split(".")[0] + ".npy"
            # print("Reading plume mask from: " + mask_path)
            two_channel_mask = np.load(mask_path)  # Manually drawn plume mask (value 1 indicates plume)
            all_plume_mask = two_channel_mask[0, :, :] + two_channel_mask[1, :, :]
            all_plume_mask = np.where(all_plume_mask > 0, 1, 0)

        flank_mask = cv2.imread(dictionary["flank_mask_path"], -1)

    return sequence, names, all_plume_mask, flank_mask

def normalise_by_max_value(sequence, names, plot):
    '''Scale a sequence of images such that the 95th percentile brightness is equal.'''
    percentiles = []
    scaled_sequence = []
    sss = [] #Just recording for interest
    for index in range(0, len(sequence)):
        p95 = np.percentile(sequence[index], 95)
        print(p95)
        percentiles.append(p95)
        ss = int(names[index].split("_")[4][:-2])
        sss.append(ss)
    max_perc = max(percentiles)
    for index in range(0, len(sequence)):
        ratio = percentiles[index]/max_perc
        scaled_frame = np.divide(sequence[index], ratio)
        scaled_sequence.append(scaled_frame)

    if plot==True:
        fig, axs = plt.subplots(nrows=2, ncols=3)
        og_A = axs[0, 0].imshow(sequence[0], cmap="gray")
        og_B = axs[0, 1].imshow(sequence[1], cmap="gray")
        og_diff = axs[0, 2].imshow(np.abs(sequence[0] - sequence[1]))
        s_A = axs[1, 0].imshow(scaled_sequence[0], cmap="gray")
        s_B = axs[1, 1].imshow(scaled_sequence[1], cmap="gray")
        s_diff = axs[1, 2].imshow(np.abs(scaled_sequence[0] - scaled_sequence[1]))
        fig.colorbar(og_A, ax=axs[0, 0], shrink=0.5)
        fig.colorbar(og_B, ax=axs[0, 1], shrink=0.5)
        fig.colorbar(og_diff, ax=axs[0, 2], shrink=0.5)
        fig.colorbar(s_A, ax=axs[1, 0], shrink=0.5)
        fig.colorbar(s_B, ax=axs[1, 1], shrink=0.5)
        fig.colorbar(s_diff, ax=axs[1, 2], shrink=0.5)
        axs[0, 0].set_title(str(sss[0]))
        axs[0, 1].set_title(str(sss[1]))
        plt.show()
    return scaled_sequence

def scale_to_range(sequence, max_value, plot):
    '''Scale the whole sequence such that the max value is given.'''
    scaled_sequence = []
    seq_max = np.max(sequence)
    ratio = max_value/seq_max
    for frame_ix in range(0, len(sequence)):
        scaled_sequence.append(sequence[frame_ix] * ratio)
    return scaled_sequence

def pixel_diff(current, next):
    current = denoise_bilateral(current.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    next = denoise_bilateral(next.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    diff = np.abs(next.astype("float32") - current.astype("float32"))
    return diff

def calc_rel_AA(sequence, sequence_B, flank_mask):
    edge_mask = np.where(sequence_B[0] == 0, 5, 0)
    edge_mask = cv2.blur(edge_mask, ksize=(5, 5))
    ratio = np.divide(sequence_B[0], sequence[0])
    masked_ratio = np.ma.masked_where(edge_mask > 0, ratio)
    rel_AA = np.ma.log(masked_ratio)
    flank_AA = np.ma.median(np.ma.masked_where(flank_mask == 1, rel_AA))
    rel_AA = rel_AA - flank_AA
    rel_AA = np.ma.filled(rel_AA, fill_value=0)
    rel_AA = np.where(rel_AA < 0, 0, rel_AA)
    # Subtract such that the mean AA over the flank is zero
    rel_AA = np.where(flank_mask == 0, 0, rel_AA)
    return rel_AA
def precision(plume_mask, activation, thresholds):
    '''Calculate the precision of the prediction of plume pixels at given set of threshold vals.'''
    precisions = []
    for threshold in thresholds:
        predicted_plume = np.where(activation >= threshold, 1, 0)
        if np.sum(predicted_plume) > 0:
            correct_plume = np.where(plume_mask>0, predicted_plume, 0)
            p = np.sum(correct_plume)/np.sum(predicted_plume)
        else:
            p=1
        precisions.append(p)
    return precisions

def recall(plume_mask, activation, thresholds):
    '''Calculate which proportion of the plume is identified, given a set of threshold values.'''
    recalls = []
    for threshold in thresholds:
        if np.sum(plume_mask) > 0:
            predicted_plume = np.where(activation>=threshold, 1, 0)
            predicted_plume = np.where(plume_mask>0, predicted_plume, 0)
            #show(predicted_plume)
            #show(plume_mask)
            r = np.sum(predicted_plume)/np.sum(plume_mask)
        else:
            r = np.nan
        recalls.append(r)
    return recalls

def calc_hist(image, plot):
    n_bins=40
    counts, bins = np.histogram(image.flatten(), n_bins, [np.min(image), np.max(image)])
    counts[0] = 0 #Set the first bin count value to zero to allow detection of the first peak
    bin_length = np.abs(np.max(image) - np.min(image))/n_bins
    bin_centers = np.linspace(bin_length/2, bin_length * (n_bins - 1 + 0.5), n_bins)
    peaks, properties = scipy.signal.find_peaks(counts, prominence=np.max(counts)/10, distance=5, width=1)

    threshold = bin_centers[peaks[-1]] + (properties["widths"][-1] * bin_length)#TODO plus 1/2 peak width?
    lower_threshold = bin_centers[peaks[-1]]

    if plot == True:
        fit = np.polyfit(bin_centers, counts, deg=10)
        p = np.poly1d(fit)
        plt.stairs(counts, bins)
        plt.plot(bin_centers, p(bin_centers))
        plt.scatter(bin_centers, np.ones_like(bin_centers))
        plt.plot(bin_centers[peaks], counts[peaks], "x")
        print(properties["widths"])
        plt.axvline(x=threshold)
        plt.show()
    return threshold, lower_threshold

locations = ["Cotopaxi"]
df_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/"
timesteps = ["image_name", "next_tensec_name"]
timesteps_B = ["image_name_B", "next_tensec_name_B"]
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
segmentation_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/"
sensor_mark_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/"
flank_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/"
mod = 1
save_results = False
plot_stuff = True

for llo in locations:
    results_df = pd.DataFrame(columns=["image_name", "IOU"])
    print("Running tests on " + llo + "-left-out CV Fold.")
    train_df = pd.read_excel(df_path + llo + "LeftOut_Train.xlsx")
    train_df = train_df[train_df["overall_obs"] == "No"]
    # train_df = train_df[train_df["image_name"].str.contains("Merapi")]
    train_df.reset_index(inplace=True)
    # For each sample
    for sample_index in range(0, train_df.shape[0], mod):
        print(sample_index)
        # Read the timestep sequence
        sequence, names, plume_mask, flank_mask = read_sample(sample_index, train_df, timesteps)
        sequence_B, names_B, NA, flank_mask = read_sample(sample_index, train_df, timesteps_B)

        # Now attempt segmentation:
        sequence = np.array(sequence).astype(np.float32)
        #sequence_B = np.array(sequence_B).astype(np.float32)
        masked = np.where(plume_mask == 1, 1, sequence[0])

        scaled_sequence = normalise_by_max_value(sequence, names, plot=False)
        scaled_sequence_B = normalise_by_max_value(sequence_B, names_B, plot=False)

        double_scaled_sequence = scale_to_range(scaled_sequence, max_value=100, plot=False)
        double_scaled_sequence_B = scale_to_range(scaled_sequence_B, max_value=100, plot=False)



        double_scaled_difference = pixel_diff(double_scaled_sequence[0], double_scaled_sequence[1])
        double_scaled_difference_B = pixel_diff(double_scaled_sequence_B[0], double_scaled_sequence_B[1])

        #fig, axs = plt.subplots(ncols=3)
        #plot1 = axs[0].imshow(double_scaled_difference)
        #plot2 = axs[1].imshow(double_scaled_difference_B)
        #plot3 = axs[2].imshow(double_scaled_difference - double_scaled_difference_B)
        #fig.colorbar(plot1, ax=axs[0], shrink=0.5)
        #fig.colorbar(plot2, ax=axs[1], shrink=0.5)
        #fig.colorbar(plot3, ax=axs[2], shrink=0.5)
        #plt.show()

        rel_AA = calc_rel_AA(sequence, sequence_B, flank_mask)

        # Mask out flank (and a little more!)
        flank_mask = cv2.blur(np.where(flank_mask==0, 5, 0), ksize=(10, 10))
        #show(flank_mask)
        double_scaled_difference = np.where(flank_mask>0, 0, double_scaled_difference)
        rel_AA = np.where(flank_mask>0, 0, rel_AA)

        d_thresh, d_l_thresh = calc_hist(double_scaled_difference, plot=False)
        diff_pts = np.where(double_scaled_difference > d_thresh, 0.5, 0) + np.where(double_scaled_difference > d_l_thresh, 0.5, 0)


        AA_threshold, AA_lower_threshold = calc_hist(rel_AA, plot=False)
        abs_pts = np.where(rel_AA>AA_threshold, 0.5, 0) + np.where(rel_AA>AA_lower_threshold, 0.5, 0)

        max_val = np.max(sequence[0])
        darkness_img = max_val + 1 - sequence[0].astype("float32")
        sky_pixels = np.ma.masked_where(flank_mask>0, darkness_img)
        dark_ratio = np.divide(sky_pixels, np.ma.max(sky_pixels))
        dark_pixels = np.where(flank_mask>0, 0, dark_ratio)

        total = np.multiply(dark_pixels, diff_pts * abs_pts)

        if 1 == 1:
            h = 2
            fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(5 * h, 2 * h * (486 / 648)))
            axs[0, 0].imshow(masked, cmap="gray")
            axs[1, 0].imshow(sequence[0], cmap="gray")
            axs[0, 1].imshow(double_scaled_difference, cmap="gray")
            axs[1, 1].imshow(diff_pts, cmap="gray")
            axs[0, 2].imshow(rel_AA, cmap="gray")
            axs[1, 2].imshow(abs_pts, cmap="gray")
            axs[0, 3].imshow(dark_pixels, cmap="gray")
            axs[1, 3].imshow(np.ones_like(dark_pixels), cmap="gray")
            axs[0, 4].imshow(total, cmap="gray")
            axs[1, 4].imshow(np.ones_like(total), cmap="gray")
            plt.subplots_adjust(wspace=0, hspace=0)
            for row in range(0, 2):
                for col in range(0, 5):
                    axs[row, col].set_xticklabels([])
                    axs[row, col].set_yticklabels([])
            plt.show()















