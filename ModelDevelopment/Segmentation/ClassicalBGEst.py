import os

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import scipy.signal
from skimage.restoration import denoise_bilateral
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries
import VolcDictionaryWithCorrectClears
from FastBilateral import *

def show(image, title=None):
    if plot_stuff == True:
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        if title != None:
            plt.title(title)
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
        #print(p95)
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
def precision(plume_mask, activation, thresholds, zero_mask):
    '''Calculate the precision of the prediction of plume pixels at given set of threshold vals.'''
    precisions = []
    for threshold in thresholds:
        predicted_plume = np.where(activation >= threshold, 1, 0)
        if np.sum(predicted_plume) > 0:
            if type(zero_mask) is np.ndarray:
                '''Mask out area where bandB is zero from consideration.'''
                predicted_plume = np.where(zero_mask > 0, 0, predicted_plume)

            correct_plume = np.where(plume_mask>0, predicted_plume, 0)
            #show(predicted_plume, "predicted_plume")
            #show(correct_plume, "correct_plume")
            p = np.sum(correct_plume)/np.sum(predicted_plume)
        else:
            p=1
        precisions.append(p)
    return precisions

def recall(plume_mask, activation, thresholds, zero_mask):
    '''Calculate which proportion of the plume is identified, given a set of threshold values.'''
    recalls = []
    if type(zero_mask) is np.ndarray:
        plume_mask = np.where(zero_mask > 0, 0, plume_mask)
    for threshold in thresholds:
        if np.sum(plume_mask) > 0:
            predicted_plume = np.where(activation>=threshold, 1, 0)
            predicted_plume = np.where(plume_mask>0, predicted_plume, 0)
            #show(predicted_plume, "predicted_plume")
            #show(plume_mask, "plume_mask")
            r = np.sum(predicted_plume)/np.sum(plume_mask)
        else:
            r = np.nan
        recalls.append(r)
    return recalls

def calc_hist(image, plot, peak_index=-1):
    '''"Peak_index param determines which of the identified peaks is used to set the thresholds.'''
    n_bins=40
    counts, bins = np.histogram(image.flatten(), n_bins, [np.min(image), np.max(image)])
    counts[0] = 0 #Set the first bin count value to zero to allow detection of the first peak
    bin_length = np.abs(np.max(image) - np.min(image))/n_bins
    bin_centers = np.linspace(bin_length/2, bin_length * (n_bins - 1 + 0.5), n_bins)
    peaks, properties = scipy.signal.find_peaks(counts, prominence=np.max(counts)/10, distance=5, width=1)


    upper_threshold = bin_centers[peaks[peak_index]] + (properties["widths"][peak_index] * bin_length)
    middle_threshold = bin_centers[peaks[peak_index]]
    lower_threshold = bin_centers[peaks[peak_index]] - (properties["widths"][peak_index] * bin_length)


    if plot == True:
        show(image)
        fit = np.polyfit(bin_centers, counts, deg=10)
        p = np.poly1d(fit)
        plt.stairs(counts, bins)
        plt.plot(bin_centers, p(bin_centers))
        plt.scatter(bin_centers, np.ones_like(bin_centers))
        plt.plot(bin_centers[peaks], counts[peaks], "x")
        print(properties["widths"])
        plt.axvline(x=upper_threshold)
        plt.axvline(x=middle_threshold)
        plt.axvline(x=lower_threshold)
        plt.show()
    return upper_threshold, middle_threshold, lower_threshold

def compare_hist(plume_mask, image):
    '''Compare the distribution of plume vs non-plume pixels.'''
    plume_pixels = np.ma.masked_where(plume_mask==0, image).compressed()
    #non_plume_pixels = np.ma.masked_where(plume_mask==1, image).compressed()
    #show(image)
    n_bins = 20
    cp, bp = np.histogram(plume_pixels, n_bins, [np.min(plume_pixels), np.max(plume_pixels)])
    cp[0] = 0
    cp = cp / np.max(cp)
    ca, ba = np.histogram(image.flatten(), n_bins, [np.min(image.flatten()), np.max(image.flatten())])
    ca[0] = 0
    ca = ca/ np.max(ca)
    plt.stairs(cp, bp, label="Plume")
    plt.stairs(ca, ba, label="All pixels")
    plt.legend()
    plt.show()

def get_2D_gauss_kernel(kernel_size, sigma):
    k = np.zeros((kernel_size, kernel_size))
    k[int(kernel_size/2), int(kernel_size/2)] = 1
    k = cv2.GaussianBlur(k, ksize=(kernel_size, kernel_size), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    #show(k)
    return k

def calc_bilateral_term(region_to_smooth, edge_image_region, gauss_s, gauss_r):
    central_index = int(np.floor(region_to_smooth.shape[0]/2))
    total = 0
    norm = 0
    for x in range(0, region_to_smooth.shape[1]):
        for y in range(0, region_to_smooth.shape[0]):
            #Select weight for distance from cental pixel
            s_weight = gauss_s[y, x] #TODO should this be altered to be a 1D gaussian
            #Calculate distance from central pixel in colour space
            d = np.abs(edge_image_region[central_index, central_index] - edge_image_region[y, x])
            r_weight = gauss_r[int(np.round(d, 0))]
            term = s_weight * r_weight * region_to_smooth[y, x]
            total += term
            norm += s_weight * r_weight
    return total/norm


def cross_bilateral_filter_bf(to_smooth, edge_image, sigma_s, sigma_r):
    #For each pixel in the image to be smoothed
    original_shape = to_smooth.shape
    gauss_width = max([sigma_s*2 + 1, 5]) #Must be odd - gives kernel size of gaussian filters
    pad_width = int(np.floor(gauss_width / 2))
    to_smooth = cv2.copyMakeBorder(to_smooth, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REFLECT)
    edge_image = cv2.copyMakeBorder(edge_image, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REFLECT)

    gauss_s = get_2D_gauss_kernel(gauss_width, sigma_s)
    gauss_r = cv2.getGaussianKernel(int(np.ceil(np.max(edge_image)) * 2 + 1), sigma_r)
    gauss_r = gauss_r[int(np.floor(gauss_r.shape[0]/2) + 1):].flatten()
    #show(np.stack([gauss_r] * 20, axis=0))

    result = np.zeros(shape=original_shape)
    for i in range(0, original_shape[1]):
        print(i)
        for j in range(0, original_shape[0]):
            central_pixel = [j + pad_width, i + pad_width]
            to_smooth_region = to_smooth[central_pixel[0] - pad_width:central_pixel[0] + pad_width + 1, central_pixel[1] - pad_width:central_pixel[1] + pad_width + 1]
            edge_image_region = edge_image[central_pixel[0] - pad_width:central_pixel[0] + pad_width + 1, central_pixel[1] - pad_width:central_pixel[1] + pad_width + 1]
            filtered_value = calc_bilateral_term(to_smooth_region, edge_image_region, gauss_s, gauss_r)
            result[j, i] = filtered_value

    fig, axs = plt.subplots(ncols=3)
    axs[0].imshow(edge_image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(to_smooth, cmap="gray")
    axs[1].set_title("Activation")
    axs[2].imshow(result, cmap="gray")
    axs[2].set_title("Smoothed")
    plt.show()


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

plot_stuff = True
#activation_thresholds = [0, 0.05, 0.1, 0.2, 0.3, "adp"]
#results_save_path = "C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/26 - Cross Bilateral Filter/"
#df_columns = []
#for threshold in activation_thresholds:
#    df_columns.append("pr_" + str(threshold))
#    df_columns.append("re_" + str(threshold))

for llo in locations:
    #results_df = pd.DataFrame(columns=["image_name"] + df_columns)
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
        #show(rel_AA)

        d_upper_thresh, d_middle_thresh, d_lower_thresh = calc_hist(double_scaled_difference, plot=False)
        diff_pts = np.where(double_scaled_difference > d_upper_thresh, 0.5, 0) + np.where(double_scaled_difference > d_middle_thresh, 0.5, 0)

        AA_upper_threshold, AA_middle_threshold, AA_lower_threshold = calc_hist(rel_AA, plot=False)
        abs_pts = np.where(rel_AA>AA_upper_threshold, 0.5, 0) + np.where(rel_AA>AA_middle_threshold, 0.5, 0)

        max_val = np.max(sequence[0])
        darkness_img = max_val + 1 - sequence[0].astype("float32")
        sky_pixels = np.ma.masked_where(flank_mask>0, darkness_img)
        dark_ratio = np.divide(sky_pixels, np.ma.max(sky_pixels))
        dark_pixels = np.where(flank_mask>0, 0, dark_ratio)

        #total = np.multiply(dark_pixels, diff_pts * abs_pts)
        total = np.multiply(diff_pts + abs_pts, dark_pixels)
        #show(total)
        t_u_t, t_m_t, t_l_t = calc_hist(total, plot=False, peak_index=0)
        #thresh_total = np.where(total>= t_m_t, total, 0)
        #compare_hist(plume_mask, total)

        #show(sequence[0][0:5, 0:5])
        #total = total * 100
        #show(total)
        #show(sequence[0])
        #filtered_total = cross_bilateral_filter_fast(total * 100, sequence[0], sigma_s=50, sigma_r=40, sa_s=10, sa_r=5)
        filtered_total = cross_bilateral_filter_fast(total * 100, sequence[0], sigma_s=30, sigma_r=30, sa_s=6, sa_r=6)
        filtered_total = filtered_total / 100
        thresh_filtered = np.where(filtered_total > t_m_t, 0, sequence[0])
        #show(filtered_total)
        #fig, axs = plt.subplots(ncols=4)
        #axs[0].imshow(sequence[0], cmap="gray")
        #axs[1].imshow(total * 100, cmap="gray")
        #axs[2].imshow(thresh_total * 100, cmap="gray")
        #axs[3].imshow(filtered_total, cmap="gray")
        #plt.show()
        #np.save(results_save_path + "bandA_" + names[0], sequence[0])
        #np.save(results_save_path + "activation_" + names[0], total)

        if 1 == 1:
            h = 2
            fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(5 * h, 2 * h * (486 / 648)))
            axs[0, 0].imshow(sequence[0], cmap="gray")
            axs[0, 0].set_title("Original 310nm", fontsize=10)
            axs[1, 0].imshow(masked, cmap="gray")
            axs[1, 0].set_xlabel("My label")
            axs[0, 1].imshow(double_scaled_difference, cmap="gray")
            axs[0, 1].set_title("Frame-to-frame difference", fontsize=7)
            axs[1, 1].imshow(diff_pts, cmap="gray")
            axs[1, 1].set_xlabel("Diff thresholded")
            axs[0, 2].imshow(rel_AA, cmap="gray", )
            axs[0, 2].set_title("Est. absorbance", fontsize=10)
            axs[1, 2].imshow(abs_pts, cmap="gray")
            axs[1, 2].set_xlabel("Absorbance thresholded")
            axs[0, 3].imshow(dark_pixels, cmap="gray")
            axs[0, 3].set_title("Darkness", fontsize=10)
            axs[1, 3].imshow(total, cmap="YlGnBu_r")
            axs[1, 3].set_xlabel("Activation")
            plot7 = axs[0, 4].imshow(thresh_filtered, cmap="gray")
            axs[0, 4].set_title("Selected BG Pixels", fontsize=10)
            plot8 = axs[1, 4].imshow(filtered_total, cmap="YlGnBu_r",vmax=np.max(total), vmin=np.min(total))
            axs[1, 4].set_xlabel("Filtered activation")
            plt.subplots_adjust(wspace=0, hspace=0)
            for row in range(0, 2):
                for col in range(0, 5):
                    axs[row, col].set_xticklabels([])
                    axs[row, col].set_yticklabels([])
            #fig.colorbar(plot8, ax=[axs[1,4], axs[0, 3]])
            plt.show()
            #plt.savefig(results_save_path + names[0][:-4] + "_BGRegionPrediction.png", dpi=200)
            #plt.close()

        df_row = [names[0]]
        band_B_mask = cv2.blur(np.where(sequence_B[0]==0, 5, 0), ksize=(10, 10))
        band_B_mask = np.where(band_B_mask > 0, 1, 0)

        #activation_thresholds[-1] = t_m_t
        #for threshold in activation_thresholds:
            #Calculate the precision and recall for that threshold
            #p = precision(plume_mask, total, [threshold], zero_mask=band_B_mask)[0]
            #r = recall(plume_mask, total, [threshold], zero_mask=band_B_mask)[0]
            #Add to the dataframe row
            #df_row = df_row + [p, r]

        #results_df.loc[len(results_df)] = df_row
    #results_df.to_excel(results_save_path + "PrecisionRecall.xlsx")
