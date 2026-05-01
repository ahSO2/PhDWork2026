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

def pixel_diff(current, next):
    current = denoise_bilateral(current.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    next = denoise_bilateral(next.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    diff = np.abs(next.astype("float32") - current.astype("float32"))
    return diff

def equally_scale_sequences(s1, s2):
    '''Scale both band A and band B sequences to the range [0,255],
    so that the max value collectively is 255. As both bands are
    scaled equally, this means that absorbance calculation is still
    valid.'''
    sf = max(np.max(s1), np.max(s2)) / 255
    s1 = np.divide(s1, sf)
    s2 = np.divide(s2, sf)
    return s1, s2

def normalise_for_ss(sequence, names):
    #Start with images in range [0,1023]
    scaled_sequence = []
    for index in range(0, len(sequence)):
        name = names[index]
        ss = int(name.split("_")[4][:-2])
        #print(name)
        #print(ss)
        ssr = 1000000/ss
        scaled_sequence.append(sequence[index].astype("float32") * ssr)
    return scaled_sequence

def adaptive_threshold(image, dim):
    kernel = np.ones((dim, dim)) * (1/(dim*dim))
    threshold = cv2.filter2D(image, -1, kernel) + 0.1
    result = np.where(image > threshold, 1, 0)
    return result



def calc_hist(image):
    n_bins=20
    counts, bins = np.histogram(image.flatten(), n_bins, [np.min(image), np.max(image)])
    counts[0] = 0 #Set the first bin count value to zero to allow detection of the first peak
    bin_length = np.abs(np.max(image) - np.min(image))/n_bins
    bin_centers = np.linspace(bin_length/2, bin_length * (n_bins - 1 + 0.5), n_bins)
    #fit = np.polyfit(bin_centers, counts, deg=10)
    #p = np.poly1d(fit)
    peaks, properties = scipy.signal.find_peaks(counts, prominence=np.mean(counts)/5, distance=5, width=1)
    #plt.stairs(counts, bins)
    #plt.plot(bin_centers, p(bin_centers))
    #plt.scatter(bin_centers, np.ones_like(bin_centers))
    #plt.plot(bin_centers[peaks], counts[peaks], "x")
    #print(properties["widths"])
    threshold = bin_centers[peaks[-1]] + (properties["widths"][-1] * bin_length)#TODO plus 1/2 peak width?
    lower_threshold = bin_centers[peaks[-1]]
    #plt.axvline(x=threshold)

    #plt.show()
    return threshold, lower_threshold

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

def grabCut(image, activation):
    ####By Rother et al, a derivative of graph-cut method by Boykov and Jolly
    #diff_p = np.percentile(diff.flatten(), 50)

    activation_mask = np.where(activation > 0, 3, 2).astype(np.uint8) + np.where(activation>1, -2, 0).astype(np.uint8)
    show(activation_mask)

    image = (image/4).astype("uint8")
    image_c = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bg_model = np.zeros((1,65), np.float64)
    fg_model = np.zeros((1,65), np.float64)
    mask, bg_model, fg_model = cv2.grabCut(image_c, mask =activation_mask, rect=None, bgdModel=bg_model, fgdModel=fg_model, iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
    #show(mask)

    fix, axs = plt.subplots(1,2)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(np.where((mask==1)|(mask==3), image, 0), cmap="gray")
    plt.show()

def min_max_scale(img):
    img = img - np.min(img)
    sf = 255/np.max(img)
    return (img.astype(np.float32) * sf).astype(np.uint8)

def superpixels(image, diff, abs, k, comp):
    c1 = min_max_scale(image)
    c2 = min_max_scale(diff)
    c3 = min_max_scale(abs)
    three_channel = np.stack([c1, c2, c3], axis=-1)
    #segments_fz = felzenszwalb(three_channel, scale=100, sigma=0.5, min_size=50)
    print(k)
    print(comp)
    segments_slic = slic(three_channel, n_segments=k, compactness=comp, sigma=1, start_label=1)
    #print(segments_slic.shape)
    if plot_stuff == True:
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(segments_slic)
        ax[0].set_title("Superpixels")
        ax[1].imshow(mark_boundaries(three_channel, segments_slic))
        ax[1].set_title('Boundaries')
        plt.show()

    return segments_slic

def superpixel_consistency(sps, plume_mask):
    '''How consistent is each superpixel in terms of its main class?'''
    if np.sum(plume_mask) >0: #If we have plume in the image
        #For each superpixel, calculate the percentage of pixels that are plume.
        #Calculate the consistency as x if x>=50%, or 100-x if x<50%.
        plume_majority_consistencies = []
        bg_majority_consitencies = []
        for pixel_index in range(1, np.max(sps) + 1):
            n_pixels = np.sum(np.where(sps==pixel_index, 1 , 0))
            rel_pixels = np.where(sps==pixel_index, plume_mask, 0)
            plume_prop = np.sum(rel_pixels)/n_pixels
            if plume_prop >= 0.5:
                plume_majority_consistencies.append(plume_prop)
            else:
                bg_majority_consitencies.append(1-plume_prop)

        if len(plume_majority_consistencies) == 0: #If no superpixels are segmenting only plume
            plume_mean = 0
            bg_mean = np.mean(bg_majority_consitencies)
        elif len(bg_majority_consitencies) == 0: #If no pixels are segmenting only background
            if np.sum(plume_mask) > 486*648*(1/np.max(sps)): #Unless the true background area is less than the size of one superpixel
                bg_mean = 0 #Penalise the background consistency mean
                plume_mean = np.mean(plume_majority_consistencies)
            else:
                bg_mean = 1
                plume_mean = np.mean(plume_majority_consistencies)
        else:
            plume_mean = np.mean(plume_mask)
            bg_mean = np.mean(bg_majority_consitencies)

        return np.round((plume_mean + bg_mean)/2,4)
    else: #If there is no plume in the image
        return 1

def calc_IOU(m1, m2):
    I = np.where(np.logical_and(m1>0, m2>0), 1, 0)
    U = np.where(np.logical_or(m1>0, m2>0), 1, 0)
    i = np.sum(I)
    u = np.sum(U)
    if u != 0:
        return i/u
    else:
        return 0

def superpixel_IOU(sps, plume_mask):
    if np.sum(plume_mask) > 0:
        result = np.zeros_like(sps)
        for sup_index in range(1, np.max(sps)): #For each superpixel
            #If more than 5% of it overlaps with the plume mask
            sup_pixels = np.where(sps==sup_index, 1, 0)
            plume_pixels = np.where(sps==sup_index, plume_mask, 0)
            overlap = np.sum(plume_pixels)/np.sum(sup_pixels)
            if overlap >= 0.05:
                #Select those pixels
                result = np.where(sps==sup_index, 1, result)
        IOU = calc_IOU(result, plume_mask)
        #fig, axs = plt.subplots(ncols=3)
        #axs[0].imshow(plume_mask, cmap="gray")
        #axs[1].imshow(sps)
        #axs[2].imshow(result)
        #axs[2].set_title(str(np.round(IOU, 2)))
        #plt.show()
    else:
        IOU = 1 #If there is no plume, then give a perfect value
    return IOU

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


################### Main script:
#For each cross-valid split
#locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
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
save_results = True
plot_stuff = True

features_save_path = "C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/"
results_save_path = "C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/24 - Superpixel algorithm/"

for llo in locations:
    #os.mkdir(features_save_path + "wo" + llo)
    results_df = pd.DataFrame(columns=["image_name", "IOU"])
    print("Running tests on " + llo + "-left-out CV Fold.")
    train_df = pd.read_excel(df_path + llo + "LeftOut_Train.xlsx")
    train_df = train_df[train_df["overall_obs"] == "No"]
    #train_df = train_df[train_df["image_name"].str.contains("Merapi")]
    train_df.reset_index(inplace=True)
    # For each sample
    for sample_index in range(47, train_df.shape[0], mod):
        print(sample_index)
        # Read the timestep sequence
        sequence, names, plume_mask, flank_mask = read_sample(sample_index, train_df, timesteps)
        sequence_B, names_B, NA, flank_mask = read_sample(sample_index, train_df, timesteps_B)

        #Now attempt segmentation:
        sequence = np.array(sequence).astype(np.float32)
        sequence_B = np.array(sequence_B).astype(np.float32)
        #show(sequence[0] - sequence_B[0])
        #show(sequence[0])
        masked = np.where(plume_mask==1, 1, sequence[0])
        #show(masked)

        #Calculate the difference image
        scaled_sequence = normalise_for_ss(sequence, names)
        #scaled_sequence_B = normalise_for_ss(sequence_B, names_B)

        #unscaled_diff = pixel_diff(sequence[0], sequence[1])
        difference = pixel_diff(sequence[0], sequence[1])

        #Calculate relative absorbance
        #Take log of bandB/bandA, for the current timestep image
        rel_AA = calc_rel_AA(sequence, sequence_B, flank_mask)
        #show(rel_AA)

        #Goal 1: Select points which are likely to be plume

        #Mask out flank (and a little more!)
        flank_mask = cv2.blur(np.where(flank_mask==0, 5, 0), ksize=(10, 10))
        #show(flank_mask)
        difference = np.where(flank_mask>0, 0, difference)
        rel_AA = np.where(flank_mask>0, 0, rel_AA)
        #show(difference)
        #show(rel_AA)


        #Select points which are moving above the local mean
        #show(difference)
        difference = denoise_bilateral(difference.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
        #show(difference)
        #diff_pts = cv2.adaptiveThreshold(difference.astype("uint8"), maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=21, C=0)
        d_thresh, d_l_thresh = calc_hist(difference)
        diff_pts = np.where(difference > d_thresh, 0.5, 0) + np.where(difference > d_l_thresh, 0.5, 0)
        #show(diff_pts)

        #Select points which are absorbing above local mean
        AA_threshold, AA_lower_threshold = calc_hist(rel_AA)
        abs_pts = np.where(rel_AA>AA_threshold, 0.5, 0) + np.where(rel_AA>AA_lower_threshold, 0.5, 0)
        #show(abs_pts)

        max_val = np.max(sequence[0])
        darkness_img = max_val + 1 - sequence[0].astype("float32")
        sky_pixels = np.ma.masked_where(flank_mask>0, darkness_img)
        dark_ratio = np.divide(sky_pixels, np.ma.max(sky_pixels))
        dark_pixels = np.where(flank_mask>0, 0, dark_ratio)
        #show(dark_pixels)

        #denoised_AA = denoise_bilateral(rel_AA.astype("float32"), sigma_color=5, sigma_spatial=10, win_size=20)
        #show(denoised_AA)
        k = 400
        comp = 100
        sp = superpixels(sequence[0], difference, rel_AA, k, comp)
        IOU = superpixel_IOU(sp, plume_mask)

        #Goal 2: Take those points and
        total = np.multiply(dark_pixels, diff_pts * abs_pts)
        #show(total)
        if plot_stuff == True:
            h = 2
            fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(5 * h, 2 * h * (486/648)))
            axs[0,0].imshow(masked, cmap="gray")
            axs[1,0].imshow(sequence[0], cmap="gray")
            axs[0,1].imshow(difference, cmap="gray")
            axs[1, 1].imshow(diff_pts, cmap="gray")
            axs[0,2].imshow(rel_AA, cmap="gray")
            axs[1, 2].imshow(abs_pts, cmap="gray")
            axs[0,3].imshow(dark_pixels, cmap="gray")
            axs[1, 3].imshow(np.ones_like(dark_pixels), cmap="gray")
            axs[0,4].imshow(total, cmap="gray")
            axs[1, 4].imshow(np.ones_like(total), cmap="gray")
            plt.subplots_adjust(wspace=0, hspace=0)
            for row in range(0, 2):
                for col in range(0, 5):
                    axs[row, col].set_xticklabels([])
                    axs[row, col].set_yticklabels([])
        #axs.set_xticks([0, 1, 2, 3, 4], labels = ["Original", "Difference", "Absorbance", "Darkness", "Selection"])
        #axs.set_xlabel("Learned Filters")
        #axs.set_yticks(np.arange(len(c_labels)).tolist(), labels=c_labels)
            plt.show()




        '''
        fig, axs = plt.subplots()
        colors = ["c", "m"]
        labels = ["not plume", "plume"]
        index = 0
        for mask in [1, 0]:
            d = np.ma.masked_where(np.logical_or(plume_mask == mask,edge_mask > 0), difference).compressed()
            a = np.ma.masked_where(np.logical_or(plume_mask == mask,edge_mask > 0), rel_AA).compressed()
            axs.scatter(d, a, c=colors[index], alpha=0.5, label=labels[index])
            index += 1
        axs.legend()
        plt.xlabel("Pixel difference")
        plt.ylabel("Relative absorbance")
        plt.show()

        #show(sequence[0])
        #show(difference)
        #show(rel_AA)
        '''
        #thresholds = [0, 0.25, 0.5, 0.75]
        #precisions = precision(plume_mask, total, thresholds)
        #recalls = recall(plume_mask, total, thresholds)
        results_df.loc[len(results_df)] = [names[0], IOU]

    if save_results == True:
        results_df.to_excel(results_save_path + llo + "LeftOutFold_k" + str(k) + ".xlsx")








