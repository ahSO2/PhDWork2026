import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import watershed
from skimage.morphology import dilation, erosion, disk, footprint_rectangle

from ScalingFunctions import min_max_scale, scale_values_to_01, scale_by_percentile, hist_max_thresh, min_max_percentile
from Show import show

'''These functions output a numerical measure on the likely distribution
of plume or background in the image, rather than a binary mask.'''


def relative_brightness(bandA):
    brightness = min_max_scale(bandA)
    return brightness, "C-B"

def relative_brightness_99p(bandA):
    '''Min-max scale the image using the 1st and 99th percentile values, then threshold
    all pixels to range [0, 1]. For the purpose of making the channel more robust to
    small outiler values.'''

    upper = np.percentile(bandA.flatten(), 99)
    lower = np.percentile(bandA.flatten(), 1)
    scaled = scale_values_to_01(lower, upper, bandA)
    scaled = np.where(scaled > 1, 1, scaled)
    scaled = np.where(scaled < 0, 0, scaled)
    show(scaled, title="Scaled to [0, 1]")
    return scaled, "C-BS"

def relative_darkness(bandA):
    darkness = np.ones_like(bandA) - relative_brightness(bandA)[0]
    return darkness, "C-D"

def log_ratio_AA(bandA, bandB, edge_mask):
    bandA_copy = np.ma.masked_where(edge_mask > 0, bandA)
    ratio = np.ma.divide(bandA_copy.astype(np.float32), bandB.astype(np.float32))
    ratio = -1 * np.ma.log(ratio)
    return ratio
def relative_absorbance(bandA, bandB, flank_mask):
    '''Calculate the log-ratio of bandB/A, then scale such that the flank area
    has a mean absorbance of 0. Threshold the result to range [0,1].'''
    bandB_zero = np.where(bandB == 0, 5, 0)
    edge_mask = cv2.blur(bandB_zero, (11, 11))
    edge_mask[:, 0:6] = 1
    edge_mask[:, -5:] = 1
    edge_mask[0:6, :] = 1
    edge_mask[-5:, :] = 1

    ratio = log_ratio_AA(bandA, bandB, edge_mask)

    #Consider a set of flank pixels away from the sky-flank boundary
    shifted_flank_mask = np.concatenate([np.ones(shape=(20, flank_mask.shape[1])), flank_mask[0:-20, :]], axis=0)
    flank_mean = np.ma.masked_where(shifted_flank_mask == 1, ratio)
    flank_mean.mask = np.where(edge_mask > 0, 1, flank_mean.mask)
    flank_mean = np.ma.mean(flank_mean)

    # Scale the ratio such that the flank has a zero value
    ratio = ratio - flank_mean

    # Threshold to [0, 1]
    ratio = np.ma.where(ratio > 0, ratio, 0)
    ratio = np.ma.where(ratio < 1, ratio, 1)
    #show(ratio, colormap="YlGnBu_r")
    return ratio, "C-AA"

def inv_relative_absorbance(bandA, bandB, flank_mask):
    '''1 - relative absorbance, for thresholding for background.'''

    rel_abs = relative_absorbance(bandA, bandB, flank_mask)[0]
    inv = np.ones_like(rel_abs) - rel_abs

    return inv, "C-IAA"

def diff_mvmt(bandA, ts_bandA_n1, ts_bandA_p1):
    '''Calculate the absolute value of the difference between two bandA frames.'''
    #show(ts_bandA_n1)
    scaled_A_n1, scaled_ts_n1 = scale_by_percentile(bandA, ts_bandA_n1, 95)
    scaled_diff_n1 = np.abs(scaled_A_n1.astype(np.float32) - scaled_ts_n1.astype(np.float32))

    scaled_A_p1, scaled_ts_p1 = scale_by_percentile(bandA, ts_bandA_p1, 95)
    scaled_diff_p1 = np.abs(scaled_A_p1.astype(np.float32) - scaled_ts_p1.astype(np.float32))

    scaled_diff = cv2.blur(scaled_diff_n1 + scaled_diff_p1, (10, 10))
    scaled_diff = min_max_scale(scaled_diff)

    #scaled_diff_one_ts = cv2.blur(scaled_diff_p1, (10,10))
    #scaled_diff_one_ts = min_max_scale(scaled_diff_one_ts)

    #fig, axs = plt.subplots(ncols=2)
    #axs[0].imshow(scaled_diff_one_ts, cmap="gray")
    #axs[1].imshow(scaled_diff, cmap="gray")
    #plt.show()

    return scaled_diff, "C-M"

def inv_diff_mvmt(bandA, ts_bandA_n1, ts_bandA_p1):
    scaled_diff = diff_mvmt(bandA, ts_bandA_n1, ts_bandA_p1)[0]
    inv_scaled_diff = np.ones_like(scaled_diff) - scaled_diff
    return inv_scaled_diff, "C-IM"
def mvmt_above_noise(bandA, ts_bandA_n1, ts_bandA_p1, flank_mask):
    '''Select pixels with movement below the noise level as potential background pixels.'''
    scaled_diff = diff_mvmt(bandA, ts_bandA_n1, ts_bandA_p1)[0]
    scaled_diff = np.ma.masked_where(flank_mask == 0, scaled_diff)
    show(scaled_diff)
    c, l, r = hist_max_thresh(scaled_diff, plot=True)
    diff_thresh = np.where(scaled_diff > r, 1, 0)

    return diff_thresh, "C-M_t"

def rel_abs_temp_diff(bandA_seq, bandB_seq, edge_mask):
    '''Sequences should be input in order [bandA, one timestep, the other timestep]'''
    og_ratio = log_ratio_AA(bandA_seq[0], bandB_seq[0], edge_mask)
    #show(og_ratio)
    ts_ratio_1 = log_ratio_AA(bandA_seq[1], bandB_seq[1], edge_mask)
    ts_ratio_2 = log_ratio_AA(bandA_seq[2], bandB_seq[2], edge_mask)

    #Scaling so that the 95th percentile range of each image covers range [1, 0]
    #(Switched from scaling one percentile value to match the other as sometimes this results in scale factor zero if one percentile is zero).
    scaled_og = min_max_percentile(og_ratio, 95)
    scaled_ts_1 = min_max_percentile(ts_ratio_1, 95)
    scaled_ts_2 = min_max_percentile(ts_ratio_2, 95)

    AA_diff_1 = np.abs(scaled_og - scaled_ts_1)
    AA_diff_2 = np.abs(scaled_og - scaled_ts_2)

    AA_diff_1 = np.where(AA_diff_1 > 1, 1, AA_diff_1)
    AA_diff_2 = np.where(AA_diff_2 > 1, 1, AA_diff_2)

    AA_diff_1 = np.ma.masked_where(edge_mask > 0, AA_diff_1)
    AA_diff_2 = np.ma.masked_where(edge_mask > 0, AA_diff_2)

    #blur_size = 5
    #smoothed_diff_1 = cv2.blur(AA_diff_1, (blur_size, blur_size))
    #smoothed_diff_2 = cv2.blur(AA_diff_2, ksize=(blur_size, blur_size))

    #additional_mask = np.where(cv2.blur(edge_mask.astype(np.uint8) * blur_size, (blur_size, blur_size)) > 0, 1, 0)
    #smoothed_diff_1 = np.ma.masked_where(additional_mask, smoothed_diff_1)
    #smoothed_diff_2 = np.ma.masked_where(additional_mask, smoothed_diff_2)

    total_AA_diff = AA_diff_1 + AA_diff_2

    blur_size = 5
    smoothed_diff = cv2.blur(total_AA_diff, (blur_size, blur_size))
    additional_mask = np.where(cv2.blur(edge_mask.astype(np.uint8) * (blur_size + 1), (blur_size + 1, blur_size + 1)) > 0, 1, 0)
    smoothed_diff = np.ma.masked_where(additional_mask, smoothed_diff)

    #fig, axs = plt.subplots(ncols=2)
    #axs[0].imshow(total_AA_diff, cmap="YlGnBu_r")
    #axs[1].imshow(smoothed_diff, cmap="YlGnBu_r")
    #plt.show()
    return smoothed_diff, "C-AM"

def abs_var(imgA, imgB, grid_size, flank_mask, to_mask):
        '''to_mask indicates edge and flank pixels to exclude from the grid calculations.'''
        ab = relative_absorbance(imgA, imgB, flank_mask)[0]
        ab = np.ma.masked_where(to_mask > 0, ab)
        sd = function_over_grid(ab, grid_size=grid_size, function="var", nan_fill=0, max_masked_prop=0.90,
                                       plot=False)
        sqrt_sd = np.ma.sqrt(sd)
        return sqrt_sd, "C-AV"

def abs_diff_var(sequence, sequence_B, grid_size, edge_mask, to_mask):
    ad = rel_abs_temp_diff(sequence, sequence_B, edge_mask)[0]
    ad = np.ma.masked_where(to_mask > 0, ad)
    ad_sd = function_over_grid(ad, grid_size=grid_size, function="var", nan_fill=0, max_masked_prop=0.90, plot=False)
    ad_sd_sqrt = np.ma.sqrt(ad_sd)
    #show(ad_sd_sqrt)
    return ad_sd_sqrt, "C-ADV"


def function_over_grid(activation, grid_size, function, nan_fill, max_masked_prop=0.75, plot=False):
    '''Ignoring any masked values, calculate the function (e.g. mean or sd)
    of the given activation over a grid of the specified size.
    Any box which intersects less than 30% with the unmasked values is excluded,
    and filled with nan_fill value.'''

    n_v = int(np.ceil(activation.shape[0] /grid_size))
    n_h = int(np.ceil(activation.shape[1] /grid_size))

    result = np.ones_like(activation) * nan_fill
    for iv in range(0, n_v):
        for ih in range(0, n_h):
            masked_prop = 0

            if iv == n_v - 1:
                lower_bound = activation.shape[0]
            else:
                lower_bound = (iv + 1) * grid_size
            if ih == n_h - 1:
                rhs_bound = activation.shape[1]
            else:
                rhs_bound = (ih + 1) * grid_size

            box = activation[iv * grid_size: lower_bound, ih * grid_size: rhs_bound].copy()
            #show(box)
            if np.ma.isMaskedArray(activation):
                box_mask = activation[iv * grid_size: lower_bound, ih * grid_size: rhs_bound].mask.copy()
                box = np.ma.masked_where(box_mask, box)
                masked_prop = np.sum(box_mask) / (box.shape[0] * box.shape[1])
                #print(masked_prop)

            if masked_prop > max_masked_prop:
                value = nan_fill
            elif function == "mean":
                value = np.nanmean(box)
            elif function == "var":
                unmasked_pixels = box.compressed()
                value = np.std(unmasked_pixels)

            result[iv * grid_size: lower_bound, ih * grid_size: rhs_bound] = value

    if plot == True:
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(activation, cmap="gray")
        axs[1].imshow(result, cmap = "gray")
        plt.show()

    return result

def superpixels(c1, c2, c3, k, comp, plot=False):
    '''Paramter k indicates the number of approximately evenly sized superpixels to be initialised.
    Then comp indicaties the importance of compactness (spatially close, even sized pixels vs. colour distance)'''
    three_channel = np.stack([c1, c2, c3], axis=-1)
    segments_slic = slic(three_channel, n_segments=k, compactness=comp, sigma=1, start_label=1)

    if plot == True:
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(segments_slic)
        ax[0].set_title("Superpixels")
        ax[1].imshow(mark_boundaries(three_channel, segments_slic)[:,:,0], cmap="gray")
        ax[1].set_title('Boundaries')
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.show()

    return segments_slic

def ws(image, plot=False):
    c, l ,r = hist_max_thresh(image, plot=True)

    #TODO based on an example, I think this works better if seeds are small and separate,
    #and of integer values

    markers = np.where(image < l - 1, 1, 0)
    edges = filters.sobel(image)
    result = watershed(edges, markers=markers, compactness=1000)

    fig, axs = plt.subplots(ncols=3)
    axs[0].imshow(edges)
    axs[1].imshow(markers)
    axs[2].imshow(result)
    plt.show()

    return result

def dil(grayscale, r, mask_out, plot=False):
    fp = footprint_rectangle((r,r))
    gs = np.where(mask_out > 0, np.ma.min(grayscale), grayscale)
    result = dilation(gs, fp)

    if plot == True:
        joined = np.concatenate([gs, result], axis=1)
        u = np.nanmax(joined)
        l = np.nanmin(joined)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(gs, vmax=u, vmin=l)
        ax[0].set_title("Original")
        ax[1].imshow(result, vmax=u, vmin=l)
        ax[1].set_title('Transformed')
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.show()
    return result

def ero(grayscale, r, mask_out, plot=False):
    fp = footprint_rectangle((r,r))
    gs = np.where(mask_out > 0, np.ma.min(grayscale), grayscale)
    result = erosion(gs, fp)

    if plot == True:
        joined = np.concatenate([gs, result], axis=1)
        u = np.nanmax(joined)
        l = np.nanmin(joined)

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(gs, vmax=u, vmin=l)
        ax[0].set_title("Original")
        ax[1].imshow(result, vmax=u, vmin=l)
        ax[1].set_title('Transformed')
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.show()
    return result



