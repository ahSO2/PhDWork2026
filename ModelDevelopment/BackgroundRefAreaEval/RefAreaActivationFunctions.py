import cv2
import matplotlib.pyplot as plt
import numpy as np

from ScalingFunctions import min_max_scale, scale_by_percentile, hist_max_thresh, min_max_percentile

'''These functions output a numerical measure on the likely distribution
of plume or background in the image, rather than a binary mask.'''

def show(image, colormap="gray", title="None"):
    plt.imshow(image, cmap=colormap)
    if "None" in title:
        pass
    else:
        plt.title(title)
    plt.colorbar()
    plt.show()

def relative_brightness(bandA):
    brightness = min_max_scale(bandA)
    return brightness, "C-B"

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
    ratio = np.where(ratio < 1, ratio, 1)
    #show(ratio, colormap="YlGnBu_r")

    return ratio, "C-AA"

def inv_relative_absorbance(bandA, bandB, flank_mask):
    '''1 - relative absorbance, for thresholding for background.'''

    rel_abs = relative_absorbance(bandA, bandB, flank_mask)[0]
    inv = np.ones_like(rel_abs) - rel_abs

    return inv, "C-IAA"

def diff_mvmt(bandA, ts_bandA):
    '''Calculate the absolute value of the difference between two bandA frames.'''

    scaled_A, scaled_ts = scale_by_percentile(bandA, ts_bandA, 95)
    scaled_diff = np.abs(scaled_A.astype(np.float32) - scaled_ts.astype(np.float32))
    scaled_diff = cv2.blur(scaled_diff, (10, 10))
    scaled_diff = min_max_scale(scaled_diff)
    #show(scaled_diff)
    return scaled_diff, "C-M"

def inv_diff_mvmt(bandA, ts_bandA):
    scaled_diff = diff_mvmt(bandA, ts_bandA)[0]
    inv_scaled_diff = np.ones_like(scaled_diff) - scaled_diff
    return inv_scaled_diff, "C-IM"
def mvmt_above_noise(bandA, ts_bandA, flank_mask):
    '''Select pixels with movement below the noise level as potential background pixels.'''
    scaled_diff = diff_mvmt(bandA, ts_bandA)[0]
    scaled_diff = np.ma.masked_where(flank_mask == 0, scaled_diff)
    c, l, r = hist_max_thresh(scaled_diff, plot=False)
    diff_thresh = np.where(scaled_diff > r, 1, 0)
    return diff_thresh, "C-M_t"

def rel_abs_temp_diff(bandA, bandB, ts_bandA, ts_bandB):
    bandB_zero = np.where(bandB == 0, 5, 0)
    edge_mask = cv2.blur(bandB_zero, (11, 11))
    edge_mask[:, 0:6] = 1
    edge_mask[:, -5:] = 1
    edge_mask[0:6, :] = 1
    edge_mask[-5:, :] = 1

    og_ratio = log_ratio_AA(bandA, bandB, edge_mask)
    ts_ratio = log_ratio_AA(ts_bandA, ts_bandB, edge_mask)
    #Scaling so that the 95th percentile range of each image covers range [1, 0]
    #(Switched from scaling one percentile value to match the other as sometimes this results in scale factor zero if one percentile is zero).
    scaled_og = min_max_percentile(og_ratio, 95)
    scaled_ts = min_max_percentile(ts_ratio, 95)

    AA_diff = np.abs(scaled_og - scaled_ts)
    AA_diff = np.where(AA_diff > 1, 1, AA_diff)
    AA_diff = np.ma.masked_where(edge_mask > 0, AA_diff)
    return AA_diff, "C-AM"

def function_over_grid(activation, grid_size, function, nan_fill, min_box_prop=0.75, plot=False):
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

            if masked_prop > min_box_prop:
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






