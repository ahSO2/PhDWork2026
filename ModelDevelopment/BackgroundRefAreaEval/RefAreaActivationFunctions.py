import cv2
import matplotlib.pyplot as plt
import numpy as np

from ScalingFunctions import min_max_scale, scale_by_percentile, hist_max_thresh

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
def mvmt_below_noise_gridmean(bandA, ts_bandA, flank_mask):
    '''Select pixels with movement below the noise level as potential background pixels.'''
    inv_scaled_diff = inv_diff_mvmt(bandA, ts_bandA)[0]
    inv_scaled_diff = np.ma.masked_where(flank_mask == 0, inv_scaled_diff)
    c, l, r = hist_max_thresh(inv_scaled_diff, plot=True)
    inv_diff_thresh = np.where(inv_scaled_diff > c, inv_scaled_diff, 0)

    #TODO take the mean over a grid

    return inv_diff_thresh, "C-IM_t"


