import matplotlib.pyplot as plt
import numpy as np

def min_max_scale(img):
    '''Scale image to range [0, 1], considering only unmasked pixels.'''
    img_min = np.ma.min(img)
    img_max = np.ma.max(img)
    scaled_img = (img - img_min)/(img_max-img_min)
    return scaled_img

def scale_values_to_01(lower, upper, img):
    scaled_img = (img - lower) / (upper - lower)
    return scaled_img

def min_max_percentile(img, p):
    '''Scale such that the 0 to p-th percentile values are in range [0-1].
    Considering only unmasked pixels.'''
    if np.ma.is_masked(img):
        img_values = img.compressed()
    else:
        img_values = img.flatten()
    perc = np.percentile(img_values, p)
    img_min = np.min(img_values)
    scaled_img = (img - img_min) / (perc - img_min)
    return scaled_img

def scale_by_percentile(img1, img2, p):
    '''Scale either image using the 95th percentile values.'''
    if np.ma.is_masked(img1):
        img1_values = img1.compressed()
        img2_values = img2.compressed()
    else:
        img1_values = img1.flatten()
        img2_values = img2.flatten()
    p95_1 = np.percentile(img1_values, p)
    p95_2 = np.percentile(img2_values, p)
    #print(p95_1)
    #print(p95_2)

    if p95_1 > p95_2:
        scale_factor = p95_1/p95_2
        scaled_2 = img2.astype(np.float32) * scale_factor
        return img1.astype(np.float32), scaled_2
    else:
        scale_factor = p95_2/p95_1
        scaled_1 = img1.astype(np.float32) * scale_factor
        return scaled_1, img2.astype(np.float32)

def hist_max_thresh(image, plot=False):
    '''Return a threshold for the image pixels based on the histogram bin
    containing the most common values. Return bin center and edge values.'''

    n_bins = 10
    if np.ma.is_masked(image):
        values = image.compressed()
    else:
        values = image.flatten()

    counts, bins = np.histogram(values, n_bins, [np.nanmin(values), np.nanmax(values)])
    bin_length = np.abs(np.nanmax(values) - np.nanmin(values)) / n_bins

    max_bin_index = np.argmax(counts) #Index of tallest bin (starting from 0)
    rhs_boundary = bins[max_bin_index + 1]
    center = rhs_boundary - bin_length/2
    lhs_boundary = rhs_boundary - bin_length

    if plot == True:
        bin_centers = np.linspace(bin_length / 2, bin_length * (n_bins - 1 + 0.5), n_bins)
        plt.stairs(counts, bins)
        plt.scatter(bin_centers, np.ones_like(bin_centers))
        plt.axvline(x=rhs_boundary)
        plt.axvline(x=center)
        plt.axvline(x=lhs_boundary)
        plt.show()

    return center, lhs_boundary, rhs_boundary