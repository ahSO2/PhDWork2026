import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyplis
import scipy
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def min_max_scale(img):
    '''Scale image to range [0, 1].'''
    img_min = np.min(img)
    img_max = np.max(img)
    scaled_img = (img - img_min)/(img_max-img_min)
    return scaled_img
def calc_bin_thresh(image, plot=False):
    '''Return a threshold for the image pixels based on the histogram bin
    containing the most common values.'''

    n_bins = 10
    counts, bins = np.histogram(image.compressed(), n_bins, [np.nanmin(image), np.nanmax(image)])
    bin_length = np.abs(np.nanmax(image) - np.nanmin(image)) / n_bins

    max_bin_index = np.argmax(counts) #Index of tallest bin (starting from 0)
    rhs_boundary = bins[max_bin_index + 1] - bin_length/2

    if plot == True:
        bin_centers = np.linspace(bin_length / 2, bin_length * (n_bins - 1 + 0.5), n_bins)
        plt.stairs(counts, bins)
        plt.scatter(bin_centers, np.ones_like(bin_centers))
        plt.axvline(x=rhs_boundary)
        plt.show()

    return rhs_boundary

def thresholding(bandA, bandB, volcano_dictionary, plot=False, next_frame=None):
    '''A function to allow easy implementation of thresholding on channels
    commonly used in literature either manually or with Otsu's method.
    
    Channel options: bandA, difference between consecutive frames, difference taken
    with a reference image, ratio of bandA to B, or log ratio.'''

    channel = bandA

    threshold = 300

    ref_areas = np.where(channel > threshold, 1, 0)

    img_to_show = np.where(ref_areas == 1, bandA, 0)
    plt.imshow(img_to_show, cmap="gray")
    plt.colorbar()
    plt.show()

    return ref_areas


def delledonne_max_bandA(bandA, plot=False):
    '''Select the maximum pixel value in the band A image within a rectangular
    region at the top of the image. Select a small circular region around this
    point from which to take the average band A value. '''

    tl = (30, 30)
    br = (130, 618)
    sky_region = bandA[tl[0]:br[0] + 1, tl[1]:br[1]+1]
    circle_radius = 20

    #Smooth the sky region so that noise doesn't affect the maximum
    sky_region = cv2.blur(sky_region, (5,5))

    max_flattened = np.argmax(sky_region)
    row_in_region = int(np.floor(max_flattened/sky_region.shape[1]))
    column_in_region = int(max_flattened % sky_region.shape[1])
    max_index = (tl[0] + row_in_region, tl[1] + column_in_region) #Center in numpy coordinates

    circle_mask = cv2.circle(np.zeros_like(bandA), center=(max_index[1], max_index[0]), radius=circle_radius, color=1, thickness=-1)
    circle_pixel_vals = np.where(circle_mask == 1, bandA, 0)
    circle_mean_value = np.sum(circle_pixel_vals)/np.sum(circle_mask)

    if plot == True:
        bandA_copy = bandA.copy()
        img_to_show = cv2.rectangle(bandA_copy, (tl[1], tl[0]), (br[1], br[0]), color=int(np.min(bandA_copy)))
        # region_to_show = cv2.circle(sky_region, center=(column_in_region, row_in_region), radius=circle_radius, color=int(np.min(sky_region)))
        img_to_show = np.where(circle_mask == 1, circle_mean_value, img_to_show)
        img_to_show = cv2.circle(img_to_show, center=(max_index[1], max_index[0]), radius=circle_radius, color=int(np.min(bandA)))
        plt.imshow(img_to_show, cmap="gray")
        plt.show()


    return circle_mask, "D-A" #Return the selected circle as the reference area

def delledonne_min_ratio(bandA, bandB, plot=False):
    '''Select the minimum of -ln(bandA/bandB) value within a rectangular
        region at the top of the image. Select a small circular region around this
        point from which to take the average ratio value.

        Any overlap of the selected circle with the areas where bandB=0 is removed
        from the selected region.'''

    tl = (30, 30)
    br = (130, 618)
    circle_radius = 20

    #First mask out the edges which cause artifical high absorbance values
    bandA_copy = bandA.copy()
    bandA_copy = np.ma.masked_where(bandB==0, bandA_copy)

    #Then calculate the absorbance (without accounting for backgrounds)
    ratio = np.ma.divide(bandA_copy.astype(np.float32), bandB.astype(np.float32))
    ratio = -1 * np.ma.log(ratio)

    #Blur slightly to avoid small imperfections impacting the min value
    #Again mask out edges which are introduced by the blur
    ratio = cv2.blur(ratio, (5, 5))
    bandB_zero = np.where(bandB==0, 3, 0)
    edge_mask = cv2.blur(bandB_zero, (11, 11))
    edge_mask[:, 0] = 1
    edge_mask[:, -1] = 1
    edge_mask[0, :] = 1
    edge_mask[-1, :] = 1
    ratio = np.ma.masked_where(edge_mask>0, ratio)

    #Select the sky region, copying over the masked areas too
    sky_region = ratio[tl[0]:br[0] + 1, tl[1]:br[1] + 1]
    sky_region_mask = ratio.mask[tl[0]:br[0] + 1, tl[1]:br[1] + 1]
    sky_region = np.ma.masked_where(sky_region_mask, sky_region)

    #Find the minimum valued pixel
    min_flattened = np.ma.argmin(sky_region)
    row_in_region = int(np.floor(min_flattened / sky_region.shape[1]))
    column_in_region = int(min_flattened % sky_region.shape[1])
    min_index = (tl[0] + row_in_region, tl[1] + column_in_region)  # Center in numpy coordinates


    circle_mask = cv2.circle(np.zeros_like(bandA), center=(min_index[1], min_index[0]), radius=circle_radius, color=1, thickness=-1)
    circle_mask = np.where(ratio.mask, 0, circle_mask)
    circle_pixel_vals = np.where(circle_mask == 1, ratio, 0)
    circle_mean_value = np.sum(circle_pixel_vals) / np.sum(circle_mask)

    if plot == True:

        #sky_region_to_show = sky_region.copy()
        #sky_region_to_show = cv2.circle(sky_region_to_show, center=(column_in_region, row_in_region), radius=circle_radius, color=100)
        #sky_region_to_show = np.ma.masked_where(sky_region_mask, sky_region_to_show)
        #plt.imshow(sky_region_to_show, vmin=np.ma.min(sky_region), vmax=np.ma.max(sky_region))
        #plt.colorbar()
        #plt.show()

        img_to_show = ratio.copy()
        img_to_show = cv2.rectangle(img_to_show, (tl[1], tl[0]), (br[1], br[0]), color=100)
        img_to_show = np.where(circle_mask == 1, circle_mean_value, img_to_show)
        img_to_show = cv2.circle(img_to_show, center=(min_index[1], min_index[0]), radius=circle_radius, color=100)
        img_to_show = np.ma.masked_where(ratio.mask, img_to_show)
        plt.imshow(img_to_show, cmap="YlGnBu_r", vmin=np.ma.min(ratio), vmax=np.ma.max(ratio))
        plt.colorbar()
        plt.show()
    return circle_mask, "D-R"

def osorio_threshold_and_connect(bandA, bandB, volcano_dictionary, plot=False):
    '''Threshold on the ratio of bandA/bandB, then select largest connected component
    of pixels to represent the plume.'''

    #Calculate bandA/bandB, masking out edges where bandB is zero or has
    #weird values introduced by the registration transformation
    bandA_copy = np.ma.masked_where(bandB == 0, bandA.copy())
    ratio = np.ma.divide(bandA_copy.astype(np.float32), bandB.astype(np.float32))
    ratio = np.ma.masked_where(bandA_copy.mask, bandA_copy)

    #TODO Here I have applied smoothing to avoid getting lots of small
    #connected areas - not sure if this is applied in the original paper?
    ratio = cv2.blur(ratio, (5, 5))
    bandB_zero = np.where(bandB == 0, 3, 0)
    edge_mask = cv2.blur(bandB_zero, (11, 11))
    edge_mask[:, 0] = 1
    edge_mask[:, -1] = 1
    edge_mask[0, :] = 1
    edge_mask[-1, :] = 1
    ratio = np.ma.masked_where(edge_mask > 0, ratio)
    plt.imshow(ratio, cmap="YlGnBu_r")
    plt.colorbar()
    plt.show()


    #TODO Threshold this ratio (how to set value?)
    threshold = np.mean(bandA) #TODO Dummy choice for now
    binary = np.where(ratio > threshold, 1, 0).astype(np.uint8)
    plt.imshow(binary, cmap='gray')
    plt.show()

    #TODO Select pixels with 8-connectivity
    n_labels, labels_im = cv2.connectedComponents(binary)
    print(n_labels)
    plt.imshow(labels_im)
    plt.colorbar()
    plt.show()

    #TODO Select largest connected component as the plume
    #TODO Everything else is reference area
    return None

def kern_low_texture_and_ratio(bandA, bandB, flank_mask, plot=False):
    '''Filter for an area of the image which is smooth (low variance) and has
    a low -ln(bandA/bandB) value.

    The division into boxes used here is based on the assumption that the pixel
    dimensions are such that that np.floor(dim/10) is divisible by 4.'''

    #Calculate the absorbance ignoring backgrounds
    #First mask out the edges which cause artifical high absorbance values,
    #Masking out the flank at the same time
    bandA_copy = bandA.copy()
    edge_mask = np.where(bandB==0, 0, flank_mask) * 5
    edge_mask = cv2.blur(edge_mask, (5, 5))
    edge_mask[:,0] = 0
    edge_mask[:,-1] = 0
    edge_mask[0,:] = 0
    edge_mask[-1,:] = 0
    bandA_copy = np.ma.masked_where(edge_mask < 5, bandA_copy)

    #Then calculate the absorbance (without accounting for backgrounds)
    ratio = np.ma.divide(bandA_copy.astype(np.float32), bandB.astype(np.float32))
    ratio = -1 * np.ma.log(ratio)

    #Define the length of each box (10% of the image height and width)
    l_h = int(np.floor(bandA.shape[1]/10))
    l_v = int(np.floor(bandA.shape[0]/10))

    means = np.empty(shape=(37,37))
    vars = np.empty(shape=(37,37))
    for i in range(0, 37): #Horizontal
        for j in range(0, 37): #Vertical
            s_h = int(i * (l_h/4)) #Horizontal index of the start of the box
            s_v = int(j * (l_v/4)) #Vertical index of the start of the box
            box = ratio[s_v:s_v + l_v, s_h:s_h + l_h]
            mask = ratio.mask[s_v:s_v + l_v, s_h:s_h + l_h]
            box = np.ma.masked_where(mask, box)

            if np.array_equal(mask, np.ones_like(mask)):
                box_mean = np.nan
                var = np.nan
            else:
                box_mean = np.ma.mean(box)
                unmasked_pixels = box.compressed()
                var = np.var(unmasked_pixels)

            means[j, i] = box_mean
            vars[j, i] = var

    #Select threshold on variance:
    vars = np.ma.masked_invalid(vars)
    thresh = calc_bin_thresh(vars, plot=True)

    thresh_vars = np.ma.where(vars < thresh, vars, np.nan)
    means_thresh = np.ma.where(vars < thresh, means, np.nan)

    #Select the remaining rectangle with the lowest mean absorbance
    min_index = np.nanargmin(means_thresh)
    min_x = int(min_index % means_thresh.shape[1])
    min_y = int(np.floor(min_index/means_thresh.shape[1]))

    s_h = int(min_x * (l_h / 4))  # Horizontal index of the start of the box
    s_v = int(min_y * (l_v / 4))

    box_pixels = np.zeros_like(bandA)
    box_pixels[s_v:s_v + l_v, s_h:s_h + l_h] = 1
    box_pixels = np.where(flank_mask==0, 0, box_pixels)

    illustration = np.where(box_pixels==0, bandA, np.min(bandA))
    illustration = np.ma.masked_where(bandB==0, illustration)

    plt.imshow(means)
    plt.colorbar()
    plt.show()
    plt.imshow(ratio)
    plt.colorbar()
    plt.show()

    #Plot the image, the AA, the variance, the thresholded variance
    if plot == True:
        fig, axs = plt.subplots(ncols=2, nrows=3)
        axs[0,0].imshow(bandA, cmap="gray")
        axs[0, 0].set_title("310nm", fontsize=10)
        axs[0, 1].imshow(ratio, cmap="YlGnBu_r")
        axs[0, 1].set_title("AA", fontsize=10)
        axs[1, 0].imshow(vars, vmin=np.nanmin(vars), vmax=np.nanmax(vars))
        axs[1, 0].set_title("Variance (Grid boxes)", fontsize=10)
        axs[1, 1].imshow(thresh_vars, vmin=np.nanmin(vars), vmax=np.nanmax(vars))
        axs[1, 1].set_title("Thresholded Variance", fontsize=10)
        axs[2, 0].imshow(means_thresh, cmap="YlGnBu_r")
        axs[2, 0].set_title("Thresholded Grid-box AA", fontsize=10)
        axs[2, 1].imshow(illustration, cmap="gray")
        axs[2, 1].set_title("Selected rectangle", fontsize=10)
        plt.subplots_adjust(wspace=0, hspace=0.2)
        for row in range(0, 2):
            for col in range(0, 2):
                axs[row, col].set_xticklabels([])
                axs[row, col].set_yticklabels([])
        plt.show()

    return box_pixels, "K-SR"

def pyplis_rectangles_and_lines(bandA, plot=True, output="both"):
    '''Based on the image intensity, select three reference rectangles and
    two reference lines, and return as two separate masks.'''
    ref_params = pyplis.plumebackground.find_sky_reference_areas(bandA)

    if plot == True:
        fig, axs = plt.subplots()
        pyplis.plumebackground.plot_sky_reference_areas(bandA, ref_params, ax=axs)
        plt.show()

    lines_mask = np.zeros_like(bandA)
    lines_mask[ref_params['xgrad_line_rownum'], ref_params['xgrad_line_startcol']:ref_params['xgrad_line_stopcol']] = 1
    lines_mask[ref_params['ygrad_line_startrow']:ref_params['ygrad_line_stoprow'], ref_params['ygrad_line_colnum']] = 1

    rectangles_mask = np.zeros_like(bandA)
    rectangles_mask[ref_params['scale_rect'][1]:ref_params['scale_rect'][3], ref_params['scale_rect'][0]:ref_params['scale_rect'][2]] = 1
    rectangles_mask[ref_params['xgrad_rect'][1]:ref_params['xgrad_rect'][3],ref_params['xgrad_rect'][0]:ref_params['xgrad_rect'][2]] = 1
    rectangles_mask[ref_params['ygrad_rect'][1]:ref_params['ygrad_rect'][3],ref_params['ygrad_rect'][0]:ref_params['ygrad_rect'][2]] = 1

    if output == "lines":
        return lines_mask, "G-L"
    elif output == "rectangles":
        return rectangles_mask, "G-R"
    else:
        return np.where(lines_mask + rectangles_mask > 0, 1, 0), "G-RL"
def pyplis_background_mask(bandA, next_img, plot=True):
    '''Determine background pixels by thresholding on bandA intensity (threshold based
        on the reference areas) and then exclusion of pixels which are moving (based on motion est alg).'''
    bandA_obj = pyplis.image.Img(bandA[:-2, :])
    next_img_obj = pyplis.image.Img(next_img[:-2, :])
    bg_pixels = pyplis.plumebackground.find_sky_background(bandA_obj, next_img_obj, bgmodel_settings_dict=None,lower_thresh=None, apply_movement_search=True)
    #TODO Note I've removed the bottom two rows of each image because I think the fact that the
    #heights were not divisible by 4 was causing an mismatch in the dimension of the movement
    #mask returned (which is calulated using pyramids which I think downsample in multiples of 4)
    #I replace these rows with zeros, as the flank is masked out anyway
    bg_pixels = np.concatenate([bg_pixels, np.zeros((2, bg_pixels.shape[1]))], axis=0)

    if plot == True:
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(bandA, cmap="gray")
        axs[1].imshow(np.where(bg_pixels==1, bandA, np.min(bandA)), cmap="gray")
        plt.show()
    return bg_pixels, "G-AT"

def polynomial_fit(masked_image, degree=2, plot=False):
    '''Fit a 2D polynomial to the unmasked pixels of the given image,
    and return the value of the fitted polynomial for every pixel.'''
    #An object which maps an array of x,y coords to [intercept, x, y, xy, x^2, y^2 ...] dependent on specified degree
    poly_features_map = PolynomialFeatures(degree=degree)

    #Create a (n_pixels x 2) array of the x and y coords of each unmasked point in the image
    x_range = np.arange(0, masked_image.shape[1])
    y_range = np.arange(0, masked_image.shape[0])
    X, Y = np.meshgrid(x_range, y_range)

    masked_X = np.ma.masked_where(masked_image.mask, X)
    masked_Y = np.ma.masked_where(masked_image.mask, Y)
    unmasked_coords = np.stack([masked_X.compressed(), masked_Y.compressed()], axis=1)
    #Transform the coordinates of the unmasked points to the polynomial multiples required to input into the linear fit
    polynomial_inputs = poly_features_map.fit_transform(unmasked_coords)

    #Create an array holding coordinates of all points in the image, and map to the required polynomial imput values
    all_img_coords = np.stack([X.flatten(), Y.flatten()], axis=-1)
    all_img_coords_polynomial = poly_features_map.fit_transform(all_img_coords)

    regression_model = linear_model.LinearRegression()

    regression_model.fit(polynomial_inputs, masked_image.compressed())

    fitted_image = regression_model.predict(all_img_coords_polynomial)

    #Now map back to the original image dimension
    fitted_image = fitted_image.reshape((masked_image.shape[0], masked_image.shape[1]))

    if plot == True:
        fig, axs = plt.subplots(ncols=2)
        left = axs[0].imshow(masked_image, cmap="gray")
        axs[0].set_title("Original")
        right = axs[1].imshow(fitted_image, cmap="gray", vmin=np.ma.min(masked_image), vmax=np.ma.max(masked_image))
        axs[1].set_title("Fitted Estimate")
        fig.colorbar(right, ax=axs[0:2], location="right", shrink=0.7, pad=0.15)
        plt.show()

    return fitted_image


def smekens_repeated_fitting(bandA, flank_mask):
    '''Identify sky reference areas by repeatedly fitting a 2nd degree 2D polynomial to the
    sky pixels, and excluding any pixels which are not well represented. '''

    sky_mask = flank_mask.copy() #Mask indicating pixels thought to be clear sky with 1s

    #I have omitted the edge mask used by Smekens et al. because we have
    #clear-corrected the images (and I think the placement of the lens/filter
    # in the PiCam camera minimises the effects at the edges).

    repeat = True
    while repeat == True:
        #Fit a 2D polynomial to the sky pixels
        masked_bandA = np.ma.masked_where(sky_mask==0, bandA)
        polyfit_sky = polynomial_fit(masked_bandA, degree=2, plot=False)
        #Calculate the transmittance image
        transmittance = np.ma.divide(masked_bandA, polyfit_sky)

        #Exclude pixels with transmittance outwith (0.97, 1.03) from the
        #identified sky reference area mask
        prev_sky_mask = sky_mask.copy()
        sky_mask = np.where(transmittance >= 1.03, 0, sky_mask)
        sky_mask = np.where(transmittance <= 0.97, 0, sky_mask)

        if np.array_equal(sky_mask, prev_sky_mask): #If the sky mask hasn't changed (i.e all pixels are approximated within 3%)
            repeat = False
            plt.imshow(sky_mask)
            plt.title("Final sky ref areas")
            plt.show()

    return sky_mask

