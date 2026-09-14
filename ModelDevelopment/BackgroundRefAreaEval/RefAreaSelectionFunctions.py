import cv2
import matplotlib.pyplot as plt
import numpy as np

def delledonne_max_bandA(bandA, bandB, volcano_dictionary, plot=False):
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
        img_to_show = cv2.rectangle(bandA_copy, (tl[1], tl[0]), (br[1], br[0]), color=np.min(bandA_copy))
        # region_to_show = cv2.circle(sky_region, center=(column_in_region, row_in_region), radius=circle_radius, color=int(np.min(sky_region)))
        img_to_show = np.where(circle_mask == 1, circle_mean_value, img_to_show)
        img_to_show = cv2.circle(img_to_show, center=(max_index[1], max_index[0]), radius=circle_radius, color=int(np.min(bandA)))
        plt.imshow(img_to_show, cmap="gray")
        plt.show()


    return circle_mask #Return the selected circle as the reference area

def delledonne_min_ratio(bandA, bandB, volcano_dictionary, plot=False):
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

        sky_region_to_show = sky_region.copy()
        sky_region_to_show = cv2.circle(sky_region_to_show, center=(column_in_region, row_in_region), radius=circle_radius, color=100)
        sky_region_to_show = np.ma.masked_where(sky_region_mask, sky_region_to_show)
        plt.imshow(sky_region_to_show, vmin=np.ma.min(sky_region), vmax=np.ma.max(sky_region))
        plt.colorbar()
        plt.show()

        img_to_show = ratio.copy()
        img_to_show = cv2.rectangle(img_to_show, (tl[1], tl[0]), (br[1], br[0]), color=100)
        img_to_show = np.where(circle_mask == 1, circle_mean_value, img_to_show)
        img_to_show = cv2.circle(img_to_show, center=(min_index[1], min_index[0]), radius=circle_radius, color=100)
        img_to_show = np.ma.masked_where(ratio.mask, img_to_show)
        plt.imshow(img_to_show, cmap="YlGnBu_r", vmin=np.ma.min(ratio), vmax=np.ma.max(ratio))
        plt.colorbar()
        plt.show()
    return circle_mask

def osorio_threshold_and_connect(bandA, bandB, volcano_dictionary, plot=False):
    '''Threshold on the ratio of bandA/bandB, then select largest connected component
    of pixels to represent the plume.'''

    #TODO Calculate bandA/bandB

    #TODO Threshold this ratio (how to set value?)

    #TODO Select pixels with 8-connectivity

    #TODO Select largest connected component as the plume
    #TODO Everything else is reference area

def kern_low_texture_and_ratio(bandA, bandB, volcano_dictionary, plot=False):
    '''Filter for an area of the image which is "smooth" and has
    a low -ln(bandA/bandB) value.'''

    #TODO Maybe calculate image entropy

    #TODO Then select a point with relatively low entropy and low optical depth


