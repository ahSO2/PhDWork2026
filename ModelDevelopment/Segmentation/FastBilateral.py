import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

#Informed by example code from:
#https://people.csail.mit.edu/sparis/bf/#code

sample_img = np.load("C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/26 - Cross Bilateral Filter/bandA_Reventador_2024-08-11T161750_fltrA_1ag_1499994ss_Plume.png.npy")
sample_activation = np.load("C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/26 - Cross Bilateral Filter/activation_Reventador_2024-08-11T161750_fltrA_1ag_1499994ss_Plume.png.npy")

def show(image):
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

def gaussian_3D(s_s, s_r):
    '''Return a 3D gaussian kernel with shape (2 * s_s + 1, 2 * s_s + 1, 2 * s_r + 1).'''

    s_s = int(np.round(s_s, 0))
    s_r = int(np.round(s_r, 0))


    #Create 3D array of x and y indexes in the filter
    xs = np.arange(-1 * s_s, s_s + 1)
    X, Y = np.meshgrid(xs, xs)
    X = np.stack([X] * (2 * s_r + 1), axis=-1)
    Y = np.stack([Y] * (2 * s_r + 1), axis=-1)
    #Create 3D array of index in phi dimension
    rs = np.arange(-1 * s_r, s_r + 1)
    R = np.stack([rs] * (2 * s_s + 1), axis=0)
    R = np.stack([R] * (2 * s_s + 1), axis=0)

    #Calculate the filter values
    term1 = np.divide(-1 * (np.square(X) + np.square(Y)), 2*s_s*s_s)
    term2 = np.divide(-1 * np.square(R), 2*s_r*s_r)
    gauss = np.exp(term1 + term2)

    print("Smoothing in the (x, y, range) space with a 3D gaussian of dimension:" + str(gauss.shape))
    print("This is the dimension in the downsampled image.")

    return gauss





def cross_bilateral_filter_fast(image, ref_img, sigma_s, sigma_r, sa_s, sa_r):
    '''Apply a cross bilateral filter to smooth the image, while
    preserving edges in the refernce image.'''

    #Step 1 - Initialise downsampled array
    range_max = np.max(ref_img)
    range_min = np.min(ref_img)
    dims = (int(np.ceil(image.shape[0]/sa_s)), int(np.ceil(image.shape[1]/sa_s)), int(np.ceil((range_max - range_min)/sa_r)))
    wd_id = np.zeros(shape=dims)
    wd = np.zeros(shape=dims)

    #Step 2 - Compute minimum intensity in the range
    #Done above

    #Step 3 - For each pixel, compute the downsampled versions of
    #wi and w (i.e. fill out the arrays created in Step 1).
    x_vals = np.arange(image.shape[1])
    y_vals = np.arange(image.shape[0])
    Xd = np.floor(x_vals/sa_s).astype(np.uint8) #For each pixel, the corresponding downsampled x-coord
    Yd = np.floor(y_vals/sa_s).astype(np.uint8)
    Rd = np.floor((ref_img - range_min)/sa_r).astype(np.uint8)

    #For every pixel, add its I(x,y) and Wq = 1 values to the "bucket" corresponding to its downsampled coordinates.
    #TODO can probably do this more efficiently with array reshaping
    for x in range(0, image.shape[1]):
        for y in range(0, image.shape[0]):
            xd = Xd[x]
            yd = Yd[y]
            rd = Rd[y, x]
            wd_id[yd, xd, rd] += image[y, x]
            wd[yd, xd, rd] += 1

    #show(wd_id[:,:,30])

    #Step 4 - Define the gaussian product, and convolve.
    g = gaussian_3D(sigma_s/sa_s, sigma_r/sa_r)
    wbd_ibd = ndimage.convolve(wd_id, g, mode="reflect")
    wbd = ndimage.convolve(wd, g, mode="reflect")

    #Step 5 - Upsample and normalise to get final result.
    #Part a) for each pixel, interpolate in 3D to upsample the arrays
    #defined in step 4.

    #The interpolator is defined on the range of downsampled coordinates
    Yd_nd = np.arange(wd.shape[0]) #Get downsampled array indices with no duplicate values
    Xd_nd = np.arange(wd.shape[1])
    Rd_nd = np.arange(wd.shape[2])
    interpolator_1 = RegularGridInterpolator((Yd_nd, Xd_nd, Rd_nd), wbd_ibd)
    interpolator_2 = RegularGridInterpolator((Yd_nd, Xd_nd, Rd_nd), wbd)

    #For each pixel in the image:
    x_input = x_vals/sa_s
    y_input = y_vals/sa_s
    r_input = (ref_img - range_min)/sa_r
    Xe, Ye = np.meshgrid(x_input, y_input)
    #Replace values that are outwith the range of values the interpolator
    #has been fitted on.
    Xe = np.where(Xe >= np.max(Xd_nd) - 1, np.max(Xd_nd) - 1, Xe)
    Ye = np.where(Ye >= np.max(Yd_nd) -1, np.max(Yd_nd) - 1, Ye)
    r_input = np.where(r_input >= np.max(Rd_nd) -1, np.max(Rd_nd) -1, r_input)

    #Get an array of shape (n_points, 3) to input the coordinates into the inteprolators
    points = np.stack([Ye, Xe, r_input], axis=-1)
    points = np.reshape(points, (image.shape[0] * image.shape[1], 3))

    wb_ib = interpolator_1(points)
    wb = interpolator_2(points)

    wb_ib = np.reshape(wb_ib, (image.shape[0], image.shape[1]))
    #show(wb_ib)
    wb = np.reshape(wb, (image.shape[0], image.shape[1]))
    #show(wb)

    #Part b) Divide the arrays to get the normalised result.
    filtered_result = np.divide(wb_ib, wb)
    #show(wb_ib)
    #show(wb)
    show(filtered_result)

sample_activation = sample_activation * 100

ss_img = np.zeros(shape=(486, 648))
ss_img[410:421,510:521] = 10
ss_edge = np.zeros(shape=(486, 648))
ss_edge[400:431,500:531] = 5
show(ss_img)
show(ss_edge)
cross_bilateral_filter_fast(ss_img, ss_edge, sigma_s=5, sigma_r=4, sa_s=2, sa_r=2)