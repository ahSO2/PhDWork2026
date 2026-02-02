#Read in a set of samples and associated velocity field
#Interpolate the second timestep image, using the first
#with the given velocity field
#Evaluate the interpolation error


import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show(img):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()
def plot_optical_flow_Farneback(image, flow_components, n, save_loc=None):
    #Need array of x-coords, y-coords, then dx and dy
    #Downsample to plot every n-th coord point
    x_pixels = np.arange(0,648,n)
    y_pixels = np.arange(0,486,n)
    X, Y = np.meshgrid(x_pixels, y_pixels)

    #Downsample every nth flow vector for plotting
    flow_dx = flow_components[0::n,0::n,0]
    #flow_dx = flow_components[:,:,0]
    #flow_dy = flow_components[:,:,1]
    flow_dy = flow_components[0::n,0::n,1]

    plt.quiver(X, Y, flow_dx, flow_dy, color='g', scale_units = 'xy', scale=1, angles='xy')
    plt.gca().invert_yaxis()
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    if save_loc != None:
        plt.savefig(save_loc)
    plt.show()
    plt.close()

def find_dest(x, y, flow_x, flow_y):
    return (y+ flow_y, x + flow_x)

fd_vect = np.vectorize(find_dest)

def billinear_interpolate(values_array, coord):

    #Pad the array with copies of the edge rows
    padded_arr = np.pad(values_array, pad_width=1, mode="edge")

    padded_coord = (coord[0] + 1, coord[1] + 1)
    #Round the coordinate to nearest integer
    #Coord is given in form (y, x)
    int_coord = (int(np.round(padded_coord[0], 0)), int(np.round(padded_coord[1], 0)))

    x = int_coord[1]
    y = int_coord[0]

    x1 = int_coord[1] - 1
    x2 = int_coord[1]
    y1 = int_coord[0] - 1
    y2 = int_coord[0]

    C11 = padded_arr[y1, x1]
    C21 = padded_arr[y2, x1]
    C12 = padded_arr[y1, x2]
    C22 = padded_arr[y2, x2]

    #Interpolate in x-driection for y=y1
    R1 = C11*(x2 + 1 - padded_coord[1])/(x2 + 1 - x1) + C12*(padded_coord[1] - x1)/(x2 + 1 - x1)
    #Interpolate in x-direction for y=y2
    R2 = C21*(x2 + 1 - padded_coord[1])/(x2 + 1 - x1) + C22*(padded_coord[1] - x1)/(x2 + 1 - x1)

    V = R1*(y2 + 1 - padded_coord[0])/(y2 + 1 - y1) + R2*(padded_coord[0] - y1)/(y2 + 1 - y1)

    return V

def grid_interpolation(values_array, coord):
    # Pad the array with copies of the edge rows
    padded_arr = np.pad(values_array, pad_width=1, mode="edge")
    #print(coord)
    padded_coord = (coord[0] + 1, coord[1] + 1)

    # Round the coordinate to nearest integer
    # Coord is given in form (y, x)
    int_coord = (int(np.round(padded_coord[0], 0)), int(np.round(padded_coord[1], 0)))

    relevant_vals = padded_arr[int_coord[0] - 1: int_coord[0] + 2, int_coord[1] - 1: int_coord[1] + 2]

    x1 = int_coord[1] - 1
    x2 = int_coord[1]
    x3 = int_coord[1] + 1
    y1 = int_coord[0] - 1
    y2 = int_coord[0]
    y3 = int_coord[0] + 1

    C11 = (y1, x1)
    C21 = (y2, x1)
    C31 = (y3, x1)
    C12 = (y1, x2)
    C22 = (y2, x2)
    C32 = (y3, x2)
    C13 = (y1, x3)
    C23 = (y2, x3)
    C33 = (y3, x3)

    d11 = math.dist(C11, padded_coord)
    d12 = math.dist(C12, padded_coord)
    d13 = math.dist(C13, padded_coord)
    d21 = math.dist(C21, padded_coord)
    d22 = math.dist(C22, padded_coord)
    d23 = math.dist(C23, padded_coord)
    d31 = math.dist(C31, padded_coord)
    d32 = math.dist(C32, padded_coord)
    d33 = math.dist(C33, padded_coord)

    dists_mat = np.array([[d11, d12, d13],
                          [d21, d22, d23],
                          [d31, d32, d33]])
    #print(dists_mat)
    #Matrix of scale factor multipliers for the pixel values
    #Each pixel contributes based on the relative size of its center
    #distance from the point
    scale_mat = dists_mat / np.sum(dists_mat)
    scale_mat = np.ones_like(scale_mat) - scale_mat
    scale_mat = scale_mat / np.sum(scale_mat)

    contributions = np.multiply(relevant_vals, scale_mat)

    return np.sum(contributions)

def warp(original, next, flow_field):
    '''Warp the second "next" image back to the original,
    using the optical flow between the pair as the
    inverse transform for this mapping. This allows use
    of backward mapping, giving a better warp than forward.'''

    #original = original[0:450,0:550]
    #next = next[0:450, 0:550]
    #flow_field = flow_field[0:450, 0:550, :]

    #Create empty array to store warped frame
    warped = np.empty_like(original)

    x_pixels = np.arange(0, warped.shape[1])
    y_pixels = np.arange(0, warped.shape[0])
    X, Y = np.meshgrid(x_pixels, y_pixels)

    #Returns destinations in (y, x) coordinates
    destinations = fd_vect(X, Y, flow_field[:,:,0], flow_field[:,:,1])

    #Quick check that the calculated destinations correspond to the correct flow
    #x_dest, y_dest = fd_vect(X, Y, flow_field[:,:,0], flow_field[:,:,1])
    #x_flow = x_dest - X
    #y_flow = y_dest - Y
    #components = np.empty((486, 648, 2))
    #components[:,:,0] = x_flow
    #components[:,:,1] = y_flow
    #plot_optical_flow_Farneback(original, components, n=10, save_loc=None)

    #For each pixel in the warped frame
    for x in range(0, warped.shape[1]):
        print("Warping col: " + str(x))
        for y in range(0, warped.shape[0]):
            #For each pixel in the original, find where in
            #the "next" image it is mapped to by the flow
            #Billinear interpolate the values from the next
            #image at this point, to get the pixel val for the warped img

            dest = (destinations[0][y,x], destinations[1][y,x])

            #If the destination index is not too large
            if int(round(dest[0], 0)) < original.shape[0] and int(round(dest[1], 0)) < original.shape[1]:
                #If the destination index is not negative:
                if int(round(dest[0], 0)) >= 0 and int(round(dest[1], 0)) >= 0:
                    #value = next[int(round(dest[0], 0)), int(round(dest[1], 0))]
                    value = grid_interpolation(next, (dest[0], dest[1]))
                    #value = billinear_interpolate(next, (dest[0], dest[1]))
                    warped[y, x] = value


    plt.imshow(warped, cmap='gray')
    plt.show()
    return warped






def interpolation_error(interpolated, target):
    diff = np.square(interpolated - target)
    sum = np.sum(diff) * (1/(diff.shape[0] * diff.shape[1]))
    return np.sqrt(sum)



def calc_interp_error_for_sequence(sequence, names, flow_sequence):
    for sequence_index in range(0, len(sequence) - 1):
        current_img = sequence[sequence_index]
        #plt.imshow(current_img)
        #plt.colorbar()
        #plt.show()
        next_img = sequence[sequence_index + 1]

        flow_vals = flow_sequence[sequence_index]
        #flow_vals = np.ones((486, 648, 2)) * 5

        plot_optical_flow_Farneback(current_img, flow_vals, n=10)
        warped_image = warp(current_img, next_img, flow_vals)

        IE = interpolation_error(warped_image, current_img)
        print("IE:" + str(IE))

def convert_sequence_to_UINT8(sequence):
    converted_sequence = []
    for image in sequence:
        converted_image = (image/4).astype("uint8")
        converted_sequence.append(converted_image)
    return converted_sequence



sample_prev = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/Rev22SampleImages/2022-04-24T172015_fltrA_1ag_1499994ss_Plume.png", -1)
sample_current = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/Rev22SampleImages/2022-04-24T172025_fltrA_1ag_1499994ss_Plume.png", -1)
names = ["022-04-24T172015_fltrA_1ag_1499994ss_Plume.png", "2022-04-24T172025_fltrA_1ag_1499994ss_Plume.png"]
clear = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/Rev22SampleImages/2022-04-07T200435_fltrA_1ag_3499986ss_Plume.png", -1)
dark = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/Rev22SampleImages/2022-03-31T041219_fltrA_1ag_1499994ss_Dark.png", -1)
#show(clear)
#show(dark)
sample_prev = sample_prev - dark
sample_current = sample_current - dark
vin_mask = clear/np.max(clear)
sample_prev = np.divide(sample_prev, vin_mask)
sample_current = np.divide(sample_current, vin_mask)
#show(sample_prev)
#show(sample_current)
sample_sequence = [sample_prev, sample_current]
sample_sequence = convert_sequence_to_UINT8(sample_sequence)
sample_sequence_flow_vals = np.load("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/Expmt10 - Std FB Interpolation Error/RevGoodQualCorrFlowValsFB.npy")
print(sample_sequence_flow_vals.shape)

calc_interp_error_for_sequence(sample_sequence, names, sample_sequence_flow_vals)
