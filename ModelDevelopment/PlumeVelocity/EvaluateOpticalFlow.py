#Read in a set of samples and associated velocity field
#Interpolate the second timestep image, using the first
#with the given velocity field
#Evaluate the interpolation error


import cv2
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
    return (x + flow_x, y+ flow_y)

fd_vect = np.vectorize(find_dest)

def billinear_interpolate(values_array, coord):

    #Pad the array with copies of the edge rows
    padded_arr = np.pad(values_array, pad_width=1, mode="edge")

    #Round the coordinate to nearest integer
    #Coord is given in form (y, x)
    int_coord = (int(np.round(coord[0], 0)), int(np.round(coord[1], 0)))

    x = int_coord[1]
    y = int_coord[0]

    x1 = int_coord[1]
    x2 = int_coord[1] + 1
    y1 = int_coord[0]
    y2 = int_coord[0] + 1

    Q11 = padded_arr[x1, y1]
    Q12 = padded_arr[x1, y2]
    Q21 = padded_arr[x2, y1]
    Q22 = padded_arr[x2, y2]

    R1 = Q11*(x2 - x)/(x2 - x1) + Q21*(x - x1)/(x2 - x1)
    R2 = Q12*(x2 - x)/(x2 - x1) + Q22*(x - x1)/(x2 - x1)

    V = R1*(y2 - y)/(y2 - y1) + R2*(y - y1)/(y2 - y1)

    return V

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

            if int(round(dest[0], 0)) < original.shape[0] and int(round(dest[1], 0)) < original.shape[1]:
                value = next[int(round(dest[1], 0)), int(round(dest[0], 0))]

                #value = billinear_interpolate(next, (int(round(dest[0], 0)), int(round(dest[1], 0))))
                warped[y, x] = value


    plt.imshow(warped, cmap='gray')
    plt.show()






def interpolation_error(interpolated, target):
    pass

def calc_interp_error_for_sequence(sequence, names, flow_sequence):
    for sequence_index in range(0, len(sequence) - 1):
        current_img = sequence[sequence_index]
        #plt.imshow(current_img)
        #plt.colorbar()
        #plt.show()
        next_img = sequence[sequence_index + 1]

        flow_vals = flow_sequence[sequence_index]
        #flow_vals = np.ones((486, 648, 2)) * 5

        #plot_optical_flow_Farneback(current_img, flow_vals, n=10)
        warped_image = warp(current_img, next_img, flow_vals)





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
sample_sequence_flow_vals = np.load("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/Expmt10 - Std FB Interpolation Error/RevGoodQualCorrFlowValsFB.npy")
print(sample_sequence_flow_vals.shape)
calc_interp_error_for_sequence(sample_sequence, names, sample_sequence_flow_vals)


#TODO Warping is giving result with the wrong orientation
#TODO do a double check over the billinear interpolation calc