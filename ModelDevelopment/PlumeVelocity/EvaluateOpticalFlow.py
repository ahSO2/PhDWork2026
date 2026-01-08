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

def warp(original, target, flow_field):
    #Create empty array to store warped frame

    #For each pixel in the original frame
        #Map to its new location
    #Use bil
    pass

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

        plot_optical_flow_Farneback(current_img, flow_vals, n=10)





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


