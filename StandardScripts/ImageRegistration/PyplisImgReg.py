#Trying out the pyplis method for image registration:
#A simple translation is assumed.
#The optical flow between the two frames is computed and the peak
#of the histograms of magnitude and direction are selected.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyplis

band_A_path = "DataToView/2023-06-16T215115_fltrA_1ag_299986ss_Plume.png"
band_B_path = "DataToView/2023-06-16T215115_fltrB_1ag_34998ss_Plume.png"

def plot_dense_flow(flow, image, n, color_array=None):
    # Need array of x-coords, y-coords, then dx and dy
    # Downsample to plot every n-th coord point
    x_pixels = np.arange(0, image.shape[1], n)
    y_pixels = np.arange(0, image.shape[0], n)
    X, Y = np.meshgrid(x_pixels, y_pixels)

    # Downsample every nth flow vector for plotting
    flow_dx = flow[0::n, 0::n, 0]
    flow_dy = flow[0::n, 0::n, 1]

    if isinstance(color_array, np.ndarray):
        color_array = color_array[0::n, 0::n]
        plt.quiver(X, Y, flow_dx, flow_dy, [color_array], scale_units='xy', scale=1, angles='xy')
        plt.colorbar()
    else:
        plt.quiver(X, Y, flow_dx, flow_dy, color='g', scale_units='xy', scale=1, angles='xy')
    plt.gca().invert_yaxis()
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

bandA = cv2.imread(band_A_path, -1)
bandB = cv2.imread(band_B_path, -1)

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(bandA, cmap="gray")
axs[1].imshow(bandB, cmap="gray")
plt.show()

flowCalculator = pyplis.OptflowFarneback()
ImgA = pyplis.Img(bandA)
ImgB = pyplis.Img(bandB)
flow_field = flowCalculator.calc_flow(ImgB, ImgA)

#TODO Visualise the flow field
plot_dense_flow(flow_field, bandB, n=10)

#TODO calculate histograms of the magnitude and direction and select the peak values

#TODO Shift the bandB image using the identified displacements



