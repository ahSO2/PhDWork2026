#Calculate the optical flow for each sample in a given dataframe,
#then evaluate.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
import sys
sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

samples_sheet = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/KilaueaLeftOut_Train.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
mod = 1
timesteps = ["prev_tensec_name", "image_name"]

df = pd.read_excel(samples_sheet)
df = df[df["overall_obs"]=="No"]


def convert_sequence_to_UINT8(sequence):
    converted_sequence = []
    for image in sequence:
        converted_image = (image/4).astype("uint8")
        converted_sequence.append(converted_image)
    return converted_sequence

def add_gauss_noise(sequence):
    noisy_sequence = []
    for image in sequence:
        scale = np.max(image)/2
        noise = np.random.normal(loc=0, scale =5, size=image.shape)
        noisy_sequence.append(image + noise)
    return noisy_sequence

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
    # if save_loc != None:
    #    plt.savefig(save_loc)
    plt.show()
    # plt.close()

def calculate_optical_flow_pair_Farneback(i1, i2, initial_flow=None, plot=False, n=5):
    flow = cv2.calcOpticalFlowFarneback(prev=i1, next=i2, flow =initial_flow, pyr_scale=0.5, levels=4, winsize=20, iterations=5, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    if plot==True:
        # Need array of x-coords, y-coords, then dx and dy
        # Downsample to plot every n-th coord point
        x_pixels = np.arange(0, i1.shape[1], n)
        y_pixels = np.arange(0, i1.shape[0], n)
        X, Y = np.meshgrid(x_pixels, y_pixels)

        # Downsample every nth flow vector for plotting
        flow_dx = flow[0::n, 0::n, 0]
        flow_dy = flow[0::n, 0::n, 1]

        plt.quiver(X, Y, flow_dx, flow_dy, color='g', scale_units='xy', scale=1, angles='xy')
        plt.gca().invert_yaxis()
        plt.imshow(i1, cmap="gray")
        plt.colorbar()
        #if save_loc != None:
        #    plt.savefig(save_loc)
        plt.show()
        #plt.close()

    return flow

def calculate_optical_flow_pair_HS(i1, i2, alpha=1, epsilon=1, iterations=300, initial_flow=None, plot=False):
    #Flow is indexed as (x, y, 0 for u=dI/dx or 1 for v=dI/dy)
    #X is the horixzontal direction, y the vertical

    #Initialise the flow
    if initial_flow != None:
        flow = initial_flow
    else:
        flow = np.zeros(shape=(i1.shape[0], i1.shape[1], 2))
    plot_dense_flow(flow, i1, n=20)

    #Smooth the images
    i1 = cv2.GaussianBlur(i1, (5, 5), 0)
    i2 = cv2.GaussianBlur(i2, (5, 5), 0)

    #Approximate derivatives of pixel value:
    Ix_kernel = np.array([[1, -1],
                           [1, -1]])

    Iy_kernel = np.array([[1, 1],
                          [-1, -1]])

    It_kernel = np.ones(shape=(2, 2, 2))
    It_kernel[:, :, 1] = np.array([[-1, -1], [-1, -1]])

    stacked = np.empty(shape=(i1.shape[0], i1.shape[1], 2))
    stacked[:, :, 0] = i1
    stacked[:, :, 1] = i2
    Ix = 1/4 * (ndimage.convolve(i1, Ix_kernel, mode="reflect") + ndimage.convolve(i2, Ix_kernel, mode="reflect"))
    Ix2 = 1/4 * (ndimage.convolve(Ix, Ix_kernel, mode="reflect") + ndimage.convolve(Ix, Ix_kernel, mode="reflect"))
    Iy = 1/4 * (ndimage.convolve(i1, Iy_kernel, mode="reflect") + ndimage.convolve(i2, Iy_kernel, mode="reflect"))
    Iy2 = 1/4 * (ndimage.convolve(Iy, Iy_kernel, mode="reflect") + ndimage.convolve(Iy, Iy_kernel, mode="reflect"))
    It = 1/4 * (ndimage.convolve(stacked, It_kernel))[:, :, 0]

    mean_kernel = 1 / 9 * np.ones(shape=(3, 3))
    for iter in range(1, iterations + 1):
        #Calculate mean flow around each pixel
        mean_u = cv2.filter2D(flow[:, :, 0], -1, kernel=mean_kernel, anchor=(-1,-1), borderType=cv2.BORDER_REFLECT)
        mean_v = cv2.filter2D(flow[:, :, 1], -1, kernel=mean_kernel, anchor=(-1,-1), borderType=cv2.BORDER_REFLECT)

        num_x = np.multiply(Ix, (np.multiply(Ix, mean_u) + np.multiply(Iy, mean_v) + It))
        den_x = alpha * alpha + np.multiply(Ix, Ix) + np.multiply(Iy, Iy) + epsilon
        updated_flow_x = mean_u - np.divide(num_x, den_x)

        num_y = np.multiply(Iy, (np.multiply(Ix, mean_u) + np.multiply(Iy, mean_v) + It))
        updated_flow_y = mean_v - np.divide(num_y, den_x)


        flow[:,:,0] = updated_flow_x
        flow[:,:,1] = updated_flow_y

        if iter % 50 == 0:
            plot_dense_flow(flow, i1, n=20)

    return flow

def calculate_optical_flow_pair_LK(i1, i2, n=20, plot=False):
    # Flow is indexed as (x, y, 0 for u=dx/dt or 1 for v=dy/dt)
    # X is the horixzontal direction, y the vertical

    # Need array of x-coords, y-coords, then dx and dy
    # Downsample to calculate and plot for every n-th coord point
    x_pixels = np.arange(0, i1.shape[1], n)
    y_pixels = np.arange(0, i1.shape[0], n)

    n_points = x_pixels.shape[0] * y_pixels.shape[0]

    points_to_track = np.empty((n_points, 1, 2), dtype=np.float32)
    counter = 0
    for y in y_pixels:
        for x in x_pixels:
            #Record point location in format x,y
            points_to_track[counter,0,0] = x
            points_to_track[counter,0,1] = y
            counter += 1

    updated_points, status, error = cv2.calcOpticalFlowPyrLK(prevImg=i1, nextImg=i2, prevPts=points_to_track, nextPts=None)
    print("Calculated flow!")

    start_x_coords = points_to_track[:, :, 0]
    start_y_coords = points_to_track[:, :, 1]

    new_x_coords = updated_points[:, :, 0]
    new_y_coords = updated_points[:, :, 1]
    x_displ = new_x_coords - start_x_coords
    y_displ = new_y_coords - start_y_coords
    flow_dx = np.reshape(x_displ, (y_pixels.shape[0], x_pixels.shape[0]))
    flow_dy = np.reshape(y_displ, (y_pixels.shape[0], x_pixels.shape[0]))
    flow = np.stack([flow_dx, flow_dy], axis=-1)

    print(flow.shape)

    if plot == True:
        plot_dense_flow(flow, i1, n=5)
        #fig, axs = plt.subplots(figsize=(10, 10))
        #vectors = axs.quiver(start_x_coords, start_y_coords, x_displ, y_displ, color='darkorange', scale_units='xy',
        #                 scale=1, angles='xy')
        #axs.set_title("LK Method")
        #axs.imshow(i1, cmap="gray")
        #plt.show()

    return flow


######################### Flux calculation functions:
def find_cicle_points(image_dim, x_range, y_range, center, radius):

    #For each x-value, find and plot corresponding y-values:
    x_to_consider = np.linspace(x_range[0], x_range[1], (x_range[1] - x_range[0] + 1) * 10)

    y_sq = np.ones_like(x_to_consider) * radius**2 - np.square(x_to_consider- np.ones_like(x_to_consider) * center[0])
    y_rt = np.sqrt(y_sq)

    y1_vals = y_rt + np.ones_like(x_to_consider) * center[1]
    y1_vals = np.round(y1_vals, 0).astype(int)

    y2_vals = -y_rt + np.ones_like(x_to_consider) * center[1]
    y2_vals = np.round(y2_vals, 0).astype(int)

    array = np.zeros(shape=image_dim)
    x_to_plot = np.round(x_to_consider, 0).astype(int)

    for x_index in range(0, len(x_to_consider)):
        array[y1_vals[x_index], x_to_plot[x_index]] = 1
        array[y2_vals[x_index], x_to_plot[x_index]] = 1

    #Then for each y-value, find and plot corresponding x-vals
    y_to_consider = np.linspace(y_range[0], y_range[1], (y_range[1] - y_range[0] + 1) * 10)

    x_sq = np.ones_like(y_to_consider) * radius ** 2 - np.square(y_to_consider - np.ones_like(y_to_consider) * center[1])
    x_rt = np.sqrt(x_sq)

    x1_vals = x_rt + np.ones_like(y_to_consider) * center[0]
    x1_vals = np.round(x1_vals, 0).astype(int)

    x2_vals = -x_rt + np.ones_like(y_to_consider) * center[0]
    x2_vals = np.round(x2_vals, 0).astype(int)

    y_to_plot = np.round(y_to_consider, 0).astype(int)

    for y_index in range(0, len(y_to_consider)):
        array[y_to_plot[y_index], x1_vals[y_index]] = 1
        array[y_to_plot[y_index], x2_vals[y_index]] = 1

    return array

def dist(p1, p2):
    sum = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return np.sqrt(sum)


def find_POI_vertical_line(x, c, r, point_approx):
    '''
    Find the POI between line x=x and circle with given center and radius.
    Return the value closest to the given y-value.
    :return:
    '''

    y_sqr = r**2 - (x-c[0])**2
    if y_sqr >= 0:
        #If a solution exists:
        y_rt = np.sqrt(y_sqr)
        y1 = y_rt + c[1]
        y2 = -y_rt + c[1]
        #Take the one closest to the y-val of given.
        if (point_approx[1] - 0.5) <= y1 < (point_approx[1] + 0.5):
            return y1
        elif (point_approx[1]-0.5) <= y2 < (point_approx[1]+0.5):
            return y2
        else:
            return "N"
    else:
        return "N"

def find_POI_horizontal_line(y, c, r, point_approx):
    '''
        Find the POI between line y=y and circle with given center and radius.
        Return the value closest to the given x-value.
        :return:
        '''

    x_sqr = r ** 2 - (y - c[1]) ** 2
    if x_sqr >= 0:
        # If a solution exists:
        x_rt = np.sqrt(x_sqr)
        x1 = x_rt + c[0]
        x2 = - x_rt + c[0]
        # Take the one closest to the y-val of given.
        if (point_approx[0] - 0.5) <= x1 < (point_approx[0] + 0.5):
            return x1
        elif (point_approx[0] - 0.5) <= x2 < (point_approx[0] + 0.5):
            return x2
        else:
            return "N"
    else:
        return "N"



def calculate_intersection_lengths(all_points, c, r):
    '''
    Intakes a set of points and returns the length of the "chunk" of the circle
    which intersects the pixel centered at each point, approximated by a straight line.
    :param all_points: Array containing pixels for which to calculate the intersection.
    :param circle_center: (x, y) as in (horizontal, vertical)
    :param circle_raidus: radius of circle to consider.
    :return:
    '''
    n_poi = np.zeros_like(all_points)
    arc_lengths = np.zeros_like(all_points).astype(np.float32)
    for x in range(0, all_points.shape[1]):
        for y in range(0, all_points.shape[0]):
            POIs = []

            # Find POI with each edge of that pixel
            x_l = x - 0.5
            x_r = x + 0.5
            y_t = y - 0.5
            y_b = y + 0.5

            #Find POIs between vertical boundary and the circle:
            x_l_POI_y_val = find_POI_vertical_line(x_l, c, r, (x, y))
            if x_l_POI_y_val != "N":
                POIs.append((x_l, x_l_POI_y_val))
            x_r_POI_y_val = find_POI_vertical_line(x_r, c, r, (x, y))
            if x_r_POI_y_val != "N":
                POIs.append((x_r, x_r_POI_y_val))
            #Find POIs between horizontal boundary and the circle:
            y_t_POI_x_val = find_POI_horizontal_line(y_t, c, r, (x, y))
            if y_t_POI_x_val != "N":
                POIs.append((y_t_POI_x_val, y_t))
            y_b_POI_x_val = find_POI_horizontal_line(y_b, c, r, (x, y))
            if y_b_POI_x_val != "N":
                POIs.append((y_b_POI_x_val, y_b))


            #print(len(POIs))
            n_poi[y, x] = len(POIs)

            # For each pixel with intersections
            # Calculate the distance of the arc approximated by a line
            if n_poi[y, x] == 2:
                arc_len = dist(POIs[0], POIs[1])
                arc_lengths[y, x] = arc_len


    #Check that each pixel has two POIs with the circle:
    #plt.imshow(boundary_points * 2 - n_poi)
    #plt.colorbar()
    #plt.show()
    #Note there is a slight mismatch where some pixels flagged as having intersection
    #aren't recorded as boundary points. This is because the method used to calculate
    #intersections accounts for even miniscule intersections, whereas the step size used
    #to find points on the circle is not infinitesimally small. So take the intersection
    #calculation as the correct standard.

    #TODO sense check that the total distance is roughly the circle circumference
    #plt.imshow(arc_lengths)
    #plt.colorbar()
    #plt.show()

    circ = np.pi * 2 * r
    print("Error in arc length approximation:")
    print(str(np.round(np.sum(arc_lengths) - circ, 4)))

    return arc_lengths



def calculate_flux_1D(frame1, flow, circle_center, circle_radius, flank_mask):
    '''Circle center: (x, y) as in (horizontal, vertical)'''

    #Select the set of points on the circle, stored as 1s in an array
    lhs = max(0,circle_center[0] - circle_radius)
    rhs = min(frame1.shape[1] -1, circle_center[0] + circle_radius)
    x_range = (lhs, rhs)
    top = max(0, circle_center[1] - circle_radius)
    bottom = min(frame1.shape[0] -1, circle_center[1] + circle_radius)
    y_range=(top, bottom)
        #np.linspace(lhs,rhs, rhs-lhs + 1).astype(int))
    boundary_points = find_cicle_points(frame1.shape, x_range, y_range, circle_center, circle_radius)

    #Use the flank mask to remove overlap
    boundary_points = np.where(flank_mask==0, 0, boundary_points)
    #plt.imshow(boundary_points)
    #plt.show()

    #Calculate the unit normal for each point
    #(take vector from center to that point, then scale so magnitude is one)
    x_coords = np.linspace(0, frame1.shape[1] - 1, frame1.shape[1])
    y_coords = np.linspace(0, frame1.shape[0] - 1, frame1.shape[0])
    #First create arrays of the x and y coords, and mask where we don't need to calculate
    X,Y = np.meshgrid(x_coords, y_coords)

    n_x = X - circle_center[0]
    n_y = Y - circle_center[1]
    n_m = np.sqrt(np.square(n_x) + np.square(n_y))
    n_x = np.divide(n_x, n_m)
    n_y = np.divide(n_y, n_m)
    #Set the nromal at the center point equal to zero rather than invalid value
    n_x[circle_center[1], circle_center[0]] = 0
    n_y[circle_center[1], circle_center[0]] = 0

    normals = np.stack([n_x, n_y], axis=-1)
    frame_w_circle = np.where(boundary_points==1, np.max(frame1), frame1)
    #plot_dense_flow(normals, frame_w_circle, n=5)

    #Then take dot product with the velocity at that point
    dot = np.multiply(n_x, flow[:, :, 0]) + np.multiply(n_y, flow[:, :, 1])

    flow_to_plot = flow.copy()
    flow_to_plot[:,:,0] = np.where(boundary_points == 0, 0, flow[:, :, 0])
    flow_to_plot[:, :, 1] = np.where(boundary_points == 0, 0, flow[:, :, 1])
    #Plot showing the flow, colourmapped by the magnitude of the dot product (component of velocity normal to the circle) at each point on circle)
    #plot_dense_flow(flow_to_plot, frame_w_circle, n=5, color_array=dot)

    #Calculate the arc length of the "piece" of circle passing though each pixel
    #Approximated using a straight line
    arc_lengths = calculate_intersection_lengths(boundary_points, circle_center, circle_radius)

    #TODO mask out the flank

    #Lastly, mulptiply by the intensity values, then sum
    #TODO change this to use values where intersection length is nonzero, and scale by the intersection length 
    contributions = np.multiply(arc_lengths, np.multiply(frame1, dot))
    intensity_flux_1D = np.sum(contributions)

    #TODO need to scale by calibration value
    #TODO need to scale by pixel size

    plt.imshow(contributions)
    plt.colorbar()
    plt.show()

def dist_w_fixed_point(X, Y, fixed_point):
    '''
    :param X: Array of x-values for points to calc distance
    :param Y: Array of y-values for points to calc distance
    :param fixed_point: Fixed point coords as (x, y) as in (horizontal, vertical)
    :return:
    '''

    term1 = np.square(X - fixed_point[0])
    term2 = np.square(Y - fixed_point[1])
    return np.sqrt(term1 + term2)

def closest_dist_w_fixed_point(X, Y, fixed_point):
    '''Calculate distance of each corner of each pixel in the
    given array with the fixed point, and return the smallest.'''
    d1s = dist_w_fixed_point(X - 0.5, Y - 0.5, fixed_point)
    d2s = dist_w_fixed_point(X + 0.5, Y - 0.5, fixed_point)
    d3s = dist_w_fixed_point(X - 0.5, Y + 0.5, fixed_point)
    d4s = dist_w_fixed_point(X + 0.5, Y + 0.5, fixed_point)
    ds = np.stack([d1s, d2s, d3s, d4s], axis=-1)
    return np.min(ds, axis=2)

def calculate_flux_2D(frame1, flow, circle_center, circle_radius, flank_mask):
    #arc_lengths = calculate_intersection_lengths(frame1, circle_center, circle_radius)
    #circle_boundary = np.where(arc_lengths>0, 1, 0) #Array of points of intersection of the
    #circle. Want to use this as the boundary.

    #plt.imshow(circle_boundary)
    #plt.show()

    x_coords = np.linspace(0, frame1.shape[1] - 1, frame1.shape[1])
    y_coords = np.linspace(0, frame1.shape[0] - 1, frame1.shape[0])
    X_start, Y_start = np.meshgrid(x_coords, y_coords)

    start_dists = closest_dist_w_fixed_point(X_start, Y_start, circle_center)
    start_in_circle = np.where(start_dists < circle_radius, 1, 0)
    #Sense check that the area we are using is bounded exactly by the boundary
    #used in the 1D method (found though POI of each pixel with the circle) - looks good, yay!
    #plt.imshow(circle_boundary * 2 - start_in_circle)
    #plt.colorbar()
    #plt.show()

    X_dest = X_start + flow[:, :, 0]
    Y_dest = Y_start + flow[:, :, 1]
    end_dists = closest_dist_w_fixed_point(X_dest, Y_dest, circle_center)
    end_in_circle = np.where(end_dists < circle_radius, 1, 0)

    #Next need to work out which points have moved into or out of the circle
    movement = start_in_circle - end_in_circle
    plt.imshow(movement)
    plt.colorbar()
    plt.show()

    #To calculate emission mass, multiply each pixel value by this movement:
    mass = np.sum(np.multiply(frame1, movement))
    #TODO need to scale for timestep length to give an average emission rate

    return mass

############################# Optical flow evaluation functions
def flow_magnitude(flow):
    return np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:, :, 1]))

def movement_eval(image_name, image, flow):
    '''Does the area that is counted as moving by the flow contain the whole plume
    that I have annotated?'''

    #We want the area that is counted as moving contain the whole plume
    flow_mag = flow_magnitude(flow)

    plt.imshow(flow_mag)
    plt.colorbar()
    plt.show()

    moving = np.where(flow_mag > 0, 1, 0)
    plt.imshow(moving)
    plt.colorbar()
    plt.show()










######################### Run the code:
#For each sample:
df.reset_index(inplace=True)
for index in range(0, df.shape[0], mod):
    #Create a sequence of timestep images
    sequence = []
    names = []
    dictionary_name = df["volcano_dictionary_name"][index]
    for timestep_name in timesteps:
        if timestep_name == "image_name":
            folder_to_read = data_path
        elif timestep_name == "image_name_B":
                folder_to_read = data_path
        else:
            folder_to_read = data_path_temporal
        name_to_read = df[timestep_name][index]
        timestep_image = cv2.imread(folder_to_read + "/" + name_to_read, -1)
        sequence.append(timestep_image)
        names.append(name_to_read)

    #TODO Conversion to UINT8 may affect optimisation or other calculations
    sequence = convert_sequence_to_UINT8(sequence)
    #sequence = add_gauss_noise(sequence)

    #Calculate the FB optical flow with standard parameters
    #flow = calculate_optical_flow_pair_Farneback(sequence[0], sequence[1], plot=True)
    #flow = calculate_optical_flow_pair_HS(sequence[0], sequence[1], alpha=1, epsilon=0.5, plot=True)
    flow = calculate_optical_flow_pair_LK(sequence[0], sequence[1], n=1, plot=True)

    #Define the integration boundary
    dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
    int_circle_center = dictionary["integration_region_center"]
    int_circle_radius = dictionary["integration_radius"]
    flank_mask = cv2.imread(dictionary["flank_mask_path"], -1)


    #calculate_flux_1D(sequence[0], flow, circle_center=int_circle_center, circle_radius=int_circle_radius, flank_mask=flank_mask)
    #calculate_flux_2D(sequence[0], flow, circle_center=int_circle_center, circle_radius=int_circle_radius, flank_mask=flank_mask)

    movement_eval(names[0], sequence[0], flow)


#Calculate the interpolation error

#TODO consider the impact of scaling image to UINT8

