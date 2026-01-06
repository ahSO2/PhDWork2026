#Write a function which takes in a sequence of images
#And calculates the optical flow between them
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show(img):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()

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

def calculate_optical_flow_pair_Farneback(i1, i2, initial_flow):
    flow = cv2.calcOpticalFlowFarneback(prev=i1, next=i2, flow =None, pyr_scale=0.5, levels=4, winsize=20, iterations=5, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    print(type(flow))
    return flow

def calculate_optical_flow_pair_LK(i1, i2, n, initial_flow):
    # Need array of x-coords, y-coords, then dx and dy
    # Downsample to plot every n-th coord point
    print(i1.dtype)
    x_pixels = np.arange(0, 648, n)
    y_pixels = np.arange(0, 486, n)
    #X, Y = np.meshgrid(x_pixels, y_pixels)
    n_points = x_pixels.shape[0] * y_pixels.shape[0]
    #print(n_points)
    points_to_track = np.empty((n_points,1, 2), dtype=np.float32)
    counter = 0
    for x in x_pixels:
        for y in y_pixels:
            #Record point location in format x,y
            points_to_track[counter,0,0] = x
            points_to_track[counter,0,1] = y
            counter += 1

    print(points_to_track)
    #output1 = cv2.buildOpticalFlowPyramid(img=i1, winSize=(15,15), maxLevel=0)
    #print(output1)
    #output2 = cv2.buildOpticalFlowPyramid(img=i2, winSize=(15,15), maxLevel=0)
    updated_points, status, error = cv2.calcOpticalFlowPyrLK(prevImg=i1, nextImg=i2, prevPts=points_to_track, nextPts=None)
    print("Calculated flow!")
    return points_to_track, updated_points

def calculate_optical_flow_pair_pyramids(i1, i2):
    #Decide how many pyramid levels
    #Create the pyramid images
    pass


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
        noise = np.random.normal(loc=0, scale =2, size=image.shape)
        noisy_sequence.append(image + noise)
    return noisy_sequence

def plot_optical_flow_Farneback(image, flow_components, n, save_loc=None):
    #Need array of x-coords, y-coords, then dx and dy
    #Downsample to plot every n-th coord point
    x_pixels = np.arange(0,648,n)
    y_pixels = np.arange(0,486,n)
    X, Y = np.meshgrid(x_pixels, y_pixels)

    #Downsample every nth flow vector for plotting
    flow_dx = flow_components[0::n,0::n,0]
    flow_dy = flow_components[0::n,0::n,1]

    plt.quiver(X, Y, flow_dx, flow_dy, color='g')
    plt.gca().invert_yaxis()
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    if save_loc != None:
        plt.savefig(save_loc)
    #plt.show()
    plt.close()

def plot_optical_flow_LK(original_points, updated_points, image, save_loc=None):
    # To plot flow field, need to make arrays of x-coords, y-coords, x_flow and y_flow
    # Need to take negative of y_displacent because of indexing #This gives much more reasonable results!!

    start_x_coords = original_points[:, :, 0]
    start_y_coords = original_points[:, :, 1]
    # print(start_y_coords)
    new_x_coords = updated_points[:, :, 0]
    new_y_coords = updated_points[:, :, 1]
    x_displ = new_x_coords - start_x_coords
    y_displ = new_y_coords - start_y_coords

    #y_displ_to_plot = -y_displ
    #start_y_coords_to_plot = 486 - start_y_coords

    plt.quiver(start_x_coords, start_y_coords, x_displ, y_displ)
    plt.title("LK Method")
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    #if save_loc != None:
    #    plt.savefig(save_loc)
    plt.show()
    #plt.close()
def calculate_optical_flow_sequence(sequence, names):
    #For each image in the sequence
    #Calculate the flow from that image to the next
    prev_flow = None
    flow_sequence = []
    for sequence_index in range(0, len(sequence) - 1):
        current_img = sequence[sequence_index]
        #plt.imshow(current_img)
        #plt.colorbar()
        #plt.show()
        next_img = sequence[sequence_index + 1]

        #Farneback #################
        flow = calculate_optical_flow_pair_Farneback(current_img, next_img, prev_flow)
        #plot_optical_flow_Farneback(image=current_img,flow_components=flow, n=20, save_loc=folder_to_save + "/" + names[sequence_index] + ".png")
        plot_optical_flow_Farneback(image=current_img, flow_components=flow, n=10, save_loc=folder_to_save)
        #prev_flow = flow

        #LK ########################
        #original_points, updated_points = calculate_optical_flow_pair_LK(i1 = current_img, i2=next_img, n=20, initial_flow=None, )
        #plot_optical_flow_LK(original_points=original_points, updated_points=updated_points, image=current_img, save_loc=folder_to_save + "/" + names[sequence_index] + ".png")

        flow_sequence.append(flow)
    return flow_sequence

def calculate_optical_flow_on_samples(samples_sheet, data_path, data_path_temporal, mod):
    optical_flow_sequences = []
    timesteps = ["prev_tensec_name", "image_name"]
    #For each image in the specified training set
    dataset = pd.read_excel(samples_sheet)
    #dataset = dataset[dataset["overall_obs"] == "No"]
    #dataset = dataset[dataset["volcano_name"]=="Merapi"]
    dataset.reset_index(inplace=True)
    for index in range(0, dataset.shape[0], mod):
        #Read the sequence
        sequence = []
        names = []
        for timestep_name in timesteps:
            if timestep_name == "image_name":
                folder_to_read = data_path
                print(dataset[timestep_name][index])
            elif timestep_name == "image_name_B":
                folder_to_read = data_path
            else:
                folder_to_read = data_path_temporal
            name_to_read = dataset[timestep_name][index]
            timestep_image = cv2.imread(folder_to_read + "/" + name_to_read, -1)
            #plt.imshow(timestep_image)
            #plt.show()
            sequence.append(timestep_image)
            names.append(name_to_read)
        sequence = convert_sequence_to_UINT8(sequence)
        print("Data type after converting sequence:")
        print(sequence[0].dtype)
        sequence = add_gauss_noise(sequence)
        #Calculate optical flow
        flow_sequence = calculate_optical_flow_sequence(sequence, names)
        optical_flow_sequences.append(flow_sequence)
        #Save the results
    return optical_flow_sequences

#A easy sample pair #############################
sample_sequence = convert_sequence_to_UINT8(sample_sequence)
sample_sequence = add_gauss_noise(sample_sequence)
folder_to_save = "None"
flow_values = calculate_optical_flow_sequence(sample_sequence, names)
np.save("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/Expmt10 - Std FB Interpolation Error/RevGoodQualCorrFlowValsFB.npy", flow_values)
###########################################


samples_sheet = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/FinalSplit/Train.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
#folder_to_save = "Optical Flow Outputs/Expmt7 - FBStdPlusNoise - OnWoCotTrainSet"
folder_to_save = None
mod = 1

corr_flow_sequences = calculate_optical_flow_on_samples(samples_sheet, data_path, data_path_temporal, mod)
np.save("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/Expmt10 - Std FB Interpolation Error/TrainSetCorrFlowValsFB.npy", corr_flow_sequences)