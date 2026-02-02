import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import generic_filter

def show(img):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()

def convert_sequence_to_UINT8(sequence):
    converted_sequence = []
    for image in sequence:
        converted_image = (image/4).astype("uint8")
        converted_sequence.append(converted_image)
    return converted_sequence

'''
sample_prev = cv2.imread("Rev22SampleImages/2022-04-24T172015_fltrA_1ag_1499994ss_Plume.png", -1)
sample_current = cv2.imread("Rev22SampleImages/2022-04-24T172025_fltrA_1ag_1499994ss_Plume.png", -1)
names = ["022-04-24T172015_fltrA_1ag_1499994ss_Plume.png", "2022-04-24T172025_fltrA_1ag_1499994ss_Plume.png"]
clear = cv2.imread("Rev22SampleImages/2022-04-07T200435_fltrA_1ag_3499986ss_Plume.png", -1)
dark = cv2.imread("Rev22SampleImages/2022-03-31T041219_fltrA_1ag_1499994ss_Dark.png", -1)
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
'''

#Trying to mask for areas that move/change
#changed_pixels = np.where(sample_prev==sample_current, 0, sample_prev)
#show(changed_pixels)

def haralick_feature(image):
    # Calculate a GLCM with pixel offset 1 for each angle in [0, pi/4, p1/2, (3/4)pi]
    # Setting symmetric = True means that co-ocurrences at angles [pi, (5/4)pi, (3/2)pi, (7/4)pi]
    # are included in the counts for the calculated matrices
    glcms = graycomatrix(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, (3 / 4) * np.pi], levels=256,
                             symmetric=True, normed=False)
    # Sum the co-occurrences calculated for each angle
    invariant_glcm_o1_A = glcms[:, :, 0, 0] + glcms[:, :, 0, 1] + glcms[:, :, 0, 2] + glcms[:, :, 0, 3]
    # Then normalise
    total_count_o1_A = np.sum(invariant_glcm_o1_A)
    norm_glcm_o1_A = invariant_glcm_o1_A / total_count_o1_A
    # Then place into a 4D array (sklearn predefined functions used below require this)
    norm_glcm_4D_o1_A = np.empty((norm_glcm_o1_A.shape[0], norm_glcm_o1_A.shape[1], 1, 1))
    norm_glcm_4D_o1_A[:, :, 0, 0] = norm_glcm_o1_A

    #show(norm_glcm_4D_o1_A[:,:,0,0])
    hom_value = graycoprops(norm_glcm_4D_o1_A, '')[0, 0]
    return hom_value

#Difference in pixel value:
def pixel_diff(sample_current, sample_next):
    #show(sample_next)
    diff = sample_next.astype("float32") - sample_current.astype("float32")
    #Shift values to be non-zero
    #if np.min(diff) < 0:
    #    diff = diff - np.min(diff)
    #    print(np.min(diff))
    show(diff)
    # Convert to 8bit
    #diff_8bit = diff.astype('uint8')

    #show(diff_8bit)
    #entropy_img = entropy(diff_8bit, disk(10))
    #show(entropy_img)

    image = diff
    print(diff.dtype)
    '''
    grid_block_size = 10
    nv = math.ceil(image.shape[0] / grid_block_size)
    nh = math.ceil(image.shape[1] / grid_block_size)
    results_img = np.ones_like(image).astype("float32")
    
    for v in range(0, nv):
        for h in range(0, nh):
            #Select relevant pixels
            v_start = v * grid_block_size
            if v == (nv -1): #If its the last grid box
                v_end = image.shape[0]
            else:
                v_end = v_start + grid_block_size
            h_start = h * grid_block_size
            if h == (nh - 1):
                h_end = image.shape[1]
            else:
                h_end = h_start + grid_block_size

            image_tile = image[v_start:v_end, h_start:h_end]
            #print(image_tile.shape)
            #tile_value = haralick_feature(image_tile)
            tile_value = np.mean(image_tile)
            results_img[v_start:v_end, h_start:h_end] = np.ones_like(image_tile).astype("float32") * tile_value
    #show(results_img)
    '''
    image_min = np.min(image)
    denoised = cv2.medianBlur((image - image_min).astype("uint8"),ksize=9)
    denoised = denoised.astype("float32") + image_min
    print(denoised.dtype)
    show(denoised)
    hist_data = cv2.calcHist([diff], [0], None, [511], [-255, 255])
    grey_vals = np.linspace(-255,255, 511)
    #print(hist_data.shape)
    #print(grey_vals.shape)
    plt.scatter(grey_vals, hist_data)
    plt.show()
    results_img = np.abs(denoised)
    results_img = np.where(results_img >= 2, 1, 0)
    show(results_img)
    #results_img = np.where(results_img>(np.percentile(results_img[450:480,:], 95)), 1, 0)
    #show(results_img)

    #noise_mask = np.where(np.abs(diff) == 1, 1, 0).astype("uint8")
    #noise_infilled_img = cv2.inpaint(diff,noise_mask,3,cv2.INPAINT_TELEA)
    #show(noise_infilled_img)

    #results_img = np.where(np.abs(noise_infilled_img > 0), 1, 0)
    #show(results_img)


#Trying background subtraction OpenCV
#Implementation based on: https://www.geeksforgeeks.org/python/python-opencv-background-subtraction/
def MOG(sample_current, sample_next):
    fgbg_mog = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg_mog.apply(sample_current)
    fgmask = fgbg_mog.apply(sample_next)
    show(fgmask)


def apply_function_on_train_samples(samples_sheet, data_path, data_path_temporal, mod):
    timesteps = ["prev_tensec_name", "image_name"]
    #For each image in the specified training set
    dataset = pd.read_excel(samples_sheet)
    dataset = dataset[dataset["overall_quality"] == "Good"]
    #dataset = dataset[dataset["volcano_name"]=="Reventador"]
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
        #sequence = add_gauss_noise(sequence)
        #Calculate optical flow
        for sequence_index in range(0, len(sequence) - 1):
            current_img = sequence[sequence_index]
            #show(current_img)
            next_img = sequence[sequence_index + 1]
            #show(next_img)
            pixel_diff(current_img, next_img)
            #MOG(current_img, next_img)

        #Save the results

#TODO This is the old CV split that I was initialy using: seen notes in Seg Plans and Progress Log 1 - samples_sheet = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/PlumeSegmentation/TrainValidTestSplits/CrossValidation/WithoutCotopaxi/Train.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
#folder_to_save = "Optical Flow Outputs/Expmt7 - FBStdPlusNoise - OnWoCotTrainSet"
folder_to_save = "none"
mod = 20
apply_function_on_train_samples(samples_sheet, data_path, data_path_temporal, mod)
