import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.restoration import denoise_bilateral
from skimage.filters.rank import entropy
from skimage.morphology import disk

def show(image):
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

def read_sample(sample_index, df, timesteps):
    # Create a sequence of timestep images
    sequence = []
    names = []
    dictionary_name = df["volcano_dictionary_name"][sample_index]
    batch = df["labelling_batch_name"][sample_index]
    all_plume_mask = None
    for timestep_name in timesteps:
        if timestep_name == "image_name":
                folder_to_read = data_path
        elif timestep_name == "image_name_B":
            folder_to_read = data_path
        else:
            folder_to_read = data_path_temporal
        name_to_read = df[timestep_name][sample_index]
        timestep_image = cv2.imread(folder_to_read + "/" + name_to_read, -1)
        sequence.append(timestep_image)
        names.append(name_to_read)
        if timestep_name == "image_name":
            print(name_to_read)
            mask_path = segmentation_masks_path + batch + "/PlumeAndExpPixels_" + name_to_read.split(".")[0] + ".npy"
            # print("Reading plume mask from: " + mask_path)
            two_channel_mask = np.load(mask_path)  # Manually drawn plume mask (value 1 indicates plume)
            all_plume_mask = two_channel_mask[0, :, :] + two_channel_mask[1, :, :]
            all_plume_mask = np.where(all_plume_mask > 0, 1, 0)


    return sequence, names, all_plume_mask

def pixel_diff(current, next):
    current = denoise_bilateral(current.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    next = denoise_bilateral(next.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    diff = np.abs(next.astype("float32") - current.astype("float32"))
    return diff

def equally_scale_sequences(s1, s2):
    '''Scale both band A and band B sequences to the range [0,255],
    so that the max value collectively is 255. As both bands are
    scaled equally, this means that absorbance calculation is still
    valid.'''
    sf = max(np.max(s1), np.max(s2)) / 255
    s1 = np.divide(s1, sf)
    s2 = np.divide(s2, sf)
    return s1, s2

def normalise_for_ss(sequence, names):
    #Start with images in range [0,1023]
    for index in range(0):
        image = sequence[index].astype("float32")
        name = names[index]
        ss = int(name.split("_")[3][:-2])
        ssr = 1000000/ss

def adaptive_threshold(image, dim):
    kernel = np.ones((dim, dim)) * (1/(dim*dim))
    threshold = cv2.filter2D(image, -1, kernel)
    result = np.where(image > threshold, 1, 0)
    return result



def show_hist(image):
    n_bins=20
    counts, bins = np.histogram(image.flatten(), n_bins, [np.min(image), np.max(image)])
    bin_length = np.abs(np.max(image) - np.min(image))/n_bins
    bin_centers = np.linspace(bin_length/2, bin_length * (n_bins + 0.5), n_bins)
    fit = np.polyfit(bin_centers, counts, deg=10)
    p = np.poly1d(fit)
    plt.stairs(counts, bins)
    plt.plot(bin_centers, p(bin_centers))
    plt.show()


################### Main script:
#For each cross-valid split
locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]
df_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/"
timesteps = ["image_name", "next_tensec_name"]
timesteps_B = ["image_name_B", "next_tensec_name_B"]
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
segmentation_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/"

for llo in locations:
    print("Running tests on " + llo + "-left-out CV Fold.")
    train_df = pd.read_excel(df_path + llo + "LeftOut_Train.xlsx")
    train_df = train_df[train_df["overall_obs"] == "No"]
    train_df.reset_index(inplace=True)
    # For each sample
    for sample_index in range(0, train_df.shape[0]):
        # Read the timestep sequence
        sequence, names, plume_mask = read_sample(sample_index, train_df, timesteps)
        sequence_B, names_B, plume_mask = read_sample(sample_index, train_df, timesteps_B)
        #Now attempt segmentation:
        sequence = np.array(sequence).astype(np.float32)
        sequence_B = np.array(sequence_B).astype(np.float32)

        #Scale to range [0,255]
        #sequence, sequence_B = equally_scale_sequences(sequence, sequence_B)

        #Calculate the difference image
        difference = pixel_diff(sequence[0], sequence[1])
        #diff_entropy = entropy(difference, disk(20))
        show(difference)
        #show(diff_entropy)
        #show_hist(difference)

        #Calculate relative absorbance
        #Take log of bandB/bandA, for the current timestep image
        edge_mask = np.where(sequence_B[0]==0, 5, 0)
        edge_mask = cv2.blur(edge_mask, ksize=(5, 5))
        ratio = np.divide(sequence_B[0], sequence[0])
        masked_ratio = np.ma.masked_where(edge_mask>0, ratio)
        rel_AA = np.ma.log(masked_ratio)
        #show(rel_AA)

        #Goal 1: Select points which are likely to be plume

        #Mask out flank

        #Select points which are moving above the local mean
        difference = denoise_bilateral(difference.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
        diff_pts = cv2.adaptiveThreshold(difference.astype("uint8"), maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=21, C=-3)
        show(diff_pts)

        #Select points which are absorbing above local mean
        abs_pts = adaptive_threshold(rel_AA, dim=21)
        show(abs_pts) #TODO Need to add a C value

        #Goal 2: Take those points and

        #product = np.ma.multiply(rel_AA, difference)
        #product = np.ma.divide(product, rel_AA + difference)

        #fig, ax = plt.subplots(ncols=3)
        #ax[0].imshow(difference)
        #ax[1].imshow(rel_AA)
        #ax[2].imshow(product)
        #plt.show()




