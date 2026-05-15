#Calculate the between-frame pixel value difference (for movement ID)
#and the rough AA (ignoring background subtraction) for each sample
#in a given dataframe.
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import VolcDictionaryWithCorrectClears
from skimage.restoration import denoise_bilateral
import numpy as np

dataframe = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/CrossValidationSplits/CotopaxiLeftOut_Valid.xlsx")
data_source = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
temporal_data_source = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal/"
destination = "C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/InputChannels/woCotopaxi/"
sensor_mark_masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/"

def show(image):
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + "/" + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

def pixel_diff(current, next):
    current = denoise_bilateral(current.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    next = denoise_bilateral(next.astype("float32"), sigma_color = 5, sigma_spatial = 10, win_size=20)
    diff = np.abs(next.astype("float32") - current.astype("float32"))
    return diff

def calc_rel_AA(i_A, i_B, flank_mask):
    edge_mask = np.where(i_B == 0, 5, 0)
    edge_mask = cv2.blur(edge_mask, ksize=(5, 5))
    ratio = np.divide(i_B, i_A)
    masked_ratio = np.ma.masked_where(edge_mask > 0, ratio)
    rel_AA = np.ma.log(masked_ratio)
    flank_AA = np.ma.median(np.ma.masked_where(flank_mask == 1, rel_AA))
    rel_AA = rel_AA - flank_AA
    rel_AA = np.ma.filled(rel_AA, fill_value=0)
    rel_AA = np.where(rel_AA < 0, 0, rel_AA)
    # Subtract such that the mean AA over the flank is zero
    rel_AA = np.where(flank_mask == 0, 0, rel_AA)
    return rel_AA

dataframe.reset_index(inplace=True)
for sample_index in range(0, dataframe.shape[0]):
    print(sample_index)
    dictionary_name = dataframe["volcano_dictionary_name"][sample_index]
    batch = dataframe["labelling_batch_name"][sample_index]
    dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
    smmn_A = dictionary["sensor_marks_mask_A"]
    smmn_B = dictionary["sensor_marks_mask_B"]
    flank_mask = cv2.imread(dictionary["flank_mask_path"], -1)

    #Read image A and B
    image_name_A = dataframe["image_name"][sample_index]
    image_name_B = dataframe["image_name_B"][sample_index]
    image_A = cv2.imread(data_source + image_name_A, -1)
    image_B = cv2.imread(data_source + image_name_B, -1)

    #Read image for the next timestep
    plus_ten_s_name = dataframe["next_tensec_name"][sample_index]
    next_frame = cv2.imread(temporal_data_source + plus_ten_s_name, -1)

    #Apply sensor mark masks
    image_A = mask_sensor_marks(image_A, smmn_A)
    image_B = mask_sensor_marks(image_B, smmn_B)
    next_frame = mask_sensor_marks(next_frame, smmn_A)

    #Calculate difference
    difference = pixel_diff(image_A, next_frame)

    #Calculate absorbance
    rel_AA = calc_rel_AA(image_A, image_B, flank_mask)

    flank_mask = cv2.blur(np.where(flank_mask == 0, 5, 0), ksize=(10, 10))
    difference = np.where(flank_mask > 0, 0, difference)
    rel_AA = np.where(flank_mask > 0, 0, rel_AA)


    np.save(destination + "D_" + image_name_A[:-3] + ".npy", difference)
    np.save(destination + "A_" + image_name_A[:-3] + ".npy", rel_AA)