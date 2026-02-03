import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/VolcanoDictionaries")
import VolcDictionaryWithCorrectClears

label_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/FinalSplit/Train.xlsx"
#label_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/UpdatedCorrectedDataframes/AllCorrectedChunkandIndividualLabelsWithoutLastarria.xlsx"
#folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2"
folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2/"
masks_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ProcessedLabels_UpdatedAfterReview/"

sensor_mark_masks_path = ("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/")

mod = 9
plot_type = "Mask"
filter_for_location = None

def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + "/" + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

labels = pd.read_excel(label_path)
if filter_for_location != None:
    labels = labels[labels["volcano_name"]==filter_for_location]
    labels.reset_index(inplace=True)
    print(labels.shape)

def mask_with_color(image, mask, code_for_mask):

    red = np.where(mask > 0, code_for_mask[0], image)
    green = np.where(mask > 0, code_for_mask[1], image)
    blue = np.where(mask > 0, code_for_mask[2], image)
    color_img = cv2.merge((blue, green, red))
    return color_img


for index in range(0, labels.shape[0], mod):
    #TODO mask sensor marks
    print(index)

    image_name = labels["image_name"][index]
    image_name_B = labels["image_name_B"][index]
    precip = labels["precip_level"][index]
    bg_clear = labels["bg_clear"][index]
    fg_cloud = labels["fg_cloud_level"][index]

    batch = labels["labelling_batch_name"][index]
    print(image_name)
    image = cv2.imread(folder_path + image_name, -1)
    dictionary_name = labels['volcano_dictionary_name'][index]
    if dictionary_name == "MerapiTenthJune":
        dictionary_name = "MerapiTenthMay"
    elif dictionary_name == "MerapiSixteenthJune":
        dictionary_name = "MerapiSixteenthMay"
    volcano_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
    sensor_mask_path_A = volcano_dictionary["sensor_marks_mask_A"]
    image = mask_sensor_marks(image, sensor_mask_path_A)
    image = (image/4).astype("uint8")
    #image_name_B = labels["image_name_B"][index]
    #image_B = cv2.imread(folder_path + image_name_B, -1)

    labels_folder = masks_path + batch

    matching_labels = np.load(labels_folder + "/" + "PlumeAndExpPixels_" + image_name[:-4] + ".npy")

    minimum = np.min(image)
    plume_mask = matching_labels[0, :, :]
    #plume_masked_img = np.where(plume_mask > 0, minimum, image)
    plume_masked_img = mask_with_color(image, plume_mask, (250, 9, 216))
    exp_mask = matching_labels[1, :, :]
    #exp_masked_img = np.where(exp_mask > 0, minimum, image)
    exp_masked_img = mask_with_color(image, exp_mask, code_for_mask=(126, 24, 168))

    #fig, axs = plt.subplots(1, 3, figsize=(10,5))
    #minimum = min(np.min(plume_masked_img), np.min(exp_masked_img))
    #maximum = np.max(image)
    #original_plot = axs[0].imshow(image, cmap='gray', vmin=minimum, vmax=maximum)
    #plume_labels_plot = axs[1].imshow(plume_masked_img, cmap='gray', vmin=minimum, vmax=maximum)
    #exp_labels_plot = axs[2].imshow(exp_masked_img, cmap='gray', vmin=minimum, vmax=maximum)
    #axs[0].set_title("Original")
    #axs[1].set_title("Plume")
    #axs[2].set_title("Explosive")
    #plt.tight_layout()
    #plt.suptitle(image_name + "  OL:" + precip + " BGC:" + bg_clear + " FGC:" + fg_cloud)
    #plt.show()
    #plt.savefig("VisualisedLabels/" + image_name)

    plt.imshow(image, cmap='gray')
    #plt.imshow(plume_masked_img, alpha=(np.ones_like(plume_mask) - plume_mask))
    plt.show()
