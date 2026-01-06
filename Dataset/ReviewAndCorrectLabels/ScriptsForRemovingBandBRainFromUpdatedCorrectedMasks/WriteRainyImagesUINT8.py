import cv2
import numpy as np
import pandas as pd

rainy_data = pd.read_excel("LabelsWithBandBRain.xlsx")
all_labelled_images = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList.xlsx")
folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"

all_band_A_names = all_labelled_images["image_name"].tolist()
all_band_B_names = all_labelled_images["image_name_B"].tolist()

corresponding_band_B_names = []

def scale_brightness(image):
    image_max = np.percentile(image, 99)
    scale_factor = 255/image_max
    scaled_image = image.astype("float") * scale_factor
    scaled_image = np.where(scaled_image > 255, 255, scaled_image)
    return scaled_image.astype("uint8")

#Write the data to label to a folder
for image_name in rainy_data["image_name"]:
    image_name_index = all_band_A_names.index(image_name)
    band_B_name = all_band_B_names[image_name_index]
    corresponding_band_B_names.append(band_B_name)
    image_B = cv2.imread(folder_path + "/" + band_B_name, -1)
    converted_img = (image_B.astype('float32') / 4).astype("uint8")
    converted_img = scale_brightness(converted_img)


    cv2.imwrite("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/RainyBandBDataUINT8_ForUpdatedMasks/" + band_B_name, converted_img)

rainy_data["image_name_B"] = corresponding_band_B_names
rainy_data.to_excel("LabelsWithBandBRain.xlsx")