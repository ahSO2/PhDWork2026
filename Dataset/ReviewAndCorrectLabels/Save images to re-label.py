import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

location = "Kilauea"
save_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/ReviewAndUpdateLabels/"

#For each image name in the list
#Read it, scale the brightness for visibility and convert to UINT8
#Save in a folder

#os.mkdir(save_path + location + "/ToLabel")
images_to_convert = pd.read_excel(save_path + location + "/LabelsToUpdate.xlsx")['image_name']

def scale_brightness(image):
    image_max = np.percentile(image, 95)
    scale_factor = 900/image_max
    scaled_image = image.astype("float") * scale_factor
    scaled_image = np.where(scaled_image > 1023, 1023, scaled_image)
    return scaled_image

for image_name in images_to_convert:
    image = cv2.imread("C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2" + "/" + image_name, -1)
    image = scale_brightness(image)
    image = (image/4).astype('uint8')
    cv2.imwrite(save_path + location + "/ToLabel/" + image_name, image)

