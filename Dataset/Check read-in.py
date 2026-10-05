import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labels_path = "AllLabelledImagesList_UpdatedClassifications.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_UpdatedCorrections2026/"
labels = pd.read_excel(labels_path)

for index in range(0, labels.shape[0]):
    if "Kilauea" in labels["image_name"][index]:
        print(index)
        image_A = cv2.imread(data_path + labels["image_name"][index], -1)
        image_B = cv2.imread(data_path + labels["image_name_B"][index], -1)

        bandA_copy = image_A.copy()
        edge_mask = np.where(image_B == 0, 0, 1) * 5
        edge_mask = cv2.blur(edge_mask, (5, 5))
        edge_mask[:, 0] = 0
        edge_mask[:, -1] = 0
        edge_mask[0, :] = 0
        edge_mask[-1, :] = 0
        bandA_copy = np.ma.masked_where(edge_mask < 5, bandA_copy)

        #Then calculate the absorbance (without accounting for backgrounds)
        ratio = np.ma.divide(bandA_copy.astype(np.float32), image_B.astype(np.float32))
        ratio = -1 * np.ma.log(ratio)

        plt.imshow(ratio, cmap="YlGnBu_r")
        plt.colorbar()
        plt.title(labels["image_name"][index])
        plt.show()