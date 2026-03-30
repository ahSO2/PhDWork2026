#Read the dataset folder, and for every image resize and overwrite it
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset_path = "C:/Users/ggp24ash/Documents/CV Course/Assignment/dataset/"

sub_folders = ["train/", "test/"]

for sub_folder_name in sub_folders:
    for sub_sub_folder in os.listdir(dataset_path + sub_folder_name):
        #For each class folder, list all the images
        for image_name in os.listdir(dataset_path + sub_folder_name + sub_sub_folder + "/"):
            image = cv2.imread(dataset_path + sub_folder_name + sub_sub_folder + "/" + image_name)
            min_dim = min(image.shape[0], image.shape[1])
            extra_v = int(np.floor((image.shape[0] - min_dim)/2))
            extra_h = int(np.floor((image.shape[1] - min_dim)/2))

            small = image[extra_v:extra_v+min_dim, extra_h:extra_h+min_dim]

            small = cv2.resize(small, (128,128))

            if "jfif" in image_name:
                image_name = image_name[0:-4] + "png"

            cv2.imwrite(dataset_path + sub_folder_name + sub_sub_folder + "/" + image_name, small)