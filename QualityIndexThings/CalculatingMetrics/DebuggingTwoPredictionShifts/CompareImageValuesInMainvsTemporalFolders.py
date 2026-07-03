#Loop through every sample in the main dataset folder
#If it also exists in the temporal folder, then read both images,
#Check the pair are equal, and if not then save a plot showing
#each image and the difference between them.
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

main_folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/PotentialNewDataForRecursiveLabellingKilauea/"
temporal_folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/PotentialNewDataForRecursiveLabellingKilaueaTemporal/"
plots_save_path = "C:/Users/ggp24ash/Documents/Scratch Data/CheckingDuplicateSamples/KilaueaCVAdditionalPool_MainvsTemporal/"

main_folder_samples = os.listdir(main_folder_path)
temporal_folder_samples = os.listdir(temporal_folder_path)

index = 0
for image_name in main_folder_samples:
    print(index)
    index += 1
    if image_name in temporal_folder_samples:
        image_1 = cv2.imread(main_folder_path + image_name, -1)
        image_2 = cv2.imread(temporal_folder_path + image_name, -1)
        diff = np.abs(image_1.astype("float32") - image_2.astype("float32"))
        if np.array_equal(image_1, image_2) == False:
            fig, axs = plt.subplots(ncols=3, figsize=(10,6))
            left_plot = axs[0].imshow(image_1, cmap='gray')
            center_plot = axs[1].imshow(image_2, cmap='gray')
            right_plot = axs[2].imshow(diff, cmap="gray")
            axs[0].set_title("Main Folder")
            axs[1].set_title("Temporal Folder")
            axs[2].set_title("Difference")
            fig.colorbar(left_plot, ax=axs[0], shrink=0.5)
            fig.colorbar(center_plot, ax=axs[1], shrink=0.5)
            fig.colorbar(right_plot, ax=axs[2], shrink=0.5)
            plt.tight_layout()
            plt.savefig(plots_save_path + image_name, dpi=300)
            plt.close()