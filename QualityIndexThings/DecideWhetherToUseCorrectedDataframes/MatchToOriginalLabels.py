#For each image in the set of my corrections,
#find its original precip label and add this as a column.
#Visualise each image, the associated band B, and the old vs. corrected label.
import os

import cv2
import pandas as pd
import matplotlib.pyplot as plt

samples_to_correct = pd.read_excel("CorrectedMistakes_NoDuplicates.xlsx")
original_labels_df = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/UpdatedCorrectedDataframes/AllCorrectedChunkandIndivLabels.xlsx") #I think I made previous corrections to this set which is why its named "corrected"
folder_to_read_from = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2/"
additional_folder_to_read_from = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/AdditionalDataPool_SelectedForOnLensExpmt/"

'''
original_labels_key_info = original_labels_df[["image_name", "image_name_B", "on_lens_level"]]

corrections_df = pd.merge(samples_to_correct, original_labels_key_info, left_on='image_name', right_on='image_name', how='left')
corrections_df.rename({"on_lens_level":"original_precipitation_level", "correct_precip_level":"corrected_precipitation_level"}, inplace=True, axis="columns")
corrections_df.to_excel("CorrectedMistakes_NoDuplicates_wOldValsToCompare.xlsx")

#Just to check #########################################################
corrections_df["matched_correctly"] = samples_to_correct["image_name"]==corrections_df["image_name"]
print(corrections_df["matched_correctly"].value_counts())
'''

#TODO Note I have manually filled in the original prediction and band-B-name data
#for four corrected samples which were not in the original full dataset (but instead in the expanded precip training set).
corrections_df = pd.read_excel("CorrectedMistakes_NoDuplicates_wOldValsToCompare.xlsx")

for index in range(46, len(corrections_df)):
    print(index)
    print(corrections_df["image_name"].tolist()[index])
    if corrections_df["image_name"].tolist()[index] in os.listdir(folder_to_read_from):

        image_A = cv2.imread(folder_to_read_from + corrections_df["image_name"].tolist()[index], -1)
        image_B = cv2.imread(folder_to_read_from + corrections_df["image_name_B"].tolist()[index], -1)
    else:
        image_A = cv2.imread(additional_folder_to_read_from + corrections_df["image_name"].tolist()[index], -1)
        image_B = cv2.imread(additional_folder_to_read_from + corrections_df["image_name_B"].tolist()[index], -1)

    original_prediction = corrections_df["original_precipitation_level"].tolist()[index]
    corrected_prediction = corrections_df["corrected_precipitation_level"].tolist()[index]
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(image_A, cmap="gray")
    axs[1].imshow(image_B, cmap="gray")
    plt.title("Original: " + original_prediction + "    Corrected: " + corrected_prediction)
    plt.show()