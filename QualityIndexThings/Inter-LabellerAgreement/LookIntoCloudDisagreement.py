#Review the samples where the three labellers disagree on the binary cloud level
#Make a note of whether in each case there is fog/haze/light dilution
#(for the purpose of understanding whether this is a key contributor to
# the disagreement).
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#cloud_labels_w_consensus_calc = pd.read_excel("ManualLabels/MergedLabels/consensus_calc_obs_cloud.xlsx")
#samples_w_disagreement = cloud_labels_w_consensus_calc[cloud_labels_w_consensus_calc["obs_cloud_binary_consensus"]==0]
#samples_w_disagreement.to_excel("SamplesWithDisconsensusOnCloud.xlsx")

samples_w_disagreement = pd.read_excel("SamplesWithDisconsensusOnCloud.xlsx")
image_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2_UINT8_Plt/"

for image_name in samples_w_disagreement["image_name"]:
    image = cv2.imread(image_path + image_name, -1)
    plt.imshow(image)
    plt.title(image_name)
    plt.show()

samples_w_disagreement = pd.read_excel("SamplesWithDisconsensusOnCloud_ReviewedForHaze.xlsx")
print(samples_w_disagreement["Hazy?"].value_counts())
