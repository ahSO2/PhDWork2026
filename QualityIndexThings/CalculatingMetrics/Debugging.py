#The result of applying the final model to the precipitation
#train set (expanded) using my original conde (99% acc) is
#different than the evaluation I get with my new code (96% acc).
#The mismatch must be with the values in the sheet (as I still get
#99% if I read the sheet into my new "CalculateMetrics.py" script.
#The other evaluations for other sets seem to match fine.
import cv2
import numpy as np
###SOLVED - It was that I had initially run evaluations with the spreadsheet of image
#labels which hadn't had the corrections (labelling mistakes I found during all evaluations)
#applied.


import pandas as pd

original_values = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/Final Models Evaluation/OnLens/FinalEvalOutputs/OnLensSet_FinalTrain_BeforeLabelUpdatesPredictions.xlsx") #Predictions made with original code
updated_values = pd.read_excel("FinalModelsApplicationOutputs/ExpandedOnLensExpmtSeenLocationsTrainSet..xlsx") #Predictions made with updated model applic code (but the uncorrected samples dataframe, the same as the original code applic version)
#updated_values = pd.read_excel("FinalModelsApplicationOutputs/Precipitation_Full_TrainExpanded.xlsx")

print(original_values.shape[0])
print(updated_values.shape[0])

precip_match = original_values["rain_or_dirt"] == updated_values["precipitation"]
prediction_match = original_values["model_predictions"] == updated_values["precipitation_prediction"]

updated_values["precip_match"] = precip_match
updated_values["prediction_match"] = prediction_match

mismatches = updated_values[updated_values["prediction_match"]==False]
print(mismatches.value_counts(["volcano_name"]))

#updated_values.to_excel("Mismatch log.xlsx")

path1 = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2/CotopaxiView4_2023-12-29T214145_fltrA_1ag_599983ss_Plume.png"
path2 = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data/Precipitation Full Split - Seen Locations/CotopaxiView4_2023-12-29T214145_fltrA_1ag_599983ss_Plume.png" #Folder storing image data
img1 = cv2.imread(path1, -1)
img2 = cv2.imread(path2, -1)
print(np.array_equal(img1, img2))


