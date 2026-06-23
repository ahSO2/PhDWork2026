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

old_code_oldDFs = pd.read_excel("OutputsDFs/OldModelAppCode_FinalTrain_OGSet_OGDataPathPredictions.xlsx") #The original model application code, and original data source folders
newcode_newDFs = pd.read_excel("OutputsDFs/NewModelAppCode_UpdatedORDADataPaths.xlsx") #The new model application code, and updated data source folders for ORDA upload

precip_match = old_code_oldDFs["rain_or_dirt"] == newcode_newDFs["rain_or_dirt"]
prediction_match = old_code_oldDFs["model_predictions"] == newcode_newDFs["precipitation_prediction"]

newcode_newDFs["precip_match"] = precip_match
newcode_newDFs["prediction_match"] = prediction_match

mismatches = newcode_newDFs[newcode_newDFs["prediction_match"]==False]
print(mismatches.value_counts(["volcano_name"]))

#updated_values.to_excel("Mismatch log.xlsx")

#path1 = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2/CotopaxiView4_2023-12-29T214145_fltrA_1ag_599983ss_Plume.png"
#path2 = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data/Precipitation Full Split - Seen Locations/CotopaxiView4_2023-12-29T214145_fltrA_1ag_599983ss_Plume.png" #Folder storing image data
#img1 = cv2.imread(path1, -1)
#img2 = cv2.imread(path2, -1)
#print(np.array_equal(img1, img2))


