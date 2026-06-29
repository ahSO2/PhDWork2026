#Read in all the predictions from the folder of full days
#Calculate binary predictions given a threshold for each index
#Calculate metrics incl. balanced accuracy for precip, cloud, and overall_obs prediction
#Calculate the precision and recall of good quality data
import os

import pandas as pd

folder_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs/FullDays"
precip_threshold = 0.5
cloud_threshold = 0.5

def map_binary_predictcions_to_obs_prediction(row):
    precip_prediction = row["precipitation_prediction_binary"]
    cloud_prediction = row["obs_cloud_prediction_binary"]
    if precip_prediction == 1 or cloud_prediction == 1:
        return 1
    else:
        return 0

def map_level_to_numeric(level):
    if level == "No":
        return 0
    elif level == "Minor":
        return 1
    elif level == "Not Calc":
        return 2
    elif level == "In Calc":
        return 3
    elif level == "Very":
        return 4
    else:
        print("Error in level value.")
def calculate_true_obs_level_numeric(row):
    precip_level = row["precipitation_level_numeric"]
    cloud_level = row["obs_cloud_level_numeric"]
    return max[precip_level, cloud_level]

day_index = 0
for day_folder in os.listdir(folder_path):
    print("Reading predictions for day: " + day_folder)
    if "ImageNamesSorted.csv" in os.listdir(folder_path + "/" + day_folder):
        day_predictions = pd.read_csv(folder_path + "/" + day_folder + "/ImageNamesSorted.csv")
    else:
        day_predictions = pd.read_csv(folder_path + "/" + day_folder + "/ImageNames.csv")
    if day_index == 0:
        predictions_df = day_predictions
    else:
        predictions_df = pd.concat([predictions_df, day_predictions])
    day_index += 1

#Threshold to make binary predictions
predictions_df["precipitation_prediction_binary"] = np.where(predictions_df["precipitation_prediction"] >= precip_threshold, 1, 0)
predictions_df["obs_cloud_prediction_binary"] = np.where(predictions_df["obs_cloud_prediction"] >= cloud_threshold, 1, 0)
predictions_df["obscurance_prediction_binary"] = predictions_df.apply(map_binary_predictcions_to_obs_prediction, axis=1)

#Calculate metrics -------------
#Calculate the true overall obscurance level and binary value
predictions_df["precipitation_level_numeric"] = predictions_df["precipitation_level"].apply(map_level_to_numeric)
predictions_df["cloud_level_numeric"] = predictions_df["cloud_level"].apply(map_level_to_numeric)
predictions_df["obscurance_level_numeric"] = predictions_df.apply(calculate_true_obs_level_numeric, axis=1)
