import pandas as pd
import numpy as np

def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    elif value == "No":
        return 0
    else:
        print("Error in column value: Expected 'Yes'/'No'")

def threshold_to_binary(column, threshold):
    thresholded = np.where(column >= threshold, 1, 0)
    return thresholded

def accuracy_w_bootstrapCI(target, predicted):

    #Calculate accuracy and confidence interval 


results_dataframe_path = "FinalModelsApplicationOutputs/Precipitation_Full_Valid.xlsx"

results_df = pd.read_excel(results_dataframe_path)

predict = "precipitation"
#predict = "cloud"

if predict == "precipitation":
    target_YN = results_df["precipitation"]
    target_binary = target_YN.apply(map_YN_to_binary).to_numpy()
    prediction_sigmoid = results_df["precipitation_prediction"].to_numpy()

#Threshold at 0.5 and calculate the metrics which use this threshold
