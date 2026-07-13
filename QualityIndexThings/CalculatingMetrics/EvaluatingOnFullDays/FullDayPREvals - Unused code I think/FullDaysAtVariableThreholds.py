import os
import numpy as np
import pandas as pd

full_days_path = "/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/FullDays/"
thresholds_to_consider = [0.5] #Threshold below which data is considered good quality
target_to_consider = "precipitation"


def threshold_to_binary(column, threshold):
    thresholded = np.where(column >= threshold, 1, 0)
    return thresholded

def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    elif value == "No":
        return 0
    else:
        print("Error in column value: Expected 'Yes'/'No'")

def precision_recall_negative_class(target, predicted):
    '''Calculate for the negative (=0) class of a binary variable the precision and recall.'''

    outputs = pd.DataFrame()
    outputs["targets"] = target
    outputs["predicted"] = predicted

    negative_samples = outputs[outputs["targets"] == 0]
    if negative_samples.shape[0] > 0: #If we have samples of this class in the set
        #Calculate recall
        correctly_predicted= negative_samples[negative_samples["predicted"]==0]
        r = correctly_predicted.shape[0]/negative_samples.shape[0]
    else: #If there are no samples of this class
        r = np.nan

    predicted_negative = outputs[outputs["predicted"] == 0]
    if predicted_negative.shape[0] > 0:
        true_predictions = predicted_negative[predicted_negative["targets"] == 0]
        p=true_predictions.shape[0]/predicted_negative.shape[0]
    else:
         p=np.nan

    return np.round(p, 4), np.round(r, 4)

full_days_dfs = []
for sheet_name in os.listdir(full_days_path):
    print(sheet_name)
    if "Lastarria" in sheet_name:
        day_data = pd.read_excel(full_days_path + sheet_name)
        full_days_dfs.append(day_data)

all_data = pd.concat(full_days_dfs)

#True binary labels
target_binary = all_data[target_to_consider].apply(map_YN_to_binary)
prediction_sigmoid = all_data[target_to_consider + "_prediction"].copy()
for threshold in thresholds_to_consider:
    #Thresholded prediction
    prediction_binary = threshold_to_binary(prediction_sigmoid, threshold)
    precision, recall = precision_recall_negative_class(target=target_binary, predicted=prediction_binary)
    print(precision)
    print(recall)

