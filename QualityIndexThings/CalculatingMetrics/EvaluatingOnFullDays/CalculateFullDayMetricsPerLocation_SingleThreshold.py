import matplotlib.pyplot as plt
#TODO Code used to run evals where either precip, cloud or overall obs
#is evaluated using a given threshold (there is a separate script for if you
#want to set one threshold for precip and one for cloud as the calculations
# need to be organised slightly differently e.g. we can't calculate area under
# a PR curve as we have two different threshold values).

import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

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

    correct = np.where(target == predicted, 1, 0)

    accuracy = np.mean(correct)
    bootstrap_means = []

    #Bootstrap CI
    for b in range(0, 1000):
        selection = np.random.choice(correct, correct.shape[0], replace=True)
        bootstrap_means.append(np.mean(selection))

    lower = np.percentile(bootstrap_means, q=2.5)
    upper = np.percentile(bootstrap_means, q=97.5)
    return np.round(accuracy, 4), np.round(lower, 4), np.round(upper, 4)

def balanced_accuracy(dataframe):
    '''Calculate balanced accuracy, assuming a binary target.'''
    #Check there are actually two classes
    classes = dataframe["targets"].unique()
    if classes.shape[0] == 1: #In case this sample only contains one class
        return np.mean(dataframe["correct"])
    else:
        class_0_samples = dataframe[dataframe["targets"]==0]
        class_1_samples = dataframe[dataframe["targets"]==1]
        class_0_acc = np.mean(class_0_samples["correct"])
        class_1_acc = np.mean(class_1_samples["correct"])
        return (class_0_acc + class_1_acc)/2


def balanced_accuracy_w_bootstrapCI(target, predicted):
    outputs = pd.DataFrame()
    outputs["targets"] = target
    outputs["predicted"] = predicted
    outputs["correct"] = np.where(target == predicted, 1, 0)

    bal_acc = balanced_accuracy(outputs)
    bootstrap_accs = []

    for b in range(0, 1000):
        #Select a sample, with replacement (uses np.random.seed)
        selection = outputs.sample(n = outputs.shape[0], replace=True)
        #Calculate balanced accuracy
        bootstrap_accs.append(balanced_accuracy(selection))

    lower = np.percentile(bootstrap_accs, q=2.5)
    upper = np.percentile(bootstrap_accs, q=97.5)
    return np.round(bal_acc, 4), np.round(lower, 4), np.round(upper, 4)

def precision_recall_per_class(target, predicted, target_level):
    '''Calculate for each  subclass the precision and recall.'''
    classes = {"No":0, "Minor":0, "Not Calc":1, "In Calc":1, "Very":1}

    outputs = pd.DataFrame()
    outputs["targets"] = target
    outputs["predicted"] = predicted
    outputs["target_level"] = target_level

    subclass_recalls = []
    for subclass in classes.keys():
        class_samples = outputs[outputs["target_level"] == subclass]
        if class_samples.shape[0] > 0: #If we have samples of this class in the set
            #Calculate recall
            target_val = classes[subclass]
            correctly_predicted_as_target_val = class_samples[class_samples["predicted"]==target_val]
            r = correctly_predicted_as_target_val.shape[0]/class_samples.shape[0]
            subclass_recalls.append(np.round(r, 4))
        else: #If there are no samples of this class
            subclass_recalls.append(np.nan)

    binary_recalls = []
    for bin_class in [0, 1]:
        class_samples = outputs[outputs["targets"] == bin_class]
        if class_samples.shape[0] > 0:  # If we have samples of this class in the set
            # Calculate recall
            correctly_predicted_as_target_val = class_samples[class_samples["predicted"] == bin_class]
            r = correctly_predicted_as_target_val.shape[0] / class_samples.shape[0]
            binary_recalls.append(np.round(r, 4))
        else: #If there are no samples of this class
            binary_recalls.append(np.nan)

    binary_precisions = []
    for binary_class in [0, 1]:
        #Calculate the precision
        predicted_this_class = outputs[outputs["predicted"] == binary_class]
        if predicted_this_class.shape[0] > 0:
            true_predictions = predicted_this_class[predicted_this_class["targets"] == binary_class]
            binary_precisions.append(np.round(true_predictions.shape[0]/predicted_this_class.shape[0],4))
        else:
            binary_precisions.append(np.nan)

    return binary_precisions, subclass_recalls, binary_recalls

def F1_Score(target, predicted):
    '''Calculate the F1 score (considering the 0 "unobscured" class as positive).'''
    outputs = pd.DataFrame()
    outputs["targets"] = target
    outputs["predicted"] = predicted

    return_nan = False

    #Calculate precision
    predicted_unobs = outputs[outputs["predicted"]==0]
    correctly_predicted_unobs = predicted_unobs[predicted_unobs["targets"]==0]
    if predicted_unobs.shape[0] == 0:
        p = 1
    else:
        p = correctly_predicted_unobs.shape[0]/predicted_unobs.shape[0]

    #Calculate recall
    true_unobs = outputs[outputs["targets"]==0]
    if true_unobs.shape[0] == 0:
        return_nan = True
        r = 0
    else:
        r = correctly_predicted_unobs.shape[0]/true_unobs.shape[0]

    if p + r == 0 or return_nan == True:
        return np.nan
    else:
        f1 = 2 * (p*r)/(p+r)
        return np.round(f1, 4)

def AUC_PR(target, predicted_sigmoid):
    '''Considering the 0=unobscured class as positive.'''
    print(np.unique(target))
    predicted_sigmoid_inverse = 1 - predicted_sigmoid
    target_inverse = 1 - target
    precisions, recalls, thresholds = precision_recall_curve(y_true=target_inverse, y_score=predicted_sigmoid_inverse)
    auc_val = auc(recalls, precisions)
    return np.round(auc_val, 4), np.round(precisions, 4), np.round(recalls,4), thresholds

def location_balanced_BA_wBootstrapCI(target, predicted, image_names, locations):
    '''Calculate the mean class-balanced accuracy over all locations.
    With bootstrap confidence interval '''
    outputs = pd.DataFrame()
    outputs["targets"] = target
    outputs["predicted"] = predicted
    outputs["correct"] = np.where(target == predicted, 1, 0)
    outputs["image_name"] = image_names

    location_BAs = []
    for location in locations:
        location_data = outputs[outputs["image_name"].str.contains(location)]
        if location_data.shape[0] != 0:
            location_BA = balanced_accuracy(location_data)
            location_BAs.append(location_BA)
    location_balanced_BA = np.mean(location_BAs)

    #Sample randomly with replacement, any number from each location
    #Calculate the balanced accuracy for each location subset
    #Calculate the mean

    bootstrap_meanBACCs = []

    for b in range(0, 1000):
        print(b)
        # Select a sample, with replacement (uses np.random.seed)
        selection = outputs.sample(n=outputs.shape[0], replace=True)
        # Calculate balanced accuracy

        bs_location_BAs = []
        for location in locations:
            location_data = selection[selection["image_name"].str.contains(location)]
            if location_data.shape[0] != 0:
                bs_location_BA = balanced_accuracy(location_data)
                bs_location_BAs.append(bs_location_BA)
        bootstrap_meanBACCs.append(np.mean(bs_location_BAs))

    lower = np.percentile(bootstrap_meanBACCs, q=2.5)
    upper = np.percentile(bootstrap_meanBACCs, q=97.5)

    return np.round(location_balanced_BA, 4), np.round(lower, 4), np.round(upper, 4)

def map_level_to_numeric(level):
    forward_mapping = {"No":0,
                       "Minor":1,
                       "Not Calc":2,
                       "In Calc":3,
                       "Very":4}
    return forward_mapping[level]

def map_numeric_to_level(int_val):
    backward_mapping = {0:"No",
                        1:"Minor",
                        2:"Not Calc",
                        3:"In Calc",
                        4:"Very"}
    return backward_mapping[int_val]

def map_level_to_YN(level):
    to_YN_mapping = {"No":"No",
                     "Minor":"No",
                     "Not Calc":"Yes",
                     "In Calc":"Yes",
                     "Very":"Yes"}
    return to_YN_mapping[level]

'''Random seed is set for reproducibility of bootstrapping. 
All final values are rounded to 4dp.'''

np.random.seed(42)
full_days_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/FullDays/"
predict = "precipitation"
predict = "obs_cloud"
#predict = "obscurance"
threshold = 0.9372
locations = ["Cotopaxi", "Kilauea", "Lastarria", "Merapi", "Reventador"]
outputs_save_path = "FullDays_" + predict + "_EvalByLocation_Threshold_" + str(threshold) + ".xlsx"

##########################################################################
outputs_df = pd.DataFrame(columns=["location", "threshold", "ACC", "ACC_L95", "ACC_U95",
                                   "BACC", "BACC_L95", "BACC_U95",
                                   "PP", "PN", "RP", "RN", "R_No", "R_Minor",
                                   "R_NotCalc", "R_InCalc", "R_Very",
                                   "F1", "AUC_PR"])
full_days_dfs = []
for sheet_name in os.listdir(full_days_path):
    print(sheet_name)
    day_data = pd.read_excel(full_days_path + sheet_name)
    full_days_dfs.append(day_data)

all_data = pd.concat(full_days_dfs)
print(all_data.shape)
all_data = all_data[all_data["other"]!="Yes"]
print(all_data.shape)
all_data.reset_index(inplace=True)



if predict == "obscurance":
    #Calculate the obscurance level (true)
    all_data["precipitation_level_numeric"] = all_data["precipitation_level"].apply(map_level_to_numeric)
    all_data["obs_cloud_level_numeric"] = all_data["obs_cloud_level"].apply(map_level_to_numeric)
    all_data["obscurance_level_numeric"] = all_data[["precipitation_level_numeric", "obs_cloud_level_numeric"]].max(axis=1)
    all_data["obscurance_level"] = all_data["obscurance_level_numeric"].apply(map_numeric_to_level)
    #Calculate the obscurance YN (true)
    all_data["obscurance"] = all_data["obscurance_level"].apply(map_level_to_YN)
    #Calculate the obscurance prediction (max of cloud and precip sigmoid values)
    all_data["obscurance_prediction"] = all_data[["precipitation_prediction", "obs_cloud_prediction"]].max(axis=1)

for location in locations:
    location_data = all_data[all_data["image_name"].str.contains(location)]
    if location_data.shape[0] != 0:
        target_YN = location_data[predict]
        target_level = location_data[predict + "_level"].to_numpy()
        target_binary = target_YN.apply(map_YN_to_binary).to_numpy()
        prediction_sigmoid = location_data[predict + "_prediction"].to_numpy()

        #Threshold and calculate the metrics which use this threshold
        prediction_thresh = threshold_to_binary(prediction_sigmoid, threshold)
        acc, acc_l95, acc_u95 = accuracy_w_bootstrapCI(target_binary, prediction_thresh)
        bacc, bacc_l95, bacc_u95 = balanced_accuracy_w_bootstrapCI(target_binary, prediction_thresh)
        binary_precisions, subclass_recalls, binary_recalls = precision_recall_per_class(target_binary, prediction_thresh, target_level)
        f1 = F1_Score(target_binary, prediction_thresh)
        auc_pr, precisions, recalls, thresholds = AUC_PR(target=target_binary, predicted_sigmoid=prediction_sigmoid)

        #For the given dataset - save all these metrics:
        new_row = {"location":location,
               "threshold":threshold,
               "ACC":acc,
               "ACC_L95":acc_l95,
               "ACC_U95":acc_u95,
               "BACC":bacc,
               "BACC_L95":bacc_l95,
               "BACC_U95":bacc_u95,
               "PP":binary_precisions[1],
               "PN":binary_precisions[0],
               "RP":binary_recalls[1],
               "RN":binary_recalls[0],
               "R_No":subclass_recalls[0],
               "R_Minor":subclass_recalls[1],
               "R_NotCalc":subclass_recalls[2],
               "R_InCalc":subclass_recalls[3],
               "R_Very":subclass_recalls[4],
               "F1":f1,
               "AUC_PR":auc_pr}
        outputs_df.loc[len(outputs_df)] = new_row

#######Now additionally calculate metrics balanced by (taking the mean over) locations

#Calculate location-balanced bootstrap CI
target_YN = all_data[predict]
target_level = all_data[predict + "_level"]
target_binary = target_YN.apply(map_YN_to_binary).to_numpy()
prediction_sigmoid = all_data[predict + "_prediction"].to_numpy()
prediction_thresh = threshold_to_binary(prediction_sigmoid, threshold)
location_balanced_BA, LBBACC_Lower, LBBACC_Upper = location_balanced_BA_wBootstrapCI(target_binary, prediction_thresh, all_data["image_name"], locations)

#For other metrics, just take the mean over the location rows
location_means_row = {"location":"Location_Means",
                      "threshold":threshold,
                      "BACC":location_balanced_BA,
                      "BACC_L95":LBBACC_Lower,
                      "BACC_U95":LBBACC_Upper,
                      "PP": np.round(outputs_df["PP"].mean(), 4),
                      "PN": np.round(outputs_df["PN"].mean(), 4),
                      "RP": np.round(outputs_df["RP"].mean(), 4),
                      "RN": np.round(outputs_df["RN"].mean(), 4),
                      "R_No": np.round(outputs_df["RN"].mean(), 4),
                      "R_Minor": np.round(outputs_df["R_Minor"].mean(), 4),
                      "R_NotCalc": np.round(outputs_df["R_NotCalc"].mean(), 4),
                      "R_InCalc": np.round(outputs_df["R_InCalc"].mean(), 4),
                      "R_Very": np.round(outputs_df["R_Very"].mean(), 4),
                      "F1": np.round(outputs_df["F1"].mean(), 4),
                      "AUC_PR":np.round(outputs_df["AUC_PR"].mean(), 4)
                      }
outputs_df.loc[len(outputs_df)] = location_means_row
outputs_df.to_excel(outputs_save_path)
#all_data.to_csv("FullDays_StdThreshold_OverallObsPredictions.csv")