#Read in a dataframe holding model predictions and quality index values
#Calculate the balanced accuracy for the models predicting obcurance, and
#for the Delle Donne indexes
#Calculate other metrics: precision and recall, and F1-Score
#Run a McNemar Test

import numpy as np
import pandas as pd
from scipy import stats

def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    elif value == "No":
        return 0
    else:
        print("Error in column value: Expected 'Yes'/'No'")
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

    binary_precisions = []
    for binary_class in [0, 1]:
        #Calculate the precision
        predicted_this_class = outputs[outputs["predicted"] == binary_class]
        if predicted_this_class.shape[0] > 0:
            true_predictions = predicted_this_class[predicted_this_class["targets"] == binary_class]
            binary_precisions.append(np.round(true_predictions.shape[0]/predicted_this_class.shape[0],4))
        else:
            binary_precisions.append(np.nan)

    return binary_precisions, subclass_recalls

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
def calculate_overall_obs_prediction_CNNs(row):
    precip_sigmoid = row["precipitation_prediction"]
    cloud_sigmoid = row["obs_cloud_prediction"]
    if precip_sigmoid>=0.5 or cloud_sigmoid>=0.5:
        return 1
    else:
        return 0

class DD_index_obs_prediction_calculator():
    def __init__(self, visibility_thresh, correlation_thresh):
        self.thresh_v = visibility_thresh
        self.thresh_c = correlation_thresh

    def calculate_overall_obs_prediction_DD(self, row):
        index_v = row["visibility_index"]
        index_c = row["correlation_index"]
        if index_v <= self.thresh_v or index_c >= self.thresh_c:
            return 1
        else:
            return 0

def map_correctness_to_McNemar_vals(row):
    CNN_correct = row["is_correct_CNN"]
    DD_correct = row["is_correct_DD"]
    if CNN_correct == True and DD_correct == True:
        return "a"
    elif CNN_correct == False and DD_correct == True:
        return "b"
    elif CNN_correct == True and DD_correct == False:
        return "c"
    elif CNN_correct == False and DD_correct == False:
        return "d"

indexes_df_path = "IndexValues/Lastarria_AllSamples_Unbalanced_QualityIndexes.xlsx"
DD_thresh_v = 3.1
DD_thresh_v_OG = 4
DD_thresh_c = -0.4
DD_thresh_c_OG = -0.5
results_folder = "EvalAndCompareResults/"
np.random.seed(42) #For reproducibility of the bootstrap CIs

if ".csv" in indexes_df_path:
    indexes_df = pd.read_csv(indexes_df_path)
else:
    indexes_df = pd.read_excel(indexes_df_path)
results_df = pd.DataFrame(columns=["Predictor", "BACC", "BACC_L95", "BACC_U95",
                                   "PP", "PN", "R_No", "R_Minor", "R_NotCalc", "R_InCalc", "R_Very",
                                   "F1"])
indexes_df["obscurance_binary"] = indexes_df["obscurance"].apply(map_YN_to_binary)

#Calculate balanced accuracy with bootstrap CI for the CNN Models:
indexes_df["obscurance_predicted_CNNs"] = indexes_df.apply(calculate_overall_obs_prediction_CNNs, axis=1)
model_bacc, model_bacc_l95, model_bacc_u95 = balanced_accuracy_w_bootstrapCI(indexes_df["obscurance_binary"], indexes_df["obscurance_predicted_CNNs"])
model_binary_precisions, model_subclass_recalls = precision_recall_per_class(target=indexes_df["obscurance_binary"], predicted=indexes_df["obscurance_predicted_CNNs"], target_level=indexes_df["obscurance_level"])
model_f1 = F1_Score(target=indexes_df["obscurance_binary"], predicted=indexes_df["obscurance_predicted_CNNs"])
results_df.loc[0] = {"Predictor":"CNNs",
                     "BACC":model_bacc, "BACC_L95":model_bacc_l95, "BACC_U95":model_bacc_u95,
                     "PP": model_binary_precisions[1],"PN": model_binary_precisions[0],
                     "R_No": model_subclass_recalls[0], "R_Minor": model_subclass_recalls[1], "R_NotCalc": model_subclass_recalls[2], "R_InCalc": model_subclass_recalls[3], "R_Very": model_subclass_recalls[4],
                     "F1": model_f1,
                     }

#Calculate balanced accuracy with bootstrap CI for the selected DD index thresholds
calculator = DD_index_obs_prediction_calculator(DD_thresh_v, DD_thresh_c)
indexes_df["obscurance_predicted_DD"] = indexes_df.apply(calculator.calculate_overall_obs_prediction_DD, axis=1)
DD_bacc, DD_bacc_l95, DD_bacc_u95 = balanced_accuracy_w_bootstrapCI(indexes_df["obscurance_binary"], indexes_df["obscurance_predicted_DD"])
DD_binary_precisions, DD_subclass_recalls = precision_recall_per_class(target=indexes_df["obscurance_binary"], predicted=indexes_df["obscurance_predicted_DD"], target_level=indexes_df["obscurance_level"])
DD_f1 = F1_Score(target=indexes_df["obscurance_binary"], predicted=indexes_df["obscurance_predicted_DD"])
results_df.loc[1] = {"Predictor":"DDIs",
                     "BACC":DD_bacc, "BACC_L95":DD_bacc_l95, "BACC_U95":DD_bacc_u95,
                     "PP": DD_binary_precisions[1], "PN": DD_binary_precisions[0],
                     "R_No": DD_subclass_recalls[0], "R_Minor": DD_subclass_recalls[1],
                     "R_NotCalc": DD_subclass_recalls[2], "R_InCalc": DD_subclass_recalls[3],
                     "R_Very": DD_subclass_recalls[4],
                     "F1": DD_f1,
                     }

#Calculate balanced accuracy with bootstrap CI for the original
calculator_OG = DD_index_obs_prediction_calculator(DD_thresh_v_OG, DD_thresh_c_OG)
indexes_df["obscurance_predicted_DD_OGthresh"] = indexes_df.apply(calculator_OG.calculate_overall_obs_prediction_DD, axis=1)
DDOG_bacc, DDOG_bacc_l95, DDOG_bacc_u95 = balanced_accuracy_w_bootstrapCI(indexes_df["obscurance_binary"], indexes_df["obscurance_predicted_DD_OGthresh"])
DDOG_binary_precisions, DDOG_subclass_recalls = precision_recall_per_class(target=indexes_df["obscurance_binary"], predicted=indexes_df["obscurance_predicted_DD_OGthresh"], target_level=indexes_df["obscurance_level"])
DDOG_f1 = F1_Score(target=indexes_df["obscurance_binary"], predicted=indexes_df["obscurance_predicted_DD_OGthresh"])
results_df.loc[2] = {"Predictor":"DDIs_OGthresh",
                     "BACC":DDOG_bacc, "BACC_L95":DDOG_bacc_l95, "BACC_U95":DDOG_bacc_u95,
                     "PP": DDOG_binary_precisions[1], "PN": DDOG_binary_precisions[0],
                     "R_No": DDOG_subclass_recalls[0], "R_Minor": DDOG_subclass_recalls[1],
                     "R_NotCalc": DDOG_subclass_recalls[2], "R_InCalc": DDOG_subclass_recalls[3],
                     "R_Very": DDOG_subclass_recalls[4],
                     "F1": DDOG_f1,
                     }

if ".csv" in indexes_df_path:
    results_df.to_csv(results_folder + "/" + "CNNwDDComparison_" + indexes_df_path.split("/")[-1])
else:
    results_df.to_csv(results_folder + "/" + "CNNwDDComparison_" + indexes_df_path.split("/")[-1][:-5] + ".csv")

#McNemar Test:
#For each observation, count whether:
#Both predictors are correct (a)
#Only Delle Donne Indexes predict correctly (b)
#Only CNNs predict correctly (c)
#Neither predictor is correct (d)

indexes_df["is_correct_CNN"] = indexes_df["obscurance_predicted_CNNs"] == indexes_df["obscurance_binary"]
indexes_df["is_correct_DD"] = indexes_df["obscurance_predicted_DD"] == indexes_df["obscurance_binary"]
indexes_df["McNemar_value"] = indexes_df.apply(map_correctness_to_McNemar_vals, axis=1)

print("McNemar Test Results: ")
b = indexes_df.value_counts(["McNemar_value"])["b"]
c = indexes_df.value_counts(["McNemar_value"])["c"]
chi_squ = (b-c)**2/(b+c)
print("Chi-Squ statistic: " + str(chi_squ))
print("b plus c: " + str(b+c))

p_val = 1 - stats.chi2.cdf(chi_squ, 1)
print("p-value: " + str(p_val))

