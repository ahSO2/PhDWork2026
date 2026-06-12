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

    correct = np.where(target == predicted, 1, 0)

    accuracy = np.mean(correct)
    bootstrap_means = []

    #Bootstrap CI
    for b in range(0, 1000):
        selection = np.random.choice(correct, correct.shape[0], replace=True)
        bootstrap_means.append(np.mean(selection))

    lower = np.percentile(bootstrap_means, q=5)
    upper = np.percentile(bootstrap_means, q=95)
    return np.round(accuracy, 4), np.round(lower, 4), np.round(upper, 4)

def balanced_accuracy(dataframe):
    '''Calculate balanced accuracy, assuming a binary target.'''
    #Check there are actually two classes
    classes = dataframe["targets"].unique()
    if classes.shape[0] == 1: #In case this sample only contains one class
        return np.mean(classes)
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

    lower = np.percentile(bootstrap_accs, q=5)
    upper = np.percentile(bootstrap_accs, q=95)
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

    #Calculate precision
    predicted_unobs = outputs[outputs["predicted"]==0]
    correctly_predicted_unobs = predicted_unobs[predicted_unobs["targets"]==0]
    p = correctly_predicted_unobs.shape[0]/predicted_unobs.shape[0]

    #Calculate recall
    true_unobs = outputs[outputs["targets"]==0]
    r = correctly_predicted_unobs.shape[0]/true_unobs.shape[0]

    f1 = 2 * (p*r)/(p+r)
    return np.round(f1, 4)


'''Random seed is set for reproducibility of bootstrapping. 
All final values are rounded to 4dp.'''

np.random.seed(42)
results_dataframe_path = "FinalModelsApplicationOutputs/Precipitation_Full_Valid.xlsx"
results_df = pd.read_excel(results_dataframe_path)

predict = "precipitation"
#predict = "obs_cloud"

target_YN = results_df[predict]
target_level = results_df[predict + "_level"]
target_binary = target_YN.apply(map_YN_to_binary).to_numpy()
prediction_sigmoid = results_df[predict + "_prediction"].to_numpy()

#Threshold at 0.5 and calculate the metrics which use this threshold
prediction_thresh = threshold_to_binary(prediction_sigmoid, 0.5)
#acc, l, u = accuracy_w_bootstrapCI(target_binary, prediction_thresh)
#bal_acc, l, u = balanced_accuracy_w_bootstrapCI(target_binary, prediction_thresh)
#binary_precisions, subclass_recalls = precision_recall_per_class(target_binary, prediction_thresh, target_level)
#f1 = F1_Score(target_binary, prediction_thresh)

#TODO adapt the threshold to calculate some form of curve