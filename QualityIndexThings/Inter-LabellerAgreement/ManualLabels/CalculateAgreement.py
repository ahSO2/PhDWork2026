import numpy as np
import pandas as pd

merged_labels = pd.read_excel("MergedLabels/merged_precipitation.xlsx")
target = "precipitation"
labellers = ["AH", "TW", "TP"]

def map_level_to_binary(level):
    if level in ["No", "Minor"]:
        return 0
    elif level in ["Not Calc", "In Calc", "Very"]:
        return 1
    else:
        return "Error in label value."

def map_level_to_numeric(level):
    mapping = {"No":0,
               "Minor":1.5,
               "Not Calc":3,
               "In Calc":4,
               "Very":5}
    return mapping[level]

def map_numeric_to_level(value):
    back_mapping = {0:"No",
               1:"Minor",
               2:"Minor",
               3:"Not Calc",
               4:"In Calc",
               5:"Very"}
    return back_mapping[value]
def calculate_binary_agreement(row):
    binary_labels = []
    for labeller in labellers:
        binary_labels.append(row[target + "_binary_" + labeller])
    unique_labels = set(binary_labels)
    if len(unique_labels) == 1:
        return 1
    else:
        return 0

def calculate_majority_vote_binary(row):
    binary_labels = []
    for labeller in labellers:
        binary_labels.append(row[target + "_binary_" + labeller])
    positive_votes = binary_labels.count(1)
    negative_votes = binary_labels.count(0)
    if positive_votes > negative_votes:
        return 1
    elif negative_votes > positive_votes:
        return 0
    else:
        print("Error in majority vote!")

def calculate_target_level_numeric_majority_vote(row):
    numeric_labels = []
    #text_labels = []
    for labeller in labellers:
        numeric_labels.append(row[target + "_level_numeric_" + labeller])
        #text_labels.append(row[target + "_level_" + labeller])
    #If two labellers agree, take that value
    #Otherwise, take a mean of all three numeric values
    majority_vote_found = False
    for level in [0, 1.5, 3, 4, 5]:
        level_count = numeric_labels.count(level)
        if level_count >= 2:
            majority_vote_found = True
            return int(np.round(level, 0)) #Rounding so that "Minor" value of 1.5 is mapped to an integer
    if majority_vote_found == False:
        return int(np.round(np.mean(np.array(numeric_labels)),0))




for labeller in labellers:
    merged_labels[target + "_binary_" + labeller] = merged_labels[target + "_level_" + labeller].apply(map_level_to_binary)
    merged_labels[target + "_level_numeric_" + labeller] = merged_labels[target + "_level_" + labeller].apply(map_level_to_numeric)

merged_labels[target + "_binary_consensus"] = merged_labels.apply(calculate_binary_agreement, axis=1)
merged_labels[target + "_binary_majority_vote"] = merged_labels.apply(calculate_majority_vote_binary, axis=1)
merged_labels[target + "_level_numeric_majority_vote"] = merged_labels.apply(calculate_target_level_numeric_majority_vote, axis=1)
merged_labels[target + "_level_majority_vote"] = merged_labels[target + "_level_numeric_majority_vote"].apply(map_numeric_to_level)

merged_labels.to_excel("MergedLabels/consensus_calc_" + target + ".xlsx")