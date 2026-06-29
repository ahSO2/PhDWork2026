import pandas as pd

merged_labels = pd.read_excel("MergedLabels/merged_obs_cloud.xlsx")
target = "obs_cloud"
labellers = ["AH", "TW", "TP"]

def map_level_to_binary(level):
    if level in ["No", "Minor"]:
        return 0
    elif level in ["Not Calc", "In Calc", "Very"]:
        return 1
    else:
        return "Error in label value."

def calculate_binary_agreement(row):
    binary_labels = []
    for labeller in labellers:
        binary_labels.append(row[target + "_binary_" + labeller])
    unique_labels = set(binary_labels)
    if len(unique_labels) == 1:
        return 1
    else:
        return 0

for labeller in labellers:
    merged_labels[target + "_binary_" + labeller] = merged_labels[target + "_level_" + labeller].apply(map_level_to_binary)

merged_labels[target + "_binary_consensus"] = merged_labels.apply(calculate_binary_agreement, axis=1)

merged_labels.to_excel("MergedLabels/consensus_calc_" + target + ".xlsx")