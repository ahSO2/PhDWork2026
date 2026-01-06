#Read in the overall train split and use this to create the
#cross-validation splits.

import pandas as pd

locations = ["Cotopaxi", "Kilauea", "Merapi", "Lascar", "Reventador"]

'''
train = pd.read_excel("UpdatedTVTSplits/FinalSplit/Train.xlsx")

for location in locations:
    print(location + " Left out:")
    location_data = train[train["volcano_name"]==location]
    seen_locations = train[train["volcano_name"]!=location]
    location_data.to_excel("UpdatedTVTSplits/CrossValidationSplits/" + location + "_UnseenTest.xlsx")
    seen_locations.to_excel("UpdatedTVTSplits/CrossValidationSplits/" + location + "LeftOut_TrainAndValid.xlsx")
    print(location_data.value_counts(["volcano_name", "overall_obs"]))
'''

unseen_location = "Reventador"
set_to_split = pd.read_excel("UpdatedTVTSplits/CrossValidationSplits/" + unseen_location + "LeftOut_TrainAndValid.xlsx")

groups_for_valid = [7, 47, 53, 61, 86, 75, 70, 66, 121, 133, 132, 118]

valid = set_to_split[set_to_split["stratification_group"].isin(groups_for_valid)]
train = set_to_split[~(set_to_split["stratification_group"].isin(groups_for_valid))]

valid.to_excel("UpdatedTVTSplits/CrossValidationSplits/" + unseen_location + "LeftOut_Valid.xlsx")
train.to_excel("UpdatedTVTSplits/CrossValidationSplits/" + unseen_location + "LeftOut_Train.xlsx")

#Check no overlap:
valid_img_names = set(valid["image_name"].tolist())
train_img_names = set(train["image_name"].tolist())
print(valid_img_names.intersection(train_img_names))

print(valid.value_counts(["volcano_name", "overall_obs"]))
print(train.value_counts(["volcano_name", "overall_obs"]))