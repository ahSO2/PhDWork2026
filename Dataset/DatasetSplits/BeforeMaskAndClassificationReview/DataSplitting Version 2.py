import pandas as pd
#Split the good and low quality data at the same time
#to avoid ending up with data from same group but different
#quality levels in the same split.

'''
balanced_samples = pd.read_excel("TrainValidTestSplits/BalancedAndGrouped_AllLabelsWithoutLastarria.xlsx")
data_to_split = balanced_samples

#First split off a final test set:
groups_for_test = [1, 27, 18, 15, 13, 11, 21, 42, 29, 32, 35, 38, 46, 54, 57, 71, 63, 69, 66, 77, 74, 85, 89, 107, 104, 100, 101, 82, 80]

train_and_valid = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_test))]
train_and_valid.to_excel("TrainValidTestSplits/FinalSplit/TrainAndPreValid.xlsx")

test = data_to_split[data_to_split["stratification_group"].isin(groups_for_test)]
test.to_excel("TrainValidTestSplits/FinalSplit/FinalValid.xlsx")

print("TrainAndValid:")
print(train_and_valid.shape[0])
print(train_and_valid.value_counts(["volcano_name", "overall_quality"]))
print("Test:")
print(test.shape[0])
print(test.value_counts(["volcano_name", "overall_quality"]))
'''
'''
#Next split off the pre-valid set:
data_to_split = pd.read_excel("TrainValidTestSplits/FinalSplit/TrainAndPreValid.xlsx")

groups_for_valid = [33, 44, 45, 2, 12, 49, 59, 55, 73, 76, 79, 83, 86, 98]

pre_valid = data_to_split[data_to_split["stratification_group"].isin(groups_for_valid)]
pre_valid.to_excel("TrainValidTestSplits/FinalSplit/PreValid.xlsx")
train = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_valid))]
train.to_excel("TrainValidTestSplits/FinalSplit/Train.xlsx")
print("Train:")
print(train.shape[0])
print(train.value_counts(["volcano_name", "overall_quality"]))
print("Pre-Valid")
print(pre_valid.shape[0])
print(pre_valid.value_counts(["volcano_name", "overall_quality"]))
'''
'''
#Check there's no overlap of stratification groups
train = pd.read_excel("TrainValidTestSplits/FinalSplit/Train.xlsx")
valid = pd.read_excel("TrainValidTestSplits/FinalSplit/PreValid.xlsx")
test = pd.read_excel("TrainValidTestSplits/FinalSplit/FinalValid.xlsx")

train_groups = set(train["stratification_group"].tolist())
valid_groups = set(valid["stratification_group"].tolist())
test_groups = set(test["stratification_group"].tolist())

print(train_groups.intersection(valid_groups))
print(train_groups.intersection(test_groups))
print(test_groups.intersection(valid_groups))
'''

'''
#Create cross-validation sets from the training set:
locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador"]

data_pool= pd.read_excel("TrainValidTestSplits/FinalSplit/Train.xlsx")

for location in locations:
    cross_valid_data = data_pool[data_pool["volcano_name"] != location]
    location_data = data_pool[data_pool["volcano_name"]==location]
    cross_valid_data.to_excel("TrainValidTestSplits/CrossValidation/Without" + location + "/TrainAndValid.xlsx")
    location_data.to_excel("TrainValidTestSplits/CrossValidation/Without" + location + "/" + location + "Test.xlsx")
'''

location = "Lascar"
data_to_split = pd.read_excel("TrainValidTestSplits/CrossValidation/Without" + location + "/TrainAndValid.xlsx")
groups_for_valid = [87, 96, 106, 34, 48, 37, 40, 10, 16, 60, 67, 65, 50]

valid = data_to_split[data_to_split["stratification_group"].isin(groups_for_valid)]
valid.to_excel("TrainValidTestSplits/CrossValidation/Without" + location + "/Valid.xlsx")
train = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_valid))]
train.to_excel("TrainValidTestSplits/CrossValidation/Without" + location + "/Train.xlsx")
print("Train:")
print(train.shape[0])
print(train.value_counts(["volcano_name", "overall_quality"]))
print("Valid")
print(valid.shape[0])
print(valid.value_counts(["volcano_name", "overall_quality"], normalize=True))
print("Total in valid:")
print(valid.value_counts(["overall_quality"]))

