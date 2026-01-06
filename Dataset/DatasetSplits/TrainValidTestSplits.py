import pandas as pd

#Split the good and low quality data at the same time
#to avoid ending up with data from same group but different
#quality levels in the same split.

updated_set = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList_UpdatedClassifications.xlsx")
data_to_split = updated_set

#Get a list of the updated group number of any observation that
#was in the original woCot Train split (used to explore movement filtering options)
wo_Cot_train = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/DatasetSplits/BeforeMaskAndClassificationReview/TrainValidTestSplits/CrossValidation/WithoutCotopaxi/Train.xlsx")
updated_set_names = updated_set["image_name"].tolist()
updated_set_groups = updated_set["stratification_group"].tolist()
groups_to_omit = []
for image_name in wo_Cot_train["image_name"].tolist():
    #Find the updated group number in the new stratification
    index_in_updated_set = updated_set_names.index(image_name)
    relevant_group_number = updated_set_groups[index_in_updated_set]
    #Add to a list to make sure I omit from new full split pre-valid and test sets
    groups_to_omit.append(relevant_group_number)

print("Groups to omit from final valid and test:")
groups_to_omit = list(set(groups_to_omit))
groups_to_omit.sort()
print(groups_to_omit)

'''
#Remove Lastarria samples to separate dataframe
lastarria_samples = data_to_split[data_to_split["volcano_name"]=="Lastarria"]
lastarria_samples.to_excel("UpdatedTVTSplits/FinalSplit/AllLastarriaSamples.xlsx")
data_to_split = data_to_split[data_to_split["volcano_name"]!="Lastarria"]

#First split off a final test set:
groups_for_test = [2, 3, 65, 69, 74, 78, 71, 82, 85, 38, 40, 42, 59, 55, 58, 64, 6, 116, 119, 122, 136, 134, 130, 9, 11, 15, 22, 25, 28, 18, 12, 36, 31]

train_and_valid = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_test))]
train_and_valid.to_excel("UpdatedTVTSplits/FinalSplit/TrainAndValid.xlsx")

test = data_to_split[data_to_split["stratification_group"].isin(groups_for_test)]
test.to_excel("UpdatedTVTSplits/FinalSplit/Test.xlsx")

print("TrainAndValid:")
print(train_and_valid.shape[0])
print(train_and_valid.value_counts(["volcano_name", "overall_obs"]))
print("Test:")
print(test.shape[0])
print(test.value_counts(["volcano_name", "overall_obs"]))

#Check there is no overlap of samples from the original
#woCot fold and the final test set.

wo_Cot_train_img_names = set(wo_Cot_train["image_name"].tolist())
final_test = pd.read_excel("UpdatedTVTSplits/FinalSplit/Test.xlsx")
test_img_names = set(final_test["image_name"].tolist())
overlap = wo_Cot_train_img_names.intersection(test_img_names)
print(overlap)
'''
'''
#Next do the overall train-valid split
train_and_valid = pd.read_excel("UpdatedTVTSplits/FinalSplit/TrainAndValid.xlsx")
data_to_split = train_and_valid

groups_for_valid = [5, 8, 14, 34, 37, 23, 30, 39, 48, 52, 54, 67, 73, 77, 81, 120, 129]

train = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_valid))]
train.to_excel("UpdatedTVTSplits/FinalSplit/Train.xlsx")

valid = data_to_split[data_to_split["stratification_group"].isin(groups_for_valid)]
valid.to_excel("UpdatedTVTSplits/FinalSplit/Valid.xlsx")

print("Train:")
print(train.shape[0])
print(train.value_counts(["volcano_name", "overall_obs"]))
print("Valid:")
print(valid.shape[0])
print(valid.value_counts(["volcano_name", "overall_obs"]))

wo_Cot_train_img_names = set(wo_Cot_train["image_name"].tolist())
valid_img_names = set(valid["image_name"].tolist())
overlap = wo_Cot_train_img_names.intersection(valid_img_names)
print(overlap)
'''

#Check there's no overlap between train-valid-test splits
train = pd.read_excel("UpdatedTVTSplits/FinalSplit/Train.xlsx")
valid = pd.read_excel("UpdatedTVTSplits/FinalSplit/Valid.xlsx")
test = pd.read_excel("UpdatedTVTSplits/FinalSplit/Test.xlsx")

train_img_names = set(train["image_name"].tolist())
valid_img_names = set(valid["image_name"].tolist())
test_img_names = set(test["image_name"].tolist())

print(train_img_names.intersection(valid_img_names))
print(train_img_names.intersection(test_img_names))
print(valid_img_names.intersection(test_img_names))