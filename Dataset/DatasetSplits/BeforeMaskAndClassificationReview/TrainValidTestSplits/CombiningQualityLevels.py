import pandas as pd
################### Haven't ended up using this - realised the method of splitting quality levels separately doesn't work
#Combine good quality and low quality

#Full train split
#train_good_qual = pd.read_excel("Train_GoodQual.xlsx")
#train_low_qual = pd.read_excel("Train_LowQual.xlsx")
#full_train = pd.concat([train_good_qual, train_low_qual])
#full_train.to_excel("FinalSplit/Train_WoLastarria.xlsx")

#Full pre-validation split:
#pre_valid_good_qual = pd.read_excel("PreValidation_GoodQual.xlsx")
#pre_valid_low_qual = pd.read_excel("PreValidation_LowQual.xlsx")
#full_pre_valid = pd.concat([pre_valid_good_qual, pre_valid_low_qual])
#full_pre_valid.to_excel("FinalSplit/PreValid_WoLastarria.xlsx")

#Full Validation Split
#valid_good_qual = pd.read_excel("FinalValid_GoodQual.xlsx")
#valid_low_qual = pd.read_excel("FinalValid_LowQual.xlsx")
#full_valid = pd.concat([valid_good_qual, valid_low_qual])
#full_valid.to_excel("FinalSplit/FinalValid_WoLastarria.xlsx")

#Check there's no overlap of stratification groups
train = pd.read_excel("FinalSplit/Train_WoLastarria.xlsx")
valid = pd.read_excel("FinalSplit/PreValid_WoLastarria.xlsx")
test = pd.read_excel("FinalSplit/FinalValid_WoLastarria.xlsx")

train_groups = set(train["stratification_group"].tolist())
valid_groups = set(valid["stratification_group"].tolist())
test_groups = set(test["stratification_group"].tolist())

print(train_groups.intersection(valid_groups))
print(train_groups.intersection(test_groups))
print(test_groups.intersection(valid_groups))
