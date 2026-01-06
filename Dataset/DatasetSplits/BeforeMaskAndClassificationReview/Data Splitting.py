from datetime import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold


def map_image_name_to_time(image_name):
    #TODO edit date when nec
    time_str = image_name.split("_")[1][11:17]
    date_str = image_name.split("_")[1][0:10]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    #TODO change this to a time object, and then plot
    print(image_name)
    print(datetime_obj)
    return datetime_obj

#all_labelled_data = pd.read_excel("AllLabelledImagesList.xlsx", index_col=0)
#print(all_labelled_data.columns)

#Step 1 - Reserve Lastarria
'''
lastarria_test_set = all_labelled_data[all_labelled_data["volcano_name"] == "Lastarria"]
print(lastarria_test_set.shape)
lastarria_test_set.to_excel("TrainValidTestSplits/LastarriaTest.xlsx")

remaining_locations = all_labelled_data[~(all_labelled_data["volcano_name"] == "Lastarria")]
print(remaining_locations.shape)
remaining_locations.to_excel("TrainValidTestSplits/AllLabelsWithoutLastarria.xlsx")
'''

#Step 2 - Visualise the distribution of the set.
'''
#Want to ensure very similar samples (from the same time period) don't end up in the same split.
remaining_locations = pd.read_excel("TrainValidTestSplits/AllLabelsWithoutLastarria.xlsx", index_col=0)

#Make timeline plot of the dates of samples, from each location?

datetimes = remaining_locations["image_name"].copy().apply(map_image_name_to_time)
remaining_locations["datetimes"] = datetimes

fig, ax = plt.subplots(figsize=(9,3))
ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
#Colormap tutorial from: https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/
colormap = {'Good':'seagreen', 'Low':'tomato'}
ax.scatter(datetimes, remaining_locations["volcano_name"], alpha=0.1, c=remaining_locations["overall_quality"].map(colormap))
#sns.stripplot(data=remaining_locations, x='datetimes', y="volcano_name", hue='obscured', alpha=0.2, ax=ax)
#plt.legend()
plt.ylabel("Volcano")
plt.title("Plume Segmentation Labelled Samples")
for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
plt.tight_layout()
plt.show()
'''

#Step3
'''
#Random sample equal number from each location
remaining_locations = pd.read_excel("TrainValidTestSplits/AllLabelsWithoutLastarria.xlsx", index_col=0)
locations = ["Reventador", "Cotopaxi", "Merapi", "Kilauea"]
to_concat = []
random_states = [42,84,10,70]
index = 0
for location in locations:
    #Sample 100 good quality and 50 low quality
    location_data = remaining_locations[remaining_locations["volcano_name"] == location]
    good_qual_pool = location_data[location_data["overall_quality"]=="Good"]
    low_qual_pool = location_data[location_data["overall_quality"]=="Low"]
    selected_good_qual = good_qual_pool.sample(n=100, replace=False, random_state=random_states[index])
    selected_low_qual = low_qual_pool.sample(n=50, replace=False, random_state=int(random_states[index]/2))
    to_concat.append(selected_good_qual)
    to_concat.append(selected_low_qual)
    index += 1

lascar_data = remaining_locations[remaining_locations["volcano_name"]=="Lascar"]
to_concat.append(lascar_data)
balanced_rem_locs = pd.concat(to_concat)
print(balanced_rem_locs.value_counts("volcano_name"))
balanced_rem_locs.to_excel("TrainValidTestSplits/Balanced_AllLabelsWithoutLastarria.xlsx")

#Add Lascar data - done
#Check the number of unique days from Mer low qual is not signif reduced
def map_image_name_to_date_string(image_name):
    return image_name.split("_")[1][0:10]

all_Mer_samples = remaining_locations[remaining_locations["volcano_name"]=="Merapi"]
#all_Mer_samples = all_Mer_samples[all_Mer_samples["overall_quality"]=="Low"]
selected_Mer_samples = balanced_rem_locs[balanced_rem_locs["volcano_name"]=="Merapi"]
#selected_Mer_samples = selected_Mer_samples[selected_Mer_samples["overall_quality"]=="Low"]
all_Mer_samples_dates = all_Mer_samples["image_name"].apply(map_image_name_to_date_string).tolist()
selected_Mer_samples_dates = selected_Mer_samples["image_name"].apply(map_image_name_to_date_string).tolist()

n_original_unique_dates = len(list(set(all_Mer_samples_dates)))
print(n_original_unique_dates)
n_selected_unique_dates = len(list(set(selected_Mer_samples_dates)))
print(n_selected_unique_dates)
'''

#Step4 - Create grouping to stratify the TTV Split By
'''
#Iterative grouping method:
#Start with a sample, if there are more observations from that
#location within 24hrs add them to the chunk. If there are more
#observations within 24hrs of the last obs, add them to the chunk,
#otherwise start a new chunk. Repeat.

samples_to_group = pd.read_excel("TrainValidTestSplits/Balanced_AllLabelsWithoutLastarria.xlsx")

locations = list(set(samples_to_group["volcano_name"].copy().tolist()))
samples_to_group["datetime"] = samples_to_group["image_name"].copy().apply(map_image_name_to_time)
group_index = 0
group_index_dict = {}
for location in locations:
    group_index += 1
    location_data = samples_to_group[samples_to_group["volcano_name"]==location]
    #Sort by datetime
    location_data = location_data.sort_values(by="datetime")
    #location_data.to_excel("testoutput.xlsx")
    location_data.reset_index(inplace=True, drop=True)
    #Select current observation
    current_observation_index = 0
    #For every observation:
    while current_observation_index < (location_data.shape[0] - 1):
        #Assign a group index
        image_name = location_data["image_name"][current_observation_index]
        group_index_dict[image_name] = group_index
        #Calculate the time diff to next sample
        current_datetime = location_data["datetime"][current_observation_index]
        next_datetime = location_data["datetime"][current_observation_index + 1]
        time_diff = next_datetime-current_datetime
        day_seconds = 36 * 60 * 60
        #If less than 24hrs keep the same group index
        if time_diff.total_seconds() <= day_seconds:
            pass
        #Else add one to the group index
        else:
            group_index += 1
        #Add one to the obs index
        current_observation_index += 1
    #Assign the group to the last index:
    image_name = location_data["image_name"][current_observation_index]
    group_index_dict[image_name] = group_index

#Then match up all the samples to their assigned groups and
#save as a column in the dataframe
assigned_groups = []
for image_name in samples_to_group["image_name"].tolist():
    assigned_group = group_index_dict[image_name]
    assigned_groups.append(assigned_group)

samples_to_group["stratification_group"] = assigned_groups
samples_to_group.to_excel("TrainValidTestSplits/BalancedAndGrouped_AllLabelsWithoutLastarria.xlsx")
'''

#Step5 - Do a stratified split for final valid and pre-valid sets
#Starting with the good quality data:
#balanced_samples = pd.read_excel("TrainValidTestSplits/BalancedAndGrouped_AllLabelsWithoutLastarria.xlsx")
#balanced_good_qual = balanced_samples[balanced_samples["overall_quality"]=="Good"]
#balanced_good_qual.to_excel("TrainValidTestSplits/BalancedAndGrouped_AllGoodQualWithoutLastarria.xlsx")
#Want to stratify by group
#Aiming for roughly equal location dist
#Split for good and low quality separately?

data_to_split = pd.read_excel("TrainValidTestSplits/BalancedAndGrouped_AllGoodQualWithoutLastarria.xlsx")

groups_for_test = [27, 23, 16, 14, 8, 29, 32, 34, 70, 64, 62, 77, 80, 90, 94, 95, 103, 107, 105]

train_and_valid = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_test))]
train_and_valid.to_excel("TrainValidTestSplits/AllTrainAndValid_GoodQual.xlsx")

test = data_to_split[data_to_split["stratification_group"].isin(groups_for_test)]
test.to_excel("TrainValidTestSplits/FinalValid_GoodQual.xlsx")

print("TrainAndValid:")
print(train_and_valid.shape[0])
print(train_and_valid.value_counts(["volcano_name"]))
print("Test:")
print(test.shape[0])
print(test.value_counts(["volcano_name"]))



#Now for the good quality pre-valid split:
data_to_split = pd.read_excel("TrainValidTestSplits/AllTrainAndValid_GoodQual.xlsx")

groups_for_valid = [24, 19, 7, 4, 33, 35, 45, 71, 65, 61, 49, 76, 81, 104, 92]

pre_valid = data_to_split[data_to_split["stratification_group"].isin(groups_for_valid)]
pre_valid.to_excel("TrainValidTestSplits/PreValidation_GoodQual.xlsx")
train = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_valid))]
train.to_excel("TrainValidTestSplits/Train_GoodQual.xlsx")
print("Train:")
print(train.shape[0])
print(train.value_counts(["volcano_name"]))
print("Pre-Valid")
print(pre_valid.shape[0])
print(pre_valid.value_counts(["volcano_name"]))


'''
#Next the low quality samples:
#balanced_samples = pd.read_excel("TrainValidTestSplits/BalancedAndGrouped_AllLabelsWithoutLastarria.xlsx")
#balanced_low_qual = balanced_samples[balanced_samples["overall_quality"]=="Low"]
#balanced_low_qual.to_excel("TrainValidTestSplits/BalancedAndGrouped_AllLowQualWithoutLastarria.xlsx")

data_to_split = pd.read_excel("TrainValidTestSplits/BalancedAndGrouped_AllLowQualWithoutLastarria.xlsx")

groups_for_test = [1, 4, 14, 8, 29, 37, 46, 41, 51, 67, 57, 58, 73, 86, 90, 96, 97, 107, 88]

train_and_valid = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_test))]
train_and_valid.to_excel("TrainValidTestSplits/AllTrainAndValid_LowQual.xlsx")

test = data_to_split[data_to_split["stratification_group"].isin(groups_for_test)]
test.to_excel("TrainValidTestSplits/FinalValid_LowQual.xlsx")

print("TrainAndValid:")
print(train_and_valid.shape[0])
print(train_and_valid.value_counts(["volcano_name"]))
print("Test:")
print(test.shape[0])
print(test.value_counts(["volcano_name"]))

#Now for the low quality pre-valid split:
data_to_split = pd.read_excel("TrainValidTestSplits/AllTrainAndValid_LowQual.xlsx")

groups_for_valid = [2, 10, 17, 35, 48, 45, 40, 55, 68, 49, 72, 74, 83, 85, 101, 89]

pre_valid = data_to_split[data_to_split["stratification_group"].isin(groups_for_valid)]
pre_valid.to_excel("TrainValidTestSplits/PreValidation_LowQual.xlsx")
train = data_to_split[~(data_to_split["stratification_group"].isin(groups_for_valid))]
train.to_excel("TrainValidTestSplits/Train_LowQual.xlsx")

print("Train:")
print(train.shape[0])
print(train.value_counts(["volcano_name"]))
print("Pre-Valid")
print(pre_valid.shape[0])
print(pre_valid.value_counts(["volcano_name"]))
'''