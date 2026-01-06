#Count the samples from each location and quality
#Create the groups for stratified splitting

import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import time
from datetime import datetime, timedelta
import pandas as pd

dataset_path = "C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList_UpdatedClassifications.xlsx"
updated_set = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDProjectStep2/Dataset/AllLabelledImagesList_UpdatedClassifications.xlsx")

time_conversion_dictionary = {"Cotopaxi":-5,
                              "Reventador":-5,
                              "Lascar":-3,
                              "Lastarria":-3,
                              "Kilauea":-10,
                              "Merapi":+7}

def map_image_name_to_time(image_name):
    time_str = image_name.split("_")[1][11:17]
    date_str = image_name.split("_")[1][0:10]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    #print(image_name)
    #print(datetime_obj)

    image_location = image_name.split("_")[0]
    hours_to_add = time_conversion_dictionary[image_location]
    updated_datetime = datetime_obj + timedelta(hours = hours_to_add)
    #print(updated_datetime)
    return updated_datetime
def map_level_to_numeric(level):
    if level == "No":
        return 0
    elif level == "Minor":
        return 1
    elif level == "Not Calc":
        return 2
    elif level == "In Calc":
        return 3
    elif level == "Very":
        return 4
    else:
        print("Error in input level name.")
        print(level)

def take_max_obs_level(row):
    return max(row["precip_level_numeric"], row["fg_cloud_level_numeric"])


def map_numeric_to_level(number):
    if number == 0:
        return "No"
    elif number == 1:
        return "Minor"
    elif number == 2:
        return "Not Calc"
    elif number == 3:
        return "In Calc"
    elif number == 4:
        return "Very"
    else:
        print("Error in input level.")
        print(number)

def map_level_to_y_n(level):
    if level == "No":
        return "No"
    elif level == "Minor":
        return "No"
    elif level == "Not Calc":
        return "Yes"
    elif level == "In Calc":
        return "Yes"
    elif level == "Very":
        return "Yes"
    else:
        print("Error in input level name.")
        print(level)

#Need to re-calculate the overall quality level
#updated_set["precip_level_numeric"] = updated_set["precip_level"].apply(map_level_to_numeric)
#updated_set["fg_cloud_level_numeric"] = updated_set["fg_cloud_level"].apply(map_level_to_numeric)
#updated_set["obs_level_numeric"] = updated_set.apply(take_max_obs_level, axis=1)
#updated_set["obs_level"] = updated_set["obs_level_numeric"].apply(map_numeric_to_level)
#updated_set["overall_obs"] = updated_set["obs_level"].apply(map_level_to_y_n)
#updated_set.to_excel(dataset_path, index="SampleIndex")

#locations = ["Cotopaxi", "Kilauea", "Lascar", "Lastarria", "Merapi", "Reventador"]

'''
for location in locations:
    print(location)
    location_data = updated_set[updated_set["volcano_name"]==location]
    good_quality = location_data[location_data["overall_obs"] == "No"]
    print("Good quality:" + str(good_quality.shape[0]))
'''

#Extract time from image names, and convert to local time
datetimes = updated_set["image_name"].copy().apply(map_image_name_to_time)
updated_set["datetimes"] = datetimes

#Make timeline plot of the dates of samples, from each location
'''
fig, ax = plt.subplots(figsize=(9,3))
ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
#Colormap tutorial from: https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/
colormap = {'No':'seagreen', 'Yes':'tomato'}
ax.scatter(datetimes, updated_set["volcano_name"], alpha=0.1, c=updated_set["overall_obs"].map(colormap))
#sns.stripplot(data=remaining_locations, x='datetimes', y="volcano_name", hue='obscured', alpha=0.2, ax=ax)
plt.ylabel("Volcano")
plt.title("Plume Segmentation Labelled Samples")
for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
plt.tight_layout()
plt.show()
'''

#Create grouping to stratify the TTV Split By
#Iterative grouping method:
#Start with a sample, if there are more observations from that
#location within 24hrs add them to the chunk. If there are more
#observations within 24hrs of the last obs, add them to the chunk,
#otherwise start a new chunk. Repeat.

samples_to_group = updated_set
#locations = list(set(samples_to_group["volcano_name"].copy().tolist()))
locations = ["Lascar", "Reventador", "Kilauea", "Cotopaxi", "Lastarria", "Merapi"]
print("Locations:")
print(locations)
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
samples_to_group.to_excel(dataset_path)

