#For the updated dataframes (with my corrections applied to labels)
#For the full split
#For each subset, calculate:
#Number of samples, location split, percentage of obscured samples
import pandas as pd
import numpy as np

folder_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudFullSplit/"
target_name = "obs_cloud"
save_path = "CloudFullSplit.xlsx"
dataset_names = ["Cloud_Full_Train.csv",
            "Cloud_Full_TrainExpanded.csv",
            "Cloud_Full_Valid.csv",
            "Cloud_Full_TestSeen.csv",
            "Cloud_Full_TestUnseen.csv"]

for_total = ["Cloud_Full_TrainExpanded.csv",
            "Cloud_Full_Valid.csv",
            "Cloud_Full_TestSeen.csv",
            "Cloud_Full_TestUnseen.csv"]
location_names = ["Cotopaxi", "Kilauea", "Lastarria", "Merapi", "Reventador"]
def map_image_name_to_volcano_name(image_name):
    if "Cotopaxi" in image_name:
        return "Cotopaxi"
    elif "Merapi" in image_name:
        return "Merapi"
    elif "Kilauea" in image_name:
        return "Kilauea"
    elif "Reventador" in image_name:
        return "Reventador"
    elif "Lastarria" in image_name:
        return "Lastarria"
    else:
        print("Error in volcano name value!")
def calculate_metrics(dataframe):
    #Calculate number of samples
    n = dataframe.shape[0]

    #Calculate location split
    dataframe["volcano_name"] = dataframe["image_name"].apply(map_image_name_to_volcano_name)
    location_count_strings = []
    location_counts_numeric = []
    for location in location_names:
        location_samples = dataframe[dataframe["volcano_name"]==location]
        location_counts_numeric.append(location_samples.shape[0])
        location_count_string = str(location_samples.shape[0])
        location_count_strings.append(location_count_string)
    all_location_counts = ":".join(location_count_strings)
    #Sense check that location counts sum to the total number of samples
    if sum(location_counts_numeric) != n:
        print("Error in location sample counts!")

    #Calculate percentage obscured samples
    obscured_samples = dataframe[dataframe[target_name]=="Yes"]
    prop_obs = np.round(obscured_samples.shape[0]/n * 100, 2)

    return n, all_location_counts, prop_obs


dfs_for_total = []
results_df = pd.DataFrame(columns=["set_name", "sample_count", "location_sample_count", "prop_obscured"])
for dataset_name in dataset_names:
    dataset = pd.read_csv(folder_path + dataset_name)

    if dataset_name in for_total:
        dfs_for_total.append(dataset)

    sample_count, location_counts, prop_obs = calculate_metrics(dataset)
    new_row = {"set_name":dataset_name,
               "sample_count":sample_count,
               "location_sample_count":location_counts,
               "prop_obscured":prop_obs}
    results_df.loc[len(results_df)] = new_row
full_dataset = pd.concat(dfs_for_total)
sample_count, location_counts, prop_obs = calculate_metrics(full_dataset)
new_row = {"set_name":"FullSet",
           "sample_count":sample_count,
           "location_sample_count":location_counts,
           "prop_obscured":prop_obs}
results_df.loc[len(results_df)] = new_row
results_df.to_excel(save_path)







