#For the updated dataframes (with my corrections applied to labels)
#For each CV split
#For each subset, calculate:
#Percentage of full CV fold, location split, percentage of obscured samples
import pandas as pd
import numpy as np

folder_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudCVSplits/"
target_name = "obs_cloud"
df_prefix = "Cloud"
save_path = "CloudCVSplits.xlsx"
dataset_names = ["_Train.csv",
            "_TrainExpanded.csv",
            "_TestSeen.csv",
            "_TestUnseen.csv"]

location_names = ["Cotopaxi", "Kilauea", "Merapi", "Reventador"]
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
def calculate_metrics(dataframe, N):
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

    #Calculate prop of that fold:
    split_prop = np.round(n/N *100, 2)

    return split_prop, all_location_counts, prop_obs


dfs_for_total = []
results_df = pd.DataFrame(columns=["left_out_name", "train_prop", "train_location_sample_count", "train_prop_obscured",
                                   "testseen_prop", "testseen_location_sample_count", "testseen_prop_obscured",
                                   "testleftout_prop", "testleftout_location_sample_count", "testleftout_prop_obscured"])
for left_out_loc in location_names:

    train = pd.read_csv(folder_path + df_prefix + "_CV_lo" + left_out_loc + "_Train.csv")
    test_seen = pd.read_csv(folder_path + df_prefix + "_CV_lo" + left_out_loc + "_TestSeen.csv")
    test_left_out = pd.read_csv(folder_path + df_prefix + "_CV_lo" + left_out_loc + "_TestUnseen.csv")

    N = train.shape[0] + test_seen.shape[0] + test_left_out.shape[0]

    train_prop, train_location_counts, train_prop_obs = calculate_metrics(train, N)
    #trainexp_prop, trainexp_location_counts, trainexp_prop_obs = calculate_metrics(train_exp, N)
    testseen_prop, testseen_location_counts, testseen_prop_obs = calculate_metrics(test_seen, N)
    testleftout_prop, testleftout_location_counts, testleftout_prop_obs = calculate_metrics(test_left_out, N)

    new_row = {"left_out_name":left_out_loc,
               "train_prop":train_prop,
               "train_location_sample_count":train_location_counts,
               "train_prop_obscured":train_prop_obs,
               "testseen_prop":testseen_prop,
               "testseen_location_sample_count":testseen_location_counts,
               "testseen_prop_obscured":testseen_prop_obs,
               "testleftout_prop":testleftout_prop,
               "testleftout_location_sample_count":testleftout_location_counts,
               "testleftout_prop_obscured":testleftout_prop_obs}

    results_df.loc[len(results_df)] = new_row

results_df.to_excel(save_path)







