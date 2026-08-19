#Just a quick check that the 26 day sequences are unseen in training of each model
import os
import pandas as pd
volcano_names = ["Merapi", "Reventador", "Kilauea", "Lastarria", "Cotopaxi"]

def day_name_from_image_name(image_name):
    date = image_name.split("_")[1][:10]
    for volcano in volcano_names:
        if volcano in image_name:
            volcano_name = volcano
    #print(volcano_name + "_" + date)
    return volcano_name + "_" + date

#full_days_folder_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/ModelApplicToFullDays_NewDates/"
#base_names = os.listdir(full_days_folder_path)
#print(full_day_names)

#base_set_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationFullSplit/Precipitation_Full_TestSeen.csv"
#base_set = pd.read_csv(base_set_path)
#base_names = base_set["image_name"]

base_names = ["Kilauea_2023-10-13"]

set_to_check_paths = ["C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationFullSplit/Precipitation_Full_TrainExpanded.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationFullSplit/Precipitation_Full_Valid.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudFullSplit/Cloud_Full_TrainExpanded.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudFullSplit/Cloud_Full_Valid.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudCVSplits/Cloud_CV_KilaueaExtraCorrectedSamples.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudCVSplits/Cloud_CV_KilaueaExtraCorrectSamples.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationCVSplits/Precipitation_CV_KilaueaExtraCorrectedSamples.csv",
                      "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationCVSplits/Precipitation_CV_KilaueaExtraCorrectSamples.csv"]
data_to_concat = []
for set_to_check in set_to_check_paths:
    set_df = pd.read_csv(set_to_check)
    data_to_concat.append(set_df)
data_to_check = pd.concat(data_to_concat)
names_to_check = data_to_check["image_name"].apply(day_name_from_image_name)

overlap = set(names_to_check).intersection(base_names)
print(overlap)
#For each day with overlap, count the occurrences in the training fold
for day in overlap:
    n = names_to_check.to_list().count(day)
    print("There are " + str(n) + " samples from full day " + day + " in the checked set.")

