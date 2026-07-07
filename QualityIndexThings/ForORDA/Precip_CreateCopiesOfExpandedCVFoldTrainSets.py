#Goal is to read in from the HPC outputs copies of the
#expanded training sets for the cross-validation folds
#(where extra data from Kilauea was included as a hyperparameter),
#then tidy up the dataframes as done for all the other dataframes
#for ORDA.
import numpy as np
import pandas as pd

#Path to the raw dataframe
raw_dataframe = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment235/TrainWithAddData.xlsx")
#Path where we want to save the cleaned version
save_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationCVSplits/Precipitation_CV_loMerapi_TrainExpanded.csv"


def copy_on_lens_level_value(row):
    if pd.isna(row["rain_or_dirt_level"]):
        return row["on_lens_level"]
    else:
        return row["rain_or_dirt_level"]

def edit_data_type_col(value):
    if value == "original":
        return "Original"
    elif value == "additional":
        return "Additional"

#Copy the "on_lens_level" value for the extra Kilauea samples into the
#"rain_or_dirt" column
raw_dataframe["rain_or_dirt_level"] = raw_dataframe.apply(copy_on_lens_level_value, axis=1)
print(raw_dataframe["rain_or_dirt_level"])

raw_dataframe["labelled"] = raw_dataframe["data_type"].apply(edit_data_type_col)

to_drop = ["prev_thirtysec_name", "prev_thirtysec_name_B",
           "next_thirtysec_name", "next_thirtysec_name_B",
           "model_predictions", "is_prediction_correct",
           "if_not_what_should_on_lens_level_be",
           "cloud", "on_lens_level", 'Unnamed: 0.1',
           'Unnamed: 0', 'data_type']
raw_dataframe = raw_dataframe.drop(to_drop, axis=1)
print(raw_dataframe.columns)

raw_dataframe.rename({"rain_or_dirt":"precipitation",
                      "rain_or_dirt_level":"precipitation_level"}, inplace=True, axis="columns")

print(raw_dataframe.columns)

def map_level_to_YN(level):
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
        print("Error in level value!")

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
        print("Error in level given.")

def map_numeric_to_level(num):
    if num == 0:
        return "No"
    elif num == 1:
        return "Minor"
    elif num == 2:
        return "Not Calc"
    elif num == 3:
        return "In Calc"
    elif num == 4:
        return "Very"
    else:
        print("Error in numeric level")

def calc_obs_level(row):
    cloud_level = row["obs_cloud_level"]
    precip_level = row["precipitation_level"]
    cloud_num = map_level_to_numeric(cloud_level)
    precip_num = map_level_to_numeric(precip_level)
    obs_num = max(cloud_num, precip_num)
    obs_level = map_numeric_to_level(obs_num)
    return obs_level
def correct_mislablled_samples(original):
    to_correct = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/DecideWhetherToUseCorrectedDataframes/CorrectedMistakes_NoDuplicates.xlsx")
    for sample_to_correct in to_correct["image_name"].tolist():
        corrections_df_row = to_correct[to_correct["image_name"] == sample_to_correct]
        #Get the updated value
        correct_precip_level = corrections_df_row["correct_precip_level"].tolist()[0]
        #Replace the precip value for any rows with that image name
        original.loc[original['image_name'] == sample_to_correct, 'precipitation_level'] = correct_precip_level

        if sample_to_correct in original["image_name"].tolist():
            print("Correcting prediction for:" + sample_to_correct)
            row_to_correct = original.loc[original["image_name"]==sample_to_correct]
            row_to_correct.reset_index(inplace=True, drop=True)
            print(row_to_correct)
            correct_precip_yn = map_level_to_YN(correct_precip_level)
            original.loc[original['image_name'] == sample_to_correct, 'precipitation'] = correct_precip_yn

            if (row_to_correct.iloc[0].isna()["obs_cloud_level"] == False):
                print(row_to_correct.iloc[0]["obs_cloud_level"])
                correct_obs_level = calc_obs_level(row_to_correct.iloc[0])
                correct_obs_yn = map_level_to_YN(correct_obs_level)
                original.loc[original['image_name'] == sample_to_correct, 'obscurance_level'] = correct_obs_level
                original.loc[original["image_name"] == sample_to_correct, "obscurance"] = correct_obs_yn
    return original

raw_dataframe = correct_mislablled_samples(raw_dataframe)
raw_dataframe.to_csv(save_path)