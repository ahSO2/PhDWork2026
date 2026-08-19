import pandas as pd
import os

full_days_folder = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/FullDays/"
save_folder = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/FullDays_Labels_and_Predictions/"

columns_to_select = ["image_name", "image_name_B",
                     "minus_one_min_name", "minus_one_min_name_B",
                     "minus_ten_s_name", "minus_ten_s_name_B",
                     "plus_ten_s_name", "plus_ten_s_name_B",
                     "plus_one_min_name", "plus_one_min_name_B",
                     "precipitation_level",
                     "obs_cloud_level",
                     "precipitation_prediction", "obs_cloud_prediction"]

#Read each updated full day application dataframe from the folder
for day in os.listdir(full_days_folder):
    print(day)
    day_df = pd.read_excel(full_days_folder + day)
    # Select only the required columns
    selected_day_columns = day_df[columns_to_select]
    ############TODO if 'other' column exists, include it before saving!
    if 'other' in day_df.columns:
        selected_day_columns["other"] = day_df["other"]

    #Save as a .csv file in the destination folder
    selected_day_columns.to_csv(save_folder + day[:-5] + ".csv")

