#For each dataframe in my folder holding results of applying models to full days,
#update the names of certain columns to match my most up-to-date versions.
import os

import pandas as pd

directory_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/ModelApplicToFullDays_UpdatedJun26/"

to_rename = {'rain_or_dirt_model_prediction':'precipitation_model_prediction',
             'cloud_model_prediction':'obs_cloud_model_prediction',
             'OnLens':'precipitation_level',
             'on_lens_level':'precipitation_level',
             'FGCloud':'obs_cloud_level',
             'cloud_level':'obs_cloud_level',
             'rain_or_dirt_Yes':'precipitation_Yes',
             'cloud_Yes':'obs_cloud_Yes',
             'rain_or_dirt':'precipitation',
             'cloud':'obs_cloud'}

for folder in os.listdir(directory_path):
    for file_name in os.listdir(directory_path + folder):
        sheet = pd.read_excel(directory_path + folder + "/" + file_name)
        for column_to_rename in to_rename:
            if column_to_rename in sheet.columns:
                sheet.rename(columns={column_to_rename: to_rename[column_to_rename]}, inplace=True)
        sheet.to_csv(directory_path + folder + "/" + file_name[:-5] + ".csv")
        os.remove(directory_path + folder + "/" + file_name)
