#I have realised that I've used the wrong timestep size when applying the final
#precipitation model to each of the full-day sequences. I will write code here to
#remove the existing predictions, and will then re-run the model application with
#the correct timesteps.

import os
import pandas as pd

directory_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/ModelApplicToFullDays_UpdatedJun26/"
to_drop = set(["precipitation_model_prediction", "obs_cloud_model_prediction"])

for folder in os.listdir(directory_path):
    for file_name in os.listdir(directory_path + folder):
        print(file_name)
        sheet = pd.read_csv(directory_path + folder + "/" + file_name)
        drop_from_df = set(sheet.columns).intersection(to_drop)
        if len(drop_from_df) != 0:
            sheet= sheet.drop(drop_from_df, axis=1)
            sheet.to_csv(directory_path + folder + "/" + file_name)