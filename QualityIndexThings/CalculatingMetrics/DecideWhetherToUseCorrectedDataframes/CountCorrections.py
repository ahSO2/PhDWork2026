#Read in a list of dataframes
#Read in the sheet of names of corrected samples
#Count the number of samples in each dataframe that have been corrected
import os
import numpy as np
import pandas as pd

directory_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/BeforeLabelCorrections_Feb19th2026/PrecipitationFullSplit/"
dataframes_to_check = os.listdir(directory_path)
corrected_df = pd.read_excel("CorrectedMistakes_NoDuplicates.xlsx")
#corrected_df["core_name"] = corrected_df["image_name"].apply(core_name_only)
corrected_names = set(corrected_df["image_name"])
print(corrected_names)

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

for df_name in dataframes_to_check:
    if "~" in df_name:
        pass
    else:
        print("For " + df_name + ":")
        df = pd.read_excel(directory_path + df_name)
        image_names = df["image_name"].tolist()
        corrected_count = corrected_names.intersection(set(image_names))
        print(str(len(corrected_count)))
        print("which is " + str((len(corrected_count)/df.shape[0])*100) + "%")

        binary_change = []
        for corrected_image_name in corrected_count:
            #Check whether the binary prediction is changed
            image_index = image_names.index(corrected_image_name)
            original_prediction = df["precipitation"].tolist()[image_index]

            corrected_index = corrected_df["image_name"].tolist().index(corrected_image_name)
            corrected_prediction_level = corrected_df["correct_precip_level"].tolist()[corrected_index]
            corrected_prediction_binary = map_level_to_YN(corrected_prediction_level)

            if original_prediction == corrected_prediction_binary:
                binary_change.append(0)
            else:
                binary_change.append(1)
        print("Prop of corrected with change in binary class: " + str(np.round((np.mean(binary_change) *100), 4)) )


