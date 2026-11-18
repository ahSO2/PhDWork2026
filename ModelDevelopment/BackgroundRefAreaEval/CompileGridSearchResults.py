import os

import numpy as np
import pandas as pd

folders_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Combination_Params_Grid_Search_Zero_Threshold/"

to_concat = []
for folder in os.listdir(folders_path):
    print(folder)
    combo_results = pd.read_csv(folders_path + folder + "/Mean_UnseenTest_mod1_quality-Good.csv")
    sorted_combo_results = combo_results.sort_values(by=["RN"], ascending=True)
    min_loc = sorted_combo_results["fold"].to_numpy()[0]
    min_RN = sorted_combo_results["RN"].to_numpy()[0]
    loc_bal_row = combo_results[combo_results["fold"]=="location-balanced"]
    loc_bal_row["min_RN"] = min_RN
    loc_bal_row["min_loc"] = min_loc
    to_concat.append(loc_bal_row)

all_results = pd.concat(to_concat, ignore_index=True)
all_results.to_csv(folders_path + "CompiledResults.csv")


