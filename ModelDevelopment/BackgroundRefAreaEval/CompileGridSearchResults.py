import os

import numpy as np
import pandas as pd

folders_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Combination_Params_Grid_search/"

to_concat = []
for folder in os.listdir(folders_path):
    print(folder)
    combo_results = pd.read_csv(folders_path + folder + "/Mean_UnseenTest_mod1_quality-Good.csv")
    sorted_combo_results = combo_results.sort_values(by=["R_AP99"], ascending=True)
    min_loc = sorted_combo_results["fold"].to_numpy()[0]
    min_R_AP99 = sorted_combo_results["R_AP99"].to_numpy()[0]
    loc_bal_row = combo_results[combo_results["fold"]=="location-balanced"]
    loc_bal_row["min_R_AP99"] = min_R_AP99
    loc_bal_row["min_loc"] = min_loc
    to_concat.append(loc_bal_row)

all_results = pd.concat(to_concat, ignore_index=True)
all_results.to_csv(folders_path + "CompiledResults.csv")


