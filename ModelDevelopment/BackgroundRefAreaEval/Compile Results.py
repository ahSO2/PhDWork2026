import pandas as pd
import numpy as np
folder_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/"
locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador", "location-balanced"]
methods = ["D-A", "D-R", "G-RL", "G-AT", "K-SR", "S-RF"]

#For each location
for location in locations:
    location_data = []
    for method in methods:
        path_to_read = folder_path + method + "_" + "mean_UnseenTest_mod1_quality-Good.csv"
        method_df = pd.read_csv(path_to_read)
        rel_row = method_df.loc[method_df["fold"]==location]
        rel_row["method"] = [method] * rel_row.shape[0]
        location_data.append(rel_row)
    location_df = pd.concat(location_data)
    location_df = location_df.iloc[:, [-1] + np.arange(0, location_df.shape[1])] #Selecting the columns by index to re-arrange so the method is at the left
    save_path = folder_path + "CompiledResults_" + location + "_UnseenTest_mod1_quality-Good.csv"
    location_df.to_csv(save_path, index=False)


