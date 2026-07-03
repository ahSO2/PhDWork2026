import pandas as pd
import numpy as np

results_df = pd.read_excel("MergedLabels/consensus_calc_precipitation.xlsx")
target = "precipitation"
save_path = "CalculatedMetricsSheets/" + target + "_binary_agreement_percentages.xlsx"

locations = ["All", "Cotopaxi", "Kilauea", "Lastarria", "Merapi", "Reventador"]

#Check that every sample has a volcano name value:
print("Samples:" + str(len(results_df)))
print("Volcano name vals:" + str(len(results_df["volcano_name"])))
outputs_df = pd.DataFrame()
for location in locations:
    if location != "All":
        location_data = results_df[results_df["volcano_name"]==location]
    else:
        location_data = results_df.copy()
    location_binary_agreement = np.mean(location_data[target + "_binary_consensus"].tolist())
    outputs_df[location] = [np.round(location_binary_agreement, 4)]
outputs_df.to_excel(save_path)