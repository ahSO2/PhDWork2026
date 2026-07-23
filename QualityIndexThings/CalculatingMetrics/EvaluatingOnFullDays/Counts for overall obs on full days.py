import pandas as pd
import numpy as np

all_data = pd.read_csv("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/EvaluatingOnFullDays/FullDays_StdThreshold_OverallObsPredictions.csv")

locations = ["Cotopaxi", "Kilauea", "Lastarria", "Merapi", "Reventador", "All"]

results_df = pd.DataFrame(columns=["location", "sample_count", "prop_unobs"])
for location in locations:
    if location == "All":
        location_data = all_data.copy()
    else:
        location_data = all_data[all_data["image_name"].str.contains(location)]
    location_n = location_data.shape[0]
    location_good_qual = location_data[location_data["obscurance"]=="No"]
    prop_good_qual = np.round(location_good_qual.shape[0]/location_n, 4)
    results_df.loc[len(results_df)] = {"location":location,
                                       "sample_count":location_n,
                                       "prop_unobs":prop_good_qual}

results_df.to_excel("FullDays_PropObscuredCounts.xlsx")
