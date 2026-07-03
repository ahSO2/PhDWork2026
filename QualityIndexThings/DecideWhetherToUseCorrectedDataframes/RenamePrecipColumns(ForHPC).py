import os
import pandas as pd

directory_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/PrecipitationCVSplits/"
destination_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/PrecipitationFullAndCVSplitsWPrecipRenamed_ForHPC/"

def date_from_image_name(name):
    return name.split("_")[1][:10]

for df_name in os.listdir(directory_path):
    df = pd.read_csv(directory_path + df_name)
    df.rename({"precipitation":"rain_or_dirt", "precipitation_level":"rain_or_dirt_level"}, inplace=True, axis="columns")
    #Add "date" column
    df["date"] = df["image_name"].apply(date_from_image_name)
    df.to_csv(destination_path + df_name)