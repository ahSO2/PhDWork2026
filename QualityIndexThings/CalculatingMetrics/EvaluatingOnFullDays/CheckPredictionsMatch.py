#Quick check that my new code for application of final models to a full day
#sequence gives the same reuslt as the original version
import pandas as pd

df_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs/ImageNamesSorted.xlsx"
df = pd.read_excel(df_path)
check_column = "obs_cloud"
df["predictions_equal"] = df[check_column + "_model_prediction"] == df[check_column + "_prediction"]
df["predictions_diff"] = df[check_column + "_model_prediction"] - df[check_column + "_prediction"]

print(df["predictions_equal"].value_counts())
print(df["predictions_diff"].abs().max())
df.to_excel(df_path)