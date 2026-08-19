import pandas as pd

df = pd.read_csv("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/EvaluatingOnFullDays/FullDays_StdThreshold_OverallObsPredictions.csv")

predicted_low_qual = df[df["obscurance_prediction"] > 0.5]
print(predicted_low_qual.shape[0]/df.shape[0])