import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

##TODO #############################################
#NOTE: The code below inverts the calculated sigmoid scores (1-value)
#and binary target values to give a prediction and true value for
#UNOBSCURED DATA. Threshold values output by the sklearn function
#will be relative to this inverted setup.

predictions_df_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs/Precipitation_Full_TestSeen.xlsx"
predictions_df = pd.read_excel(predictions_df_path)
target = "precipitation"
save_folder = "PrecisionRecallCurves/"

def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    else:
        return 0

predicted_sigmoid = predictions_df[target + "_prediction"]
target = predictions_df[target].apply(map_YN_to_binary)

predicted_sigmoid_inverse = 1 - predicted_sigmoid
target_inverse = 1-target

precisions, recalls, thresholds = precision_recall_curve(target_inverse, predicted_sigmoid_inverse)

cm = 1 / 2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(18*cm, 18*cm))
ax.plot(recalls, precisions)
ax.set_xlabel("Recall", fontsize=14)
ax.set_ylabel("Precision", fontsize=14)
plt.savefig(save_folder + "/" + predictions_df_path.split("/")[-1][:-5] + ".jpg", dpi=300)
plt.show()

corresponding_thresholds = 1 - thresholds
cm = 1 / 2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(18*cm, 18*cm))
ax.plot(corresponding_thresholds, precisions[:-1], label="precision")
ax.plot(corresponding_thresholds, recalls[:-1], label="recall")
ax.set_xlabel("Threshold", fontsize=14)
ax.set_ylabel("Precision/Recall", fontsize=14)
plt.legend()
plt.savefig(save_folder + "/PRvsThreshold_" + predictions_df_path.split("/")[-1][:-5] + ".jpg", dpi=300)
plt.show()

metrics_df = pd.DataFrame()
metrics_df["Precision"] = precisions[:-1]
metrics_df["Recall"] = recalls[:-1]
metrics_df["Threshold"] = thresholds
metrics_df.to_csv(save_folder + "/" + "PerThresholdPR_" + predictions_df_path.split("/")[-1][:-5] + ".csv")



