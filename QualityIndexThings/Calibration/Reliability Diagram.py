import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay

predictions_df = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/FinalModelOutputs/Precipitation_Full_TestSeen_ImageNamesWithModelPredictions.xlsx")
#predictions_df = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/FinalModelOutputs/Cloud_Full_TestSeen_ImageNamesWithModelPredictions.xlsx")

def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    else:
        return 0

#For precip
y_test = predictions_df["precipitation"].apply(map_YN_to_binary)
y_prob = predictions_df["precipiation_prediction"]

#For cloud
#y_test = predictions_df["obs_cloud"].apply(map_YN_to_binary)
#y_prob = predictions_df["cloud_prediction"]

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
print(prob_true)
disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
disp.plot()
plt.show()