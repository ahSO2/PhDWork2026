import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay

'''We want to evaluate the calibration of the original cross-validation fold model,
then evaluate after re-training as a regression to see if this is liekly to be a promising alteration.'''

#predictions_df = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment133/Cotopaxi_TestPredictions.xlsx")
predictions_df = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment227/TestSetPredictions.xlsx")
def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    else:
        return 0

def threshold_values(value):
    if value >= 0.5:
        return 1
    else:
        return 0

predictions_df["target_binary"] = predictions_df["target"].apply(threshold_values)
y_test = predictions_df["target_binary"]
y_prob = predictions_df["prediction"]


prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
print(prob_true)
disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
disp.plot()
plt.show()

