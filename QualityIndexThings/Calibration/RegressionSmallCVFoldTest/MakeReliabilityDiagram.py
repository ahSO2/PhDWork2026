import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay

'''We want to evaluate the calibration of the original cross-validation fold model,
then evaluate after re-training as a regression to see if this is liekly to be a promising alteration.'''

predictions_df = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment133/Cotopaxi_TestPredictions.xlsx")

def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    else:
        return 0

def expected_calibration_error():


y_test = predictions_df["cloud_Yes"]
y_prob = predictions_df["model_predictions"]


prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
print(prob_true)
disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
disp.plot()
plt.show()

