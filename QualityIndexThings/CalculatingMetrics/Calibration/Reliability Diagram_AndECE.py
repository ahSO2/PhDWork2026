import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibrationDisplay

predictions_df_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/Cloud_Full_TestUnseen.xlsx"
predictions_df = pd.read_excel(predictions_df_path)
target = "obs_cloud"
save_folder = "ReliabilityDiagrams/"
n_bins = 15
def map_YN_to_binary(value):
    if value == "Yes":
        return 1
    else:
        return 0

def calc_confidence(value):
    '''Gives confidence of the prediction (for the predicted class).'''
    if value < 0.5:
        return 1-value
    else:
        return value
def MacroCE(sigmoid_predictions, true_binary, threshold=0.5):
    CE_df = pd.DataFrame()
    CE_df["sigmoid_predictions"] = sigmoid_predictions
    CE_df["true"] = true_binary
    CE_df["binary_predictions"] = np.where(sigmoid_predictions >= threshold, 1, 0)
    CE_df["prediction_correct"] = CE_df["binary_predictions"] == CE_df["true"]
    CE_df["prediction_confidence"] = CE_df["sigmoid_predictions"].apply(calc_confidence)

    correctly_predicted = CE_df[CE_df["prediction_correct"]==True]
    incorrectly_predicted = CE_df[CE_df["prediction_correct"]==False]

    correctly_predicted["CE_indiv"] = 1 - correctly_predicted["prediction_confidence"]
    ICE_pos = correctly_predicted["CE_indiv"].sum() / correctly_predicted.shape[0]
    print("ICE_pos: " + str(np.round(ICE_pos,4)))
    ICE_neg = incorrectly_predicted["prediction_confidence"].sum() /incorrectly_predicted.shape[0]
    print("ICE_neg: " + str(np.round(ICE_neg,4)))
    return np.round((ICE_pos + ICE_neg)/2, 4)

class ECE_calculator():
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def prediction_to_bin_index(self, value):
        bin_index = 0
        cont = True
        for rhs in self.rhs_vals:
            if cont == True:
                if value > rhs:
                    bin_index += 1
                    cont = True
                else:
                    cont=False
        return bin_index
    def calculate(self, sigmoid_predictions, true_binary, threshold=0.5):
        ECE_df = pd.DataFrame()
        ECE_df["sigmoid_prediction"] = sigmoid_predictions
        ECE_df["true_binary"] = true_binary
        ECE_df["binary_prediction"] = np.where(ECE_df["sigmoid_prediction"]>=threshold, 1, 0)
        ECE_df["prediction_correct"] = ECE_df["binary_prediction"] == ECE_df["true_binary"]
        ECE_df["prediction_confidence"] = ECE_df["sigmoid_prediction"].apply(calc_confidence)

        #Sort samples into bins
        self.bin_size = 1/self.n_bins
        self.rhs_vals = np.arange(1, n_bins + 1, 1) * self.bin_size
        ECE_df["bin_index"] = ECE_df["sigmoid_prediction"].apply(self.prediction_to_bin_index)

        #Check bin allocations:
        ECE_df["y_to_plot"] = [1] * ECE_df.shape[0]
        sns.scatterplot(data=ECE_df, x="sigmoid_prediction", y="y_to_plot", hue="bin_index", palette="deep")
        plt.show()

        bin_sizes = []
        bin_accuracies = []
        bin_confidences = []
        #for each bin
        for bin_index in range(0, self.n_bins):
            bin_df = ECE_df[ECE_df["bin_index"]==bin_index]
            if bin_df.shape[0] > 0:
                #Calculate the accuracy in that bin
                acc = np.mean(bin_df["prediction_correct"])
                #Calculate the mean confidence in that bin
                confid = np.mean(bin_df["prediction_confidence"])
                bin_sizes.append(bin_df.shape[0])
                bin_accuracies.append(acc)
                bin_confidences.append(confid)
            else:
                bin_sizes.append(0)
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
        #Weighted sum
        total = 0
        for bin_index in range(0, self.n_bins):
            if bin_sizes[bin_index] > 0:
                total =  total + (bin_sizes[bin_index] * np.abs(bin_accuracies[bin_index] - bin_confidences[bin_index]))
        return np.round(total/ECE_df.shape[0], 4)


y_test = predictions_df[target].apply(map_YN_to_binary)
y_prob = predictions_df[target + "_prediction"]

Calculator = ECE_calculator(n_bins=n_bins)
ECE = Calculator.calculate(y_prob, y_test)
print("ECE: " + str(ECE))

macroCE = MacroCE(y_prob, y_test)
print("MacroCE: " + str(macroCE))


prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
#disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
#disp.plot()
#plt.show()
#plt.close()

cm = 1 / 2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(18*cm, 18*cm))
range = np.arange(0, 1.1, 0.1)
ax.plot(range, range, linestyle="dotted", color="black", label="Perfectly calibrated")
ax.plot(prob_pred, prob_true, label="Model", marker="s")
ax.set_xlabel("Mean Sigmoid Score", fontsize=16)
ax.set_ylabel("Proportion of Positives", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=16)
#plt.savefig(save_folder + "Target" + target + "_" + predictions_df_path.split("/")[-1][:-5] + ".jpg", dpi=300)
plt.show()