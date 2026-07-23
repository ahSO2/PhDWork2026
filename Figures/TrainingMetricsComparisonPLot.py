#Plot the difference in valid acc between two experiments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lhs = "155"
rhs = "165"
locations = ["OnLensSetFullSplit"]
output_save_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Version for submission - R2/Figures/"

lhs_df_path = "C:/Users/ggp24ash/Documents/HPC Outputs/Experiment" + lhs + "/Outputs/"
#lhs_df_path = "C:/Users/ggp24ash/Documents/HPC Outputs/Experiment78/"
rhs_df_path = "C:/Users/ggp24ash/Documents/HPC Outputs/Experiment" + rhs + "/Outputs/"

for location in locations:
    lhs_data = pd.read_excel(lhs_df_path + location + "_Metrics.xlsx")
    #lhs_data = pd.read_excel(lhs_df_path + "Outputs/Merapi_Metrics_nCorrected600_addCorrect-y.xlsx")
    rhs_data = pd.read_excel(rhs_df_path  + location + "_Metrics.xlsx")

    lhs_valid_accs = lhs_data["ValidAccuracies"]
    rhs_valid_accs = rhs_data["ValidAccuracies"]
    lhs_train_accs = lhs_data["TrainAccuracies"]
    rhs_train_accs = rhs_data["TrainAccuracies"]
    #lhs_test_accs = lhs_data["TestAccuracies"]
    #rhs_test_accs = rhs_data["TestAccuracies"]
    lhs_valid_losses = lhs_data["ValidLosses"]
    rhs_valid_losses = rhs_data["ValidLosses"]
    lhs_train_losses = lhs_data["TrainLosses"]
    rhs_train_losses = rhs_data["TrainLosses"]
    #lhs_test_losses = lhs_data["TestLosses"]
    #rhs_test_losses = rhs_data["TestLosses"]


    cm = 1 / 2.54  # centimeters in inches
    plt.style.use('tableau-colorblind10')
    fig, axs = plt.subplots(ncols=2, figsize=(18*cm, 9*cm))
    l_eps = np.arange(1,lhs_data.shape[0]+1, 1)
    r_eps = np.arange(1, rhs_data.shape[0] + 1, 1)
    axs[0].plot(l_eps, lhs_valid_accs, label = "Validation - Baseline")
    axs[0].plot(r_eps, rhs_valid_accs, label = "Validation - Three Branch")
    axs[0].plot(l_eps, lhs_train_accs, label="Training - Baseline")
    axs[0].plot(r_eps, rhs_train_accs, label="Training - Three Branch")
    #axs[0].plot(l_eps, lhs_test_accs, label="Test_Ctrl")
    #axs[0].plot(r_eps, rhs_test_accs, label="Test_Final")
    axs[1].plot(l_eps, lhs_valid_losses, label="Validation - Baseline")
    axs[1].plot(r_eps, rhs_valid_losses, label="Validation - Three Branch")
    axs[1].plot(l_eps, lhs_train_losses, label="Training - Baseline")
    axs[1].plot(r_eps, rhs_train_losses, label="Training - Three Branch")
    #axs[1].plot(l_eps, lhs_test_losses, label="Test_Ctrl")
    #axs[1].plot(r_eps, rhs_test_losses, label="Test_Final")
    #axs[0].axhline(y=0.99, c="red")
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")

    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("BCE Loss")
    axs[0].legend()
    axs[1].legend()
    #plt.title( location + " exp" + lhs + " and " + rhs + " acc")
    plt.tight_layout()
    plt.savefig(output_save_path + location + " exp" + lhs + " and " + rhs + ".jpg", dpi=300)
    plt.show()