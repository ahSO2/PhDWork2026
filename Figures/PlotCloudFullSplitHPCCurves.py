import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df1 = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment187/FGCloudSetFullSplit_Metrics.xlsx")
df2 = pd.read_excel("C:/USers/ggp24ash/Documents/HPC Outputs/Experiment189/FGCloudSetFullSplit_Metrics.xlsx")
save_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Version for submission - R2/Figures/Appendix_CloudFullTraining.jpg"

train_accs_no_aug = []
train_losses_no_aug = []
train_accs = []
train_losses = []
valid_accs = []
valid_losses = []

for df in [df1, df2]:
    train_accs += df["TrainAccuracies"].tolist()
    train_losses += df["TrainLosses"].tolist()
    train_accs_no_aug += df["TrainAccuracies_NoAug"].tolist()
    train_losses_no_aug += df["TrainLosses_NoAug"].tolist()
    valid_accs += df["ValidAccuracies"].tolist()
    valid_losses += df["ValidLosses"].tolist()

cm = 1 / 2.54  # centimeters in inches
fig, axs = plt.subplots(ncols=2, figsize=(18*cm, 9*cm))
eps = np.arange(1,len(train_accs_no_aug)+1, 1)
axs[0].plot(eps, train_accs, label="Train", alpha=0.8)
axs[0].plot(eps, train_accs_no_aug, label="Train (No Aug)", alpha=0.8)
axs[0].plot(eps, valid_accs, label="Valid", alpha=0.8)
axs[1].plot(eps, train_losses, label="Train", alpha=0.8)
axs[1].plot(eps, train_losses_no_aug, label="Train (No Aug)", alpha=0.8)
axs[1].plot(eps, valid_losses, label="Valid", alpha=0.8)
#axs[0].axhline(y=0.99, c="red")
axs[0].set_xlabel("Epoch")
axs[1].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[1].set_ylabel("BCE Loss")
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
