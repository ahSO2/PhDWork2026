import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/18 - LK Filtering/Baseline.xlsx")
status_filtered = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/18 - LK Filtering/StatusFiltered.xlsx")
ev_filtered = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/18 - LK Filtering/EigFiltered.xlsx")

props = [baseline["prop"].dropna(), status_filtered["prop"].dropna(), ev_filtered["prop"].dropna()]
labels = ["Baseline", "Status", "Eigenvalue"]
plt.boxplot(props, labels=labels)
plt.xlabel("Type of filtering")
plt.ylabel("Proportion of plume ID'd as moving")
plt.show()

mags = [baseline["masked_mean"], status_filtered["masked_mean"], ev_filtered["masked_mean"]]
labels = ["Baseline", "Status", "Eigenvalue"]
plt.boxplot(mags, labels=labels)
plt.xlabel("Type of filtering")
plt.ylabel("Mean velo vector magnitude within plume")
plt.show()