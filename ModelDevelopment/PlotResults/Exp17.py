import pandas as pd
import matplotlib.pyplot as plt

whole_sky_ratios = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/17 - BG Ratio Constancy/CalculatedRatios.xlsx")
bg_ratios = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/17 - BG Ratio Constancy/CalculatedBgRatios.xlsx")

sds = [whole_sky_ratios["bg_ratio_std"], bg_ratios["bg_ratio_std"]]
labels = ["Whole sky", "Non-plume only"]
plt.boxplot(sds, labels=labels)
plt.xlabel("Area considered")
plt.ylabel("SD of bandA/B ratio")
plt.show()