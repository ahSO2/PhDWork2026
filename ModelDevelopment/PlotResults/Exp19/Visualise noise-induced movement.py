import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/19 - More FB w Noise/Results.xlsx")

plt.boxplot(results["inxs_prop"].dropna())
plt.ylabel("Proportion of plume movement above 95th perc noise")
plt.show()

plt.boxplot(results["noise_velo"].dropna())
plt.ylabel("95th percentile of flow calculated from noise (pixels)")
plt.show()