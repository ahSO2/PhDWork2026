import matplotlib.pyplot as plt
import pandas as pd

sd = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/15 - FB Standard/StdFBOnGoodTrainSetSamples.xlsx")
small = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/16 - FB Gauss Noise/M0SD5_OnGoodTrainSetSamples.xlsx")
large = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/16 - FB Gauss Noise/IntRegionMean_OnGoodTrainSetSamples.xlsx")

props = [sd["prop"].dropna(), small["prop"].dropna(), large["prop"].dropna()]
labels = ["None", "Small", "Large"]
plt.boxplot(props, labels=labels)
plt.xlabel("Amount of Gauss Noise")
plt.ylabel("Prop of Plume Motion ID'd")
plt.show()
