import pandas as pd
import matplotlib.pyplot as plt

alpha0p2 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/11 - Varying Alpha in HnS/Alpha0p2.xlsx")
alpha1 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/11 - Varying Alpha in HnS/Alpha1.xlsx")
alpha5 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/11 - Varying Alpha in HnS/Alpha5.xlsx")
alphaadp = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/12 - Adaptive Alpha in HnS/AlphaAdp.xlsx")

rbs = [alpha0p2["r_b"], alpha1["r_b"], alpha5["r_b"], alphaadp["r_b"]]
labels = ["0.2", "1", "5", "Adp"]
plt.boxplot(rbs, labels=labels)
plt.xlabel("alpha")
plt.ylabel("Brightness Constancy Error")
plt.show()

props = [alpha0p2["prop"].dropna(), alpha1["prop"].dropna(), alpha5["prop"].dropna(), alphaadp["prop"].dropna()]
labels = ["0.2", "1", "5", "Adp"]
plt.boxplot(props, tick_labels=labels)
plt.xlabel("alpha")
plt.ylabel("Prop Plume Mvmt ID'd")
plt.show()


