import pandas as pd
import matplotlib.pyplot as plt

fb = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/15 - FB Standard/StdFBOnGoodTrainSetSamples.xlsx")
lk = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/13 - Basic LK Movement Est/BasicLK_Pyr4.xlsx")

props = [fb["prop"].dropna(), lk["prop"].dropna()]
labels = ["FB", "LK"]
plt.boxplot(props, labels=labels)
plt.xlabel("Motion Est Method")
plt.ylabel("Prop of Plume Motion ID'd")
plt.show()
