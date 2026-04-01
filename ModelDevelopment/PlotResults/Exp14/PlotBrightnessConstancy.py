import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/14 - Brightness Constancy Eval/ResultsGoodTrainSetSamples.xlsx")

plt.boxplot(data["bc_err"])
plt.ylabel("BC error per pixel on avg")
plt.show()
