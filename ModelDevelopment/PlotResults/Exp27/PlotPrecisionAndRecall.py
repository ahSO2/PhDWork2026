import matplotlib.pyplot as plt
import pandas as pd

metrics = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/27 - Cross Bilateral Filter on Low Quality/PrecisionRecall.xlsx")


to_plot = [metrics["precision"], metrics["recall"].dropna()]
labels = ["precision", "recall"]
plt.boxplot(to_plot, labels=labels)
plt.xlabel("Metric")
plt.ylabel("Distribution (for good quality samples)")
plt.show()
