import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ks = [10, 50, 100, 200, 400]
comps = [1, 10, 50, 100]
llo = "Cotopaxi"

plot_df = pd.DataFrame(columns=["k", "c", "m", "sd"])
for k in ks:
    for c in comps:
        df = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/24 - Superpixel algorithm/" + llo + "LeftOutFold_k" + str(k) + "_c" + str(c) + ".xlsx")
        values = df["IOU"]
        m = np.mean(values)
        sd = np.std(values)
        plot_df.loc[len(plot_df)] = [k, c, m, sd]

sns.scatterplot(data=plot_df, x="m", y="sd", hue="c", size="k",sizes=(10, 200), legend="full")
plt.show()



