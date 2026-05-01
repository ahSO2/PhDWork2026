import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Boxplot the distribution of precision and recall at each threshold value

thresholds = [0, 0.25, 0.5, 0.9]

results_df = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/CrossValidFoldSegmentation/23 - Selecting seed points/CotopaxiLeftOutFoldResults.xlsx")

results = np.empty((results_df.shape[0], len(thresholds), 2))

precisions = results_df["precision"]
recalls = results_df["recall"]

p = precisions.str.strip("[]")
p = p.str.replace(",", "")
p_arr = p.str.split(" ", expand=True).to_numpy(dtype=float)
r = recalls.str.strip("[]")
r = r.str.replace(",", "")
r_arr = r.str.split(" ", expand=True).to_numpy(dtype=float)


#p = np.asarray(precisions)
#r = np.asarray(recalls)

results[:,:,0] = p_arr
results[:,:,1] = r_arr

data_p = []
data_r = []
for t_index in range(0, len(thresholds)):
    p = results[:,t_index,0]
    r = results[:,t_index,1]
    data_p.append(p)
    data_r.append(r[~np.isnan(r)])
labels = list(map(str, thresholds))
fig, axs = plt.subplots(ncols=2)
bplot_p = axs[0].boxplot(data_p, patch_artist=True, labels=labels)
c_p = ["c"] * len(data_p)
for patch, color in zip(bplot_p['boxes'], c_p):
    patch.set_facecolor(color)
bplot_r = axs[1].boxplot(data_r, patch_artist=True, labels=labels)
c_r = ["m"] * len(data_r)
for patch, color in zip(bplot_r['boxes'], c_r):
    patch.set_facecolor(color)
axs[0].set_xlabel("Threshold")
axs[1].set_xlabel("Threshold")
axs[0].set_ylabel("Precision")
axs[1].set_ylabel("Recall")
plt.show()