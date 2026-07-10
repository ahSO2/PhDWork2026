import pandas as pd

precip_last_CM = np.load(data_folder + "OnLensSet_BalancedLastarriaFinalTest_ConfMatData.npy")
cloud_last_CM = np.load(data_folder + "FGCloudSet_BalancedLastarriaFinalTest_ConfMatValues.npy")
obs_fulldays_CM = np.load(data_folder + "OverallObs_FullDays_ConfMatVals.npy")

fig, axs = plt.subplots(3)
plot1 = axs[0].imshow(precip_last_CM, cmap="Blues")
axs[0].set_yticks([0, 1], labels=["Yes", "No"])
plot2 = axs[1].imshow(cloud_last_CM, cmap="Purples", vmax=1)
axs[1].set_yticks([0, 1], labels=["Yes", "No"])
plot3 = axs[2].imshow(obs_fulldays_CM, cmap="Oranges")
axs[2].set_xticks([0,1,2,3,4], labels=["No", "Minor", "NotCalc", "InCalc", "Very"])
axs[2].set_yticks([0, 1], labels=["Yes", "No"])
axs[2].set_xlabel("True", fontsize=12)
axs[0].set_ylabel("A)", rotation=0, y=0.9, fontsize=10)
axs[1].set_ylabel("B)", rotation=0, y=0.9, fontsize=10)
axs[2].set_ylabel("C)", rotation=0, y=0.9, fontsize=10)
axs[0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.subplots_adjust(wspace=0, hspace=0)
fig.supylabel("Predicted", x=0.15, fontsize=12)
#fig.tight_layout()

for predicted in range(0, 5):
    for true in range(0, 2):
        text = axs[0].text(predicted, true, precip_last_CM[true, predicted],
                       ha="center", va="center", fontsize=8)
        text = axs[1].text(predicted, true, cloud_last_CM[true, predicted],
                           ha="center", va="center", fontsize=8)
        text = axs[2].text(predicted, true, obs_fulldays_CM[true, predicted],
                           ha="center", va="center", fontsize=8)
fig.colorbar(plot1, ax=axs[0], shrink=0.9)
fig.colorbar(plot2, ax=[axs[1]], shrink=0.9)
fig.colorbar(plot3, ax=axs[2], shrink=0.9)
#plt.show()
plt.savefig(data_folder + "FinalTestConfMats.jpg", dpi=300)
