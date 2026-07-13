#Read in three predictions dataframes
#For each specify the target variable
#Calculate and plot subcategory confusion matrices
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Version for submission - R2/Figures/Figure6.jpg"
df1 = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/Precipitation_Full_TestSeen.xlsx")
target1 = "precipitation"
title1 = "A) Precipitation"
df2 = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/Cloud_Full_TestSeen_ExcludingKilauea.xlsx")
target2 = "obs_cloud"
title2 = "B) Cloud - Excluding Kilauea"
df3 = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/Cloud_Full_TestSeen_KilaueaOnly.xlsx")
target3 = "obs_cloud"
title3 = "C) Cloud - Kilauea Only"

subclasses = {"No":"No", "Minor":"No", "Not Calc":"Yes", "In Calc":"Yes", "Very":"Yes"}

def calculate_binary_prediction(value):
    if value >= 0.5:
        return "Yes"
    else:
        return "No"
def calculate_subcat_conf_vals(df, target):
    '''Calculate subcat confusion matrix for the target variable, in both
    raw sample count form, and column-normalised.'''

    counts_grid = np.zeros((2, 5)) #Shape (binary prediction, labelled subclass)
    df["prediction_yn"] = df[target + "_prediction"].apply(calculate_binary_prediction)

    #TODO need to calculate binary prediction, I am currently just plotting the true binary class!
    class_index = 0
    for subclass in subclasses:
        subclass_samples = df[df[target + "_level"] == subclass]
        subclass_count = subclass_samples.shape[0]
        positive_predictions = subclass_samples[subclass_samples["prediction_yn"]=="Yes"]
        counts_grid[0, class_index] = positive_predictions.shape[0]
        negative_predictions = subclass_samples[subclass_samples["prediction_yn"]=="No"]
        counts_grid[1, class_index] = negative_predictions.shape[0]
        class_index += 1

    column_normalised = counts_grid.copy()
    for column_index in range(0, 5):
        column_total = counts_grid[0, column_index] + counts_grid[1, column_index]
        if column_total == 0:
            pass
        else:
            column_normalised[0, column_index] = counts_grid[0, column_index]/column_total
            column_normalised[1, column_index] = counts_grid[1, column_index] / column_total
    print(counts_grid)
    return counts_grid, column_normalised

counts1, norm1 = calculate_subcat_conf_vals(df1, target1)
counts2, norm2 = calculate_subcat_conf_vals(df2, target2)
counts3, norm3 = calculate_subcat_conf_vals(df3, target3)

cm = 1 / 2.54
fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(15*cm, 18*cm))
plot1 = axs[0].imshow(norm1, cmap="Oranges", vmin=0, vmax=1)
plot2 = axs[1].imshow(norm2, cmap="Oranges", vmin=0, vmax=1)
plot3 = axs[2].imshow(norm3, cmap="Oranges", vmin=0, vmax=1)

count_dfs = [counts1, counts2, counts3]
norm_dfs = [norm1, norm2, norm3]
titles = [title1, title2, title3]
for axis_no in [0, 1, 2]:
    for predicted in range(0, 5):
        for true in range(0, 2):
            count = count_dfs[axis_no][true, predicted]
            norm_val = norm_dfs[axis_no][true, predicted]
            if norm_val <= 0.5:
                color = "#842804"
            else:
                color = "white"
            text = axs[axis_no].text(predicted, true, int(count),
                               ha="center", va="center", fontsize=8, color=color)
            axs[axis_no].set_yticks([0, 1], labels=["Yes", "No"])
            #axs[axis_no].set_xticks([0, 1, 2, 3, 4], labels=["No", "Minor", "NotCalc", "InCalc", "Very"])
            #axs[axis_no].set_ylabel("Predicted")
            axs[axis_no].set_title(titles[axis_no], loc="left", fontsize=10, pad=-5)

axs[2].set_xticks([0, 1, 2, 3, 4], labels=["No", "Minor", "NotCalc", "InCalc", "Very"])
axs[0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[1].set_ylabel("Predicted", fontsize=15, labelpad=15)
axs[2].set_xlabel("Manual Label", fontsize=15)
plt.subplots_adjust(wspace=0, hspace=0.3)
fig.colorbar(plot3, ax=axs[0:3], location="bottom", shrink=0.7, pad=0.15,
             label="Recall per sub-category (Column normalised)")
#plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

