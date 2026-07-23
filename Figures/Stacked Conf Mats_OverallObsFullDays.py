#Read in three predictions dataframes
#For each specify the target variable
#Calculate and plot subcategory confusion matrices
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Version for submission - R2/Figures/OverallObs_FullDays_ByLocation.jpg"
all_data = pd.read_csv("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/EvaluatingOnFullDays/FullDays_StdThreshold_OverallObsPredictions.csv")
Cotopaxi = all_data[all_data["image_name"].str.contains("Cotopaxi")]
Cotopaxi.reset_index(inplace=True)
Kilauea = all_data[all_data["image_name"].str.contains("Kilauea")]
Kilauea.reset_index(inplace=True)
Lastarria = all_data[all_data["image_name"].str.contains("Lastarria")]
Lastarria.reset_index(inplace=True)
Merapi = all_data[all_data["image_name"].str.contains("Merapi")]
Merapi.reset_index(inplace=True)
Reventador = all_data[all_data["image_name"].str.contains("Reventador")]
Reventador.reset_index(inplace=True)


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

countsC, normC = calculate_subcat_conf_vals(Cotopaxi, "obscurance")
countsK, normK = calculate_subcat_conf_vals(Kilauea, "obscurance")
countsL, normL = calculate_subcat_conf_vals(Lastarria, "obscurance")
countsM, normM = calculate_subcat_conf_vals(Merapi, "obscurance")
countsR, normR = calculate_subcat_conf_vals(Reventador, "obscurance")

cm = 1 / 2.54
fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(15*cm, 37*cm))
plot1 = axs[0].imshow(normC, cmap="Oranges", vmin=0, vmax=1)
plot2 = axs[1].imshow(normK, cmap="Oranges", vmin=0, vmax=1)
plot3 = axs[2].imshow(normL, cmap="Oranges", vmin=0, vmax=1)
plot4 = axs[3].imshow(normM, cmap="Oranges", vmin=0, vmax=1)
plot5 = axs[4].imshow(normR, cmap="Oranges", vmin=0, vmax=1)

count_dfs = [countsC, countsK, countsL, countsM, countsR]
norm_dfs = [normC, normK, normL, normM, normR]
titles = ["A) Cotopaxi", "B) Kilauea", "C) Lastarria", "D) Merapi", "E) Reventador"]
for axis_no in [0, 1, 2, 3, 4]:
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
            axs[axis_no].set_title(titles[axis_no], loc="left", fontsize=15, pad=-10)

axs[4].set_xticks([0, 1, 2, 3, 4], labels=["No", "Minor", "NotCalc", "InCalc", "Very"])

axs[0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[2].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[3].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)



axs[2].set_ylabel("Predicted", fontsize=17, labelpad=15)
axs[4].set_xlabel("Manual Label", fontsize=17)
plt.subplots_adjust(wspace=0, hspace=0.3)
fig.colorbar(plot5, ax=axs[0:5], location="bottom", shrink=0.5, pad=0.08,
             label="Recall per sub-category (Column normalised)")
#plt.tight_layout()
plt.savefig(save_path, dpi=500)
plt.show()

