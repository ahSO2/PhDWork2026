#Read in index values
#Evaluate at standard threshold
#Calculate optimal threshold and evaluate
#Create a plot showing optimal threshold result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df_type = "excel"
indexes_df_path = "IndexValues/Lastarria_AllSamples_Unbalanced_QualityIndexes.xlsx"
results_save_path = "EvalandCompareResults/Lastarria_AllSamples_Unbalanced_ThresholdChoice.csv"
figure_save_name = "LastarriaOptimalThresholdFigure_TolVibrant.jpg"
original_thresh_v = 4
original_thresh_c = -0.5
thresh_v_range = [3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
thresh_c_range = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]

def calc_filtering_result(row):
    vis_check = row["visibility_good"]
    corr_check = row["correlation_good"]
    target = row["filtering_target"]

    if vis_check == False or corr_check == False:
        #If either check fails, the obs level is Yes
        filtering_result = "Yes"
    else:
        filtering_result = "No"

    if filtering_result == target:
        return 1
    else:
        return 0


def calc_balanced_acc(df):
    class_accs = []
    for true_class in ["No", "Yes"]:
        true_class_obs = df[df["filtering_target"] == true_class]
        true_class_obs_count = true_class_obs.shape[0]
        #print(str(true_class_obs_count) + " observations with target class " + true_class)
        #Calculate the accuracy
        acc = np.mean(true_class_obs["filtering_accuracy"].to_numpy())
        class_accs.append(acc)
        #print("Filtering acc for this class: " + str(acc))
    return sum(class_accs)/2

def evaluate_using_threshold(original_df, thresh_v, thresh_c):
    evaluation_df = original_df.copy()
    evaluation_df["visibility_good"] = evaluation_df["visibility_index"].copy().to_numpy() > thresh_v
    evaluation_df["correlation_good"] = evaluation_df["correlation_index"].copy().to_numpy() < thresh_c

    evaluation_df["filtering_accuracy"] = evaluation_df.apply(calc_filtering_result, axis=1)

    balanced_acc = calc_balanced_acc(evaluation_df)
    return np.round(balanced_acc, 4), evaluation_df

def plot_indexes_w_threshold(df_to_plot, thresh_v_to_plot, thresh_c_to_plot):
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(18*cm, 18*cm))
    custom = sns.color_palette(['#245642', '#236b52', '#fe960b', '#e75227', '#bc0000'])
    palette = sns.color_palette(['#228833', '#4477AA', '#CCBB44', "#EE6677", "#AA3377"])
    tol_vibrant = sns.color_palette(['#009988', '#33BBEE', '#EE3377', '#EE7733', '#CC3311'])
    s = sns.scatterplot(data=df_to_plot, y="visibility_index", x="correlation_index", hue="obscurance_level",
                    hue_order=["No", "Minor", "Not Calc", "In Calc", "Very"],
                    palette=tol_vibrant, alpha=0.3, ax=ax, linewidth=0)
    plt.legend(title="Obscurance Level")
    ax.axvline(x=thresh_c_to_plot, linestyle="--", color="black")
    ax.axhline(y=thresh_v_to_plot, linestyle="--", color="black")
    # ax.set_title("Physics-based Indexes For Labelled Lastarria Data", fontsize=15)
    ax.set_xlabel("Correlation Index", fontsize=15)
    ax.set_ylabel("Visibility Index", fontsize=15)

    visibility_values = df_to_plot["visibility_index"]
    correlation_values = df_to_plot["correlation_index"]
    visibility_range = visibility_values.max() - visibility_values.min()
    correlation_range = correlation_values.max() - correlation_values.min()
    thresh_v_prop = (thresh_v_to_plot - visibility_values.min())/visibility_range
    thresh_c_prop = (thresh_c_to_plot - correlation_values.min())/correlation_range

    ax.annotate("correlation = " + str(thresh_c_to_plot), xy=(thresh_c_prop, 0.95), rotation=90, xycoords = 'axes fraction', xytext = (-20, 0), textcoords = 'offset pixels', verticalalignment='top')
    ax.annotate("visibility = " + str(thresh_v_to_plot), xy=(0.95, thresh_v_prop), rotation=0, xycoords='axes fraction', xytext=(0, 50),
                textcoords='offset pixels', horizontalalignment="right")

    #plt.show()
    plt.savefig(results_save_path.split("/")[0] + "/" + figure_save_name, dpi=300)


if df_type == "excel":
    indexes_df = pd.read_excel(indexes_df_path)
else:
    indexes_df = pd.read_csv(indexes_df_path)
indexes_df["filtering_target"] = indexes_df["obscurance"]

#Evaluate the standard thresholds
bacc_og, eval_df_og = evaluate_using_threshold(indexes_df, thresh_v=original_thresh_v, thresh_c=original_thresh_c)
print("Balanced accuracy with standard threshold:")
print(bacc_og)

results_df = pd.DataFrame(columns=["visibility_threshold", "correlation_threshold", "balanced_accuracy"])
#Evaluate at a range of thresholds and save results to a dataframe
for thresh_v in thresh_v_range:
    for thresh_c in thresh_c_range:
        bacc, eval_df = evaluate_using_threshold(indexes_df, thresh_v, thresh_c)
        new_row = {"visibility_threshold":thresh_v,
                   "correlation_threshold":thresh_c,
                   "balanced_accuracy":bacc}
        results_df.loc[results_df.shape[0]] = new_row
if ".csv" in results_save_path:
    results_df.to_csv(results_save_path)
else:
    results_df.to_excel(results_save_path)

results_df = results_df.sort_values(by=["balanced_accuracy"], ascending=False)
best_thresh_v = results_df["visibility_threshold"].tolist()[0]
best_thresh_c = results_df["correlation_threshold"].tolist()[0]
print("Best threshold values identified: ")
print("For visibility " + str(best_thresh_v))
print("For correlation " + str(best_thresh_c))

#best_thresh_v, best_thresh_c = 3.1, -0.4
#Recalculate for the best combination, and produce a visualisation:
bacc, eval_df = evaluate_using_threshold(indexes_df, best_thresh_v, best_thresh_c)
plot_indexes_w_threshold(eval_df, best_thresh_v, best_thresh_c)






