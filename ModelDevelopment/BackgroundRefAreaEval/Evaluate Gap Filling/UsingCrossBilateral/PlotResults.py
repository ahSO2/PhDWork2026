import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

results_folder = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/Cross-Bilateral"
'''
to_concat = []

#Read samples from folder
for comb_folder in os.listdir(results_folder):
    #For each folder extract sigma_s and sigma_r
    sigma_s = comb_folder.split("_")[0].split("-")[1]
    sigma_r = comb_folder.split("_")[1].split("-")[1]
    #Read the mean results file and extract the R_AP99 and min-location value
    comb_df = pd.read_csv(results_folder + "/" + comb_folder + "/" + "Mean_UnseenTest_mod1_quality-Good.csv")

    #Select the location-balanced mean R_AP99 value
    mean_data = comb_df[comb_df["fold"] == "location-balanced"]
    #mean_val = mean_data["R_AP99"].values[0]

    #Select the minimum R_AP99 value location
    sorted_combo_results = comb_df.sort_values(by=["R_AP99"], ascending=True)
    min_loc = sorted_combo_results["fold"].to_numpy()[0]
    min_R_AP99 = sorted_combo_results["R_AP99"].to_numpy()[0]

    new_row = mean_data.copy()
    new_row["sigma_s"] = sigma_s
    new_row["sigma_r"] = sigma_r
    new_row["min_R_AP99"] = min_R_AP99
    new_row["min_loc"] = min_loc

    to_concat.append(new_row)

results_df = pd.concat(to_concat, ignore_index=True)
#results_df.to_csv(results_folder + "/CompiledResults.csv")
'''

#Create a plot of R_AP99 against sigma_s and sigma_r, maybe colour points by location?

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador", "location-balanced"]
colours = ["tab:orange", "tab:purple", "tab:red", "tab:green", "tab:blue", "tab:pink"]

location_index = 0
#fig = plt.figure()
#axs = fig.add_subplot(projection="3d")
for location in locations:
    s_vals = []
    r_vals = []
    metric_vals = []
    for param_comb in os.listdir(results_folder):
        if os.path.isdir(results_folder + "/" + param_comb):
            print(param_comb)
            #Read the data for the chosen location and plot with the chosen colour
            sigma_s = param_comb.split("_")[0].split("-")[1]
            sigma_r = param_comb.split("_")[1].split("-")[1]
            # Read the mean results file and extract the R_AP99 and min-location value
            comb_df = pd.read_csv(results_folder + "/" + param_comb + "/" + "Mean_UnseenTest_mod1_quality-Good.csv")

            location_row = comb_df[comb_df["fold"] == location]
            mean_val = location_row["R_AP99"].values[0]

            s_vals.append(int(sigma_s))
            r_vals.append(int(sigma_r))
            metric_vals.append(mean_val)

    n_s = len(set(s_vals))
    n_r = len(set(r_vals))
    ordered_s_vals = sorted(set(s_vals), reverse=True)
    ordered_r_vals = sorted(set(r_vals))
    results_grid = np.empty((n_s, n_r))

    for index in range(0, len(s_vals)):
        s_index = ordered_s_vals.index(s_vals[index])
        r_index = ordered_r_vals.index(r_vals[index])
        results_grid[s_index, r_index] = metric_vals[index]
    fig, axs = plt.subplots()
    grid = axs.imshow(results_grid, cmap="inferno", vmax=1, vmin=0)

    for index in range(0, len(s_vals)):
        s_index = ordered_s_vals.index(s_vals[index])
        r_index = ordered_r_vals.index(r_vals[index])
        metric_val = metric_vals[index]
        if metric_val <= 0.5:
            color = "white"
        else:
            color = "black"
        text = axs.text(r_index, s_index, metric_val,
                                 ha="center", va="center", fontsize=8, color=color)

    axs.set_xticks(np.arange(len(ordered_r_vals)), labels=[str(x) for x in ordered_r_vals])
    axs.set_yticks(np.arange(len(ordered_s_vals)), labels=[str(x) for x in ordered_s_vals])
    axs.set_title(location)
    axs.set_xlabel("Range Smoothing")
    axs.set_ylabel("Spatial Smoothing")
    fig.colorbar(grid, ax=axs, location="right", shrink=0.7, pad=0.1,
                 label="Recall at precision 99%")
    #plt.show()
    plt.savefig(results_folder + "/" + "ResultPlot_" + location + ".png", dpi=300)

    #axs.scatter(s_vals, r_vals, metric_vals, c=colours[location_index])
    location_index += 1
#plt.show()


