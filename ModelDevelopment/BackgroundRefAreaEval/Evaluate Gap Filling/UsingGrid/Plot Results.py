import os
import matplotlib.pyplot as plt
import pandas as pd

folder_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/GridMean/"

locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador", "location-balanced"]

results_df = pd.DataFrame(columns=["grid_size"] + locations)

for file_name in os.listdir(folder_path):
    if "mean" in file_name:
        rel_text = file_name.split("_")[1]
        grid_size = int(rel_text.split("-")[1])

        data = pd.read_csv(folder_path + file_name)

        new_row = {'grid_size':grid_size}

        for location in locations:
            rel_data = data[data["fold"] == location]
            rel_val = rel_data["R_AP99"].values[0]
            new_row[location] = rel_val

        results_df.loc[len(results_df)] = new_row

results_df = results_df.sort_values(by=["grid_size"])
results_df.to_csv(folder_path + "CompiledResults.csv")


results_mean_only = pd.read_csv("C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/GridMean/" + "CompiledResults.csv")
results_with_sd = pd.read_csv("C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/GridMeanMinusSD/" + "CompiledResults.csv")
fig, axs = plt.subplots()
grid_sizes = results_mean_only["grid_size"]
grid_sizes_2 = results_with_sd["grid_size"]


colors = ["tab:pink", "tab:green", "tab:blue", "tab:orange", "tab:olive", "tab:purple"]
c_i = 0
for location in locations:
    plt.plot(grid_sizes, results_mean_only[location], label=location + "_m", c=colors[c_i])
    plt.plot(grid_sizes_2, results_with_sd[location], label=location + "_sd", linestyle="dashed",  c=colors[c_i])
    c_i += 1

axs.set_xlabel("Grid size (pixels)")
axs.set_ylabel("Recall at precision 99%")
axs.set_title("Plume Selection vs. Grid Size")
plt.legend(bbox_to_anchor=(1.1, 1))
plt.tight_layout()

plt.savefig("C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/GridSmoothingCompariosn.png", dpi=300)