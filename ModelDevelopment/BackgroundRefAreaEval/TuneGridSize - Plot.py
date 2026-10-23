import os
import matplotlib.pyplot as plt
import pandas as pd

folder_path = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/C-ADV_Tune_Grid_Size (P_AR90)/"

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
            rel_val = rel_data["P_AR99"].values[0]
            print(rel_val)
            new_row[location] = rel_val

        results_df.loc[len(results_df)] = new_row

results_df = results_df.sort_values(by=["grid_size"])
results_df.to_csv(folder_path + "CompiledResults.csv")


results_df = pd.read_csv(folder_path + "CompiledResults.csv")
fig, axs = plt.subplots()
grid_sizes = results_df["grid_size"]

for location in locations:
    plt.plot(grid_sizes, results_df[location], label=location)

axs.set_xlabel("Grid size (pixels)")
axs.set_ylabel("Precision at Recall=0.9")
axs.set_title("Plume Selection vs. Grid Size")
plt.legend()
plt.show()



