import matplotlib.pyplot as plt
import os
import pandas as pd

results_folder = "C:/Users/ggp24ash/Documents/Scratch Data/BackgroundRefAreaSelection/CVFolds/Activations/Smoothing/MorphTrans"
locations = ["Cotopaxi", "Kilauea", "Lascar", "Merapi", "Reventador", "location-balanced"]

colours = ["tab:orange", "tab:purple", "tab:red", "tab:green", "tab:blue", "tab:pink"]

fig, axs = plt.subplots()

#For each location
colour_index = 0
for location in locations:
    all_sheets = os.listdir(results_folder)


    #Filter for mean results for ero
    ero_metrics = []
    ero_disk_radii = []
    for sheet in all_sheets:
        if "mean" in sheet:
            if "Ero" in sheet:
                applic_df = pd.read_csv(results_folder + "/" + sheet)
                location_row = applic_df[applic_df["fold"]==location]
                R_AP99 = location_row["R_AP99"].values[0]
                #Add metric and disc radius to lists
                r = int(sheet.split("_")[0].split("-")[1])
                ero_metrics.append(R_AP99)
                ero_disk_radii.append(r)

    #Filter for mean results for opening
    opening_metrics = []
    opening_disk_radii = []
    for sheet in all_sheets:
        if "mean" in sheet:
            if "Open" in sheet:
                applic_df = pd.read_csv(results_folder + "/" + sheet)
                location_row = applic_df[applic_df["fold"] == location]
                R_AP99 = location_row["R_AP99"].values[0]
                # Add metric and disc radius to lists
                r = int(sheet.split("_")[0].split("-")[1])
                opening_metrics.append(R_AP99)
                opening_disk_radii.append(r)

    #Sort the lists by x-value for plotting
    ero_vals = zip(ero_metrics, ero_disk_radii) #Join the lists to create an iterable of tuples
    ero_vals = sorted(ero_vals, key=lambda x:x[1]) #Sort the tuples by the second element
    ero_metrics, ero_disk_radii = zip(*ero_vals)

    open_vals = zip(opening_metrics, opening_disk_radii)
    open_vals = sorted(open_vals, key=lambda x:x[1])
    opening_metrics, opening_disk_radii = zip(*open_vals)

    #Plot metric against disc radius for each type of morph trans
    axs.plot(ero_disk_radii, ero_metrics, color=colours[colour_index], linestyle="--", label=location + "_Erosion")
    axs.plot(opening_disk_radii, opening_metrics, color=colours[colour_index], label=location + "_Opening")
    colour_index += 1

axs.set_xlabel("Disk Radius")
axs.set_ylabel("Recall at Precision 99%")
plt.legend(bbox_to_anchor=(1, 0.8))
plt.tight_layout()
plt.show()