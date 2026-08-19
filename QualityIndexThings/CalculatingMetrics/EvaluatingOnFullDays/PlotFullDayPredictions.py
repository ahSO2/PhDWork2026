from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd

day_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_RetrainedPrecipModel/FullDays/Reventador_2025-07-14.xlsx"
df = pd.read_excel(day_path)
def map_image_name_to_time(image_name):
    date_str = image_name.split("_")[1][0:10]
    time_str = image_name.split("_")[1][11:17]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    return datetime_obj
def map_level_to_numeric(level):
    if level == "No":
        return 0.0
    elif level == "Minor":
        return 0.2
    elif level == "Not Calc":
        return 0.6
    elif level == "In Calc":
        return 0.8
    elif level == "Very":
        return 1.0
    else:
        print("Error in level given")

#Get datetimes from image names
times = df["image_name"].apply(map_image_name_to_time)

#Map target levels to numeric height for plotting
manual_precip_vals = df["precipitation_level"].apply(map_level_to_numeric)
manual_cloud_vals = df["obs_cloud_level"].apply(map_level_to_numeric)

precip_predictions = df["precipitation_prediction"]
cloud_predictions = df["obs_cloud_prediction"]

averager = [1 / 5] * 5
precip_predictions = np.convolve(precip_predictions, averager, 'same')
cloud_predictions = np.convolve(cloud_predictions, averager, 'same')

cm = 1 / 2.54
fig, ax1 = plt.subplots(figsize=(24*cm, 8*cm))
ax1.fill_between(times, manual_precip_vals, alpha=0.4, color="#298c8c")
ax1.fill_between(times, manual_cloud_vals, alpha=0.2, color="#800074")
ax1.set_yticks([0, 0.2, 0.6, 0.8, 1], labels=["No", "Minor", "Not Calc", "In Calc", "Very"])
ax1.set_ylim([-0.05, 1.05])
ax1.tick_params(axis='both', which='major', labelsize=7)

ax = ax1.twinx()
ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
ax.set_xlim([times[0], times[times.shape[0] - 1]])
#ax.set_xlim(datetime(2022, 4, 24, 17, 0, 0), times[times.shape[0] - 1])

ax.set_ylim([-0.05,1.05])
ax.tick_params(axis='both', which='major', labelsize=7)
ax.plot(times, precip_predictions, color="#298c8c", label="precipiation", linewidth=1)
ax.plot(times, cloud_predictions, color="#800074", label="cloud", linewidth=1)
ax1.plot(times, np.ones_like(precip_predictions) * 0.5, color="#d44608", linestyle="--")

if 'other' in df.columns:
    other_flag = np.array(df["other"]).reshape(df.shape[0])
    other_flag = np.where(other_flag == "Yes", 1, 0)
    flagged_indexes = np.where(other_flag == 1)[0]
    flagged_times = times[flagged_indexes]
    ax1.vlines(flagged_times, ymin=0, ymax=1, colors=["orange"] * flagged_times.shape[0], lw=1, alpha=1,
              label="Other")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

leg = ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.75), fontsize=10, title="Target Variable")
#ax.legend(fontsize=10, title="Target Variable")

ax.set_ylabel("Index Value", fontsize=10)
ax1.set_xlabel("Time (UTC)", fontsize=10)
ax1.set_ylabel("True Class", fontsize=10)
plt.savefig("FullDayPlots/" + day_path.split("/")[-1][:-5] + ".jpg", dpi=300, bbox_inches="tight",
    pad_inches=0.5*cm,)
plt.show()

