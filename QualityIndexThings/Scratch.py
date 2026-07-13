def map_image_name_to_time(image_name):
    date_str = image_name.split("_")[1][0:10]
    time_str = image_name.split("_")[1][11:17]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    return datetime_obj

def map_level_to_numeric(prediction_level):
    if prediction_level == "No":
        return 0.0
    elif prediction_level == "Minor":
        return 0.1
    elif prediction_level == "Not Calc":
        return 0.5
    elif prediction_level == "In Calc":
        return 0.75
    elif prediction_level == "Very":
        return 1.0
def plot_predictions_over_time(set_to_eval):
    times = set_to_eval["image_name"].apply(map_image_name_to_time)
    date = set_to_eval["image_name"][0].split("_")[1][0:10]
    print(times)
    print(date)
    manual_on_lens_label = set_to_eval["on_lens_level"]
    manual_on_lens_label_numeric = manual_on_lens_label.apply(map_level_to_numeric)
    manual_cloud_label = set_to_eval["cloud_level"]
    manual_cloud_label_numeric = manual_cloud_label.apply(map_level_to_numeric)
    on_lens_predictions_values = set_to_eval["rain_or_dirt_model_prediction"]
    fg_cloud_predictions_values = set_to_eval["cloud_model_prediction"]
    averager = [1 / 5] * 5
    fg_cloud_predictions_values = np.convolve(fg_cloud_predictions_values, averager, 'same')
    on_lens_predictions_values = np.convolve(on_lens_predictions_values, averager, 'same')

    fig, ax1 = plt.subplots(figsize=(18, 6))

    #ax1 = ax.twinx()
    ax1.scatter(times, manual_cloud_label_numeric, alpha=0.1, s=90, color="mediumpurple", label = "True cloud")
    ax1.scatter(times, manual_on_lens_label_numeric, alpha=0.1, s=25, color="deepskyblue", label = "True precip")
    ax1.set_yticks([0, 0.1, 0.5, 0.75, 1], labels=["No", "Minor", "Not Calc", "In Calc", "Very"])
    ax1.tick_params(axis='both', which='major', labelsize=15)

    # TODO alter ax.set_xlim(datetime(2022, 12, 11, 13, 0, 0), datetime(2022, 12, 11, 17, 0, 0))
    ax = ax1.twinx()
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax.set_xlim([times[0], times[times.shape[0] - 1]])
    ax.tick_params(axis='both', which='major', labelsize=15)
    #ax.set_xlim(datetime(2022, 4, 24, 17, 0, 0),times[times.shape[0] - 1] )

    ax.set_ylabel("Index Value", fontsize=20)
    #ax.set_xlabel("Time")
    #ax.set_title("Predicted Quality Levels: " + date, fontsize=20)

    ax.plot(times, on_lens_predictions_values, label="Precip index", color="royalblue", alpha=0.8)
    ax.plot(times, fg_cloud_predictions_values, label="Cloud index", color="darkviolet", alpha=0.8)
    ax.legend(fontsize=20)
    if 'other' in set_to_eval.columns:
        other_flag = np.array(set_to_eval["other"]).reshape(set_to_eval.shape[0])
        other_flag = np.where(other_flag == "Yes", 1, 0)
        flagged_indexes = np.where(other_flag == 1)[0]
        flagged_times = times[flagged_indexes]
        ax.vlines(flagged_times, ymin=0, ymax=1, colors=["orangered"] * flagged_times.shape[0], lw=1, alpha=0.2, label="Other")

    #ax.legend(fontsize=20)
    #ax1.legend()
    ax1.set_xlabel("Time (UTC)", fontsize=20)
    ax1.set_ylabel("True Class", fontsize=20)
    plt.tight_layout()
    fig.supylabel("A)", rotation=0, y=0.95, x=0, fontsize=20)
    plt.savefig("C:/Users/ggp24ash/Documents/Quality Index Write Up/Figures/Last 2022-12-11.png", dpi=1200)
    #plt.savefig(outputs_save_loc + "/" + date + "PredictionsOverTime" + ".png", dpi=fig.dpi)
    plt.show()


#set_to_eval_name = "C:/Users/ggp24ash/Documents/VolcanoData/ForEval/Reventador/2023/2023-10-06_Corrected_mod1/ImageNames.xlsx"
#set_to_eval_name = "D:/Cotopaxi/2024/2024-02-06_Corrected_mod1/ImageNamesSorted.xlsx"
set_to_eval_name = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/ModelApplicToFullDaysBackup/Lastarria_2022-12-11/ImageNamesSorted.xlsx"
set_to_eval = pd.read_excel(set_to_eval_name)
#outputs_save_loc = '/'.join(set_to_eval_name.split('/')[:-1]) + "/ModelOutputs"
#os.mkdir(outputs_save_loc)
outputs_save_loc = "OutputPlots/"

#Manual Labels:
on_lens_Yes = set_to_eval["rain_or_dirt_Yes"]
fg_cloud_Yes = set_to_eval["cloud_Yes"]
set_to_eval["on_lens_level_numeric"] = set_to_eval["on_lens_level"].apply(encode_typed_level)
set_to_eval["cloud_level_numeric"] = set_to_eval["cloud_level"].apply(encode_typed_level)
set_to_eval["obscurance_level_numeric"] = set_to_eval[["on_lens_level_numeric", "cloud_level_numeric"]].max(axis=1)
set_to_eval["obscurance_level"] = set_to_eval["obscurance_level_numeric"].apply(decode_numeric_level)
set_to_eval["obscurance"] = set_to_eval["obscurance_level"].apply(convert_level_to_y_n)
set_to_eval["obscurance_Yes"] = set_to_eval["obscurance"].apply(map_yes_no_to_binary)

#Model Predictions:
on_lens_predictions = set_to_eval["rain_or_dirt_model_prediction"]
fg_cloud_predictions = set_to_eval["cloud_model_prediction"]
set_to_eval["on_lens_prediction_binary"] = on_lens_predictions.apply(round_to_int)
set_to_eval["cloud_prediction_binary"] = fg_cloud_predictions.apply(round_to_int)
set_to_eval["obscurance_prediction_binary"] = set_to_eval[["on_lens_prediction_binary", "cloud_prediction_binary"]].max(axis=1)

#Plot the predictions
plot_predictions_over_time(set_to_eval)

#Filter out the 'Other' flagged data then evaluate
#Accuracy
if 'other' in set_to_eval.columns:
    print(set_to_eval.shape)
    set_to_eval_filtered = set_to_eval[set_to_eval["other"] != "Yes"]
    print(set_to_eval_filtered.shape)
else:
    set_to_eval_filtered = set_to_eval
on_lens_accuracy = accuracy(set_to_eval_filtered["on_lens_prediction_binary"], set_to_eval_filtered["rain_or_dirt_Yes"])
fg_cloud_accuracy = accuracy(set_to_eval_filtered["cloud_prediction_binary"], set_to_eval_filtered["cloud_Yes"])
overall_accuracy = accuracy(set_to_eval_filtered["obscurance_Yes"],set_to_eval_filtered["obscurance_prediction_binary"])
print("Accuracy:")
print("On Lens:" + str(round(on_lens_accuracy,2)))
print("FG Cloud:" + str(round(fg_cloud_accuracy,2)))
print("Overall Obscurance:" + str(round(overall_accuracy,2)))
print("Proportion of sequence manually classed as obs:")
print(set_to_eval_filtered["obscurance_Yes"].mean())
