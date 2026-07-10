#For each sample, want to create a dataframe containing
#my label, TW label, TP label, and my model predictions.
import pandas as pd

target = "precipitation"
zooniverse_labels_path = "FinalZooniverseExports/precipitation-labelling-classifications.csv"
my_existing_labels_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/UpdatedCorrectedDataframes/AllCorrectedChunkandIndivLabels.xlsx"
model_predictions_path = "/QualityIndexThings/CalculatingMetrics/FinalModelsApplicationOutputs_Original/ManualLabellingVariation/SelectedSamplesForLabelling_Batch1_PrecipIndex.xlsx"

base_df = pd.read_excel(model_predictions_path)
my_labels_df = pd.read_excel(my_existing_labels_path)
zooniverse_labels = pd.read_csv(zooniverse_labels_path)

#Merge the base df with my labels
#my_labels_selected_columns = my_labels_df[["image_name", "on_lens_level", "cloud_level"]]
#base_plus_my_labels = pd.merge(base_df, my_labels_selected_columns, left_on='image_name', right_on='image_name', how='left')
#base_plus_my_labels.rename({"on_lens_level":"precipitation_level_AH", "cloud_level":"obs_cloud_level_AH"}, inplace=True)
#base_plus_my_labels.drop(["on_lens_level_y", "on_lens_level_x", "cloud_level"])
#print(base_plus_my_labels.columns)

#Actually my selected samples dataframes already had my labels saved
if target == "precipitation":
    base_plus_my_labels = base_df.copy()
    base_plus_my_labels.rename(columns={"on_lens_level":"precipitation_level_AH"}, inplace=True)
    print("Renaming my precip labels column.")
else:
    base_plus_my_labels = base_df.copy()
    base_plus_my_labels.rename(columns={"cloud_level":"obs_cloud_level_AH"}, inplace=True)
    print("Renaming my cloud labels column")


#Next merge with the Toms' labels:
def map_subject_data_to_image_name(subject_data):
    image_name = subject_data.split('"')[7]
    return image_name

def map_annotation_to_classification(annotation):
    if target == "precipitation":
        classification = annotation.split('"')[11]
    if target == "obs_cloud":
        classification = annotation.split('"')[13]
    return classification

zooniverse_labels["image_name"] = zooniverse_labels["subject_data"].apply(map_subject_data_to_image_name)
zooniverse_labels["annotator_level"] = zooniverse_labels["annotations"].apply(map_annotation_to_classification)

TW_labels = zooniverse_labels[zooniverse_labels["user_name"]=="twvolc"][["image_name", "annotator_level"]]
print(TW_labels.shape)
TW_labels["duplicated"] = TW_labels.duplicated(["image_name"])
if target == "precipitation":
    print("Removing duplicate precipitation label by TW")
    TW_labels = TW_labels[:-1] #Dropping the last label by TW which is an exact duplicate of an earlier label
print(TW_labels["duplicated"].value_counts())

TP_labels = zooniverse_labels[zooniverse_labels["user_name"]=="tpering"][["image_name", "annotator_level"]]
print(TP_labels.shape)

all_annotator_labels = pd.merge(base_plus_my_labels, TW_labels, left_on='image_name', right_on='image_name', how='left')
all_annotator_labels.rename(columns={"annotator_level":target+"_level_TW"}, inplace=True)
print(all_annotator_labels.columns)
all_annotator_labels = pd.merge(all_annotator_labels, TP_labels, left_on='image_name', right_on='image_name', how='left')
all_annotator_labels.rename(columns={"annotator_level":target+"_level_TP"}, inplace=True)
all_annotator_labels.to_excel("MergedLabels/" + "merged_" + target + ".xlsx")


