#For each sample in the selected samples dataframe, retrieve the timestep
#image names and save them
import pandas as pd

sample_df_path = "SelectedSamplesForLabelling_Batch1_CloudIndex.xlsx"
all_samples_info_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/UpdatedCorrectedDataframes/AllCorrectedChunkandIndivLabels.xlsx"

samples_df = pd.read_excel(sample_df_path)
all_samples_df = pd.read_excel(all_samples_info_path)

selected_columns = all_samples_df[["image_name", "minus_one_min_name", "minus_one_min_name_B", "plus_one_min_name", "plus_one_min_name_B"]]
#Mixing up the samples, to confirm that the resulting df is actually being joined by image name
selected_columns = selected_columns.sample(frac=1, ignore_index=True)

output = pd.merge(samples_df, selected_columns, left_on='image_name', right_on='image_name', how='left')
print(output)
output.to_excel(sample_df_path)