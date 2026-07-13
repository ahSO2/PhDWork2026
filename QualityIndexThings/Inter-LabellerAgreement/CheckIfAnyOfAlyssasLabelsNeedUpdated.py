#Check whether there is any overlap between the set of samples used for
#comparing manual label variation, and the samples for which I identified
#labels needed updating:
import pandas as pd

labels_to_update = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/DecideWhetherToUseCorrectedDataframes/CorrectedMistakes_NoDuplicates_wOldValsToCompare.xlsx")
labels_for_comparison = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/Inter-LabellerAgreement/ManualLabels/MergedLabels/merged_obs_cloud.xlsx")

names_to_update = labels_to_update["image_name"]
samples_in_comparison = labels_for_comparison["image_name"]

print(set(names_to_update).intersection(set(samples_in_comparison)))

#TODO Result: Three samples from the cloud set would have needed my precip label updating,
#but as this value isn't actually stored in the dataframe or used in any
#analysis, I don't make any upate.
