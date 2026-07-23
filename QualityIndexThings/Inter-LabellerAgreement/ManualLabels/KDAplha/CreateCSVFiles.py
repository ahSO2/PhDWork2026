#Creating csv files to upload to KACalculator
import pandas as pd

labels = pd.read_excel("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/Inter-LabellerAgreement/ManualLabels/MergedLabels_WUpdatedModelPredictions/consensus_calc_precipitation.xlsx")
target = "precipitation"
labellers = ["AH", "TW", "TP"]
labels = labels[labels["image_name"].str.contains("Cotopaxi")]

columns_to_select = []
for labeller in labellers:
    columns_to_select.append(target + "_binary_" + labeller)
for_csv = labels[columns_to_select]
for_csv.to_csv("CSVFilesToUpload/" + target + "_binary_labels_CotopaxiOnly.csv", header=None, index=False)


