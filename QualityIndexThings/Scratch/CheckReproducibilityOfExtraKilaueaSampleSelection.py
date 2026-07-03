import pandas as pd

run1_df = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment134/ExampleTrainWithAddData_134.xlsx")
run2_df = pd.read_excel("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment124/ExampleTrainWithAddData.xlsx")

run2_df["image_names_equal"] = run1_df["image_name"] == run2_df["image_name"]
print(run2_df["image_names_equal"].value_counts())