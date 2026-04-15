import pandas as pd
import matplotlib.pyplot as plt

pyr0 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/13 - Basic LK Movement Est/BasicLK_Pyr0.xlsx")
pyr1 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/13 - Basic LK Movement Est/BasicLK_Pyr1.xlsx")
pyr2 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/13 - Basic LK Movement Est/BasicLK_Pyr2.xlsx")
pyr3 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/13 - Basic LK Movement Est/BasicLK_Pyr3.xlsx")
pyr4 = pd.read_excel("C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/13 - Basic LK Movement Est/BasicLK_Pyr4.xlsx")

errs = [pyr0["LK_err"], pyr1["LK_err"], pyr2["LK_err"], pyr3["LK_err"], pyr4["LK_err"]]
labels = ["0", "1", "2", "3", "4"]
plt.boxplot(errs, labels=labels)
plt.xlabel("Pyramid Levels")
plt.ylabel("LK L1 Error")
plt.show()

def filter_out_Kilauea(df):
    print(len(df))
    df = df[~df["f1_name"].str.contains("Kilauea")]
    print(len(df))
    return df

#pyr0 = filter_out_Kilauea(pyr0)
#pyr1 = filter_out_Kilauea(pyr1)
#pyr2 = filter_out_Kilauea(pyr2)
#pyr3 = filter_out_Kilauea(pyr3)
#pyr4 = filter_out_Kilauea(pyr4)
props = [pyr0["prop"].dropna(), pyr1["prop"].dropna(), pyr2["prop"].dropna(), pyr3["prop"].dropna(), pyr4["prop"].dropna()]
labels = ["0", "1", "2", "3", "4"]
plt.boxplot(props, tick_labels=labels)
plt.xlabel("Pyramid Level")
plt.ylabel("Prop Plume Mvmt ID'd")
plt.show()

lens = [pyr0["mean_mag"].dropna(), pyr1["mean_mag"].dropna(), pyr2["mean_mag"].dropna(), pyr3["mean_mag"].dropna(), pyr4["mean_mag"].dropna()]
labels = ["0", "1", "2", "3", "4"]
plt.boxplot(lens, tick_labels=labels)
plt.xlabel("Pyramid Level")
plt.ylabel("Mean Velo Vector Magnitude")
plt.show()


