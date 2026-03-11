#Since Lascar video is shifted ba a few pixels (registration still holds)
#on Aug19th, I needed to add a new dictionary for the flank masks
import os

import pandas as pd

directory = "DatasetSplits/UpdatedTVTSplits/CrossValidationSplits"

for df_name in os.listdir(directory):
    print("Correcting:" + df_name)
    df = pd.read_excel(directory + "/" + df_name)
    #First I ran:
    #df.loc[(df['volcano_dictionary_name'] == "Lascar") & (df["image_date"]=="2022-08-19"), "volcano_dictionary_name"] = "LascarNineteenthAugust"
    #Then:
    df.loc[df['volcano_dictionary_name'] == "Lascar", "volcano_dictionary_name"] = "LascarView1"
    df.to_excel(directory + "/" + df_name)

