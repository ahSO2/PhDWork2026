#Calculate the optical flow for each sample in a given dataframe,
#then evaluate.

import cv2
import matplotlib.pyplot as plt
import pandas as pd

samples_sheet = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/Dataset/DatasetSplits/UpdatedTVTSplits/FinalSplit/Train.xlsx"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2"
data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/AllData_CorrectedWithVolcDict2Temporal"
mod = 1

df = pd.read_excel(samples_sheet)


#For each sample:

#Create a sequence of timestep images

#Calculate the FB optical flow with standard parameters

#Calculate the 1D flux

#Calculate the 2D flux - #TODO how does Plumetrack calculate this?

#Calculate the interpolation error

#TODO consider the impact of scaling image to UINT8

