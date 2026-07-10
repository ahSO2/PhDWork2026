#This work contains fine-tuned precipitation and cloud filtering neural network models,
#intended for quality classification of UV SO2 Camera video data.
#Copyright (C) 2026 Alyssa Heggison

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see https://www.gnu.org/licenses/.

#Please contact: asheggison1@sheffield.ac.uk, or through GitHub: https://github.com/AHeggison/QualityIndexModelsodels

import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings")
import Functions

#This script is adapted from Ayyadevara and Reddy's 2020 book:
#"Modern Computer Vision with PyTorch", Birmingham, Packt,
#available under a MIT licence at:
#https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch


#################################################
#####Key variables for this application###########
image_names_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/ModelApplicToFullDays_UpdatedJun26/Lastarria_2023-05-04/ImageNamesSorted.csv"
#image_names_path = "C:/Users/ggp24ash/PycharmProjects/PhDWork2026/QualityIndexThings/LightDilutionAndLabellerAgreement/ManualLabels/SelectedSamplesForLabelling_Batch1_CloudIndex.xlsx"
#image_names_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/ForORDA_Jun18th2026/CloudFullSplit/Cloud_Full_TestUnseen.csv"
image_names_df = pd.read_csv(image_names_path) #Spreadsheet containing name of each sample, plus associated timestep and off-band sample names
#Comment out: Optionally exclude/isolate Kilauea samples
#image_names_df = image_names_df[image_names_df["volcano_name"] != "Kilauea"]
#image_names_df.reset_index(inplace=True)
images_path = "D:/Lastarria/2023/2023-05-04_Corrected_mod1" #Folder storing image data
#images_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_Updated23rdJune26/MainDataset/"
temporal_images_path = "D:/Lastarria/2023/2023-05-04_Corrected_mod1/Temporal"
#temporal_images_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_Updated23rdJune26/MainDataset_AssociatedTemporal/"
#additional_images_path = "C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_Updated23rdJune26/Cloud_Full_TrainExpanded_AdditionalSamples/"
additional_images_path = None
chunk_size = 50 #Number of images to load and predict on at one time (set lower if memory is an issue)
outputs_save_path = "FinalModelsApplicationOutputs_RetrainedPrecipModel/FullDays/"  #Predictions saved here
#outputs_save_path = "FinalModelsApplicationOutputs_RetrainedPrecipModel/"
sensor_mark_masks_path = ("C:/Users/ggp24ash/Documents/Quality Index Write Up/Supplementary/Data_Updated23rdJune26/SensorMarkMasks/") #Path to folder containing masks used to infill small consistent marks on images (can be "None" if not req.)
#################################################

#Define the timestep sizes used for each model
timesteps_for_precip = ["minus_one_min_name", "image_name", "plus_one_min_name"]
timesteps_for_cloud = ["minus_ten_s_name", "image_name", "plus_ten_s_name"]

#Make use of GPU if available:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = "cpu"
print("Using device:" + device)
torch.cuda.empty_cache()

#Load the model definitions and trained weights
precip_model = Functions.get_triple_branched_resnet18(device)
cloud_model = Functions.get_triple_branched_resnet18(device)
print("Loading trained model weights:")
precip_model.load_state_dict(torch.load("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment239/OnLensSet_epoch1300.pth", weights_only=True))
cloud_model.load_state_dict(torch.load("C:/Users/ggp24ash/PycharmProjects/QualityIndexModels/SavedModelWeights/ObsCloudModel.pth", weights_only=True))
precip_model.eval()
cloud_model.eval()

#Read the data, and apply models:
print("Reading training data from: " + images_path)
#Loop over subsets of size "chunk", reducing the memory required
all_precip_predictions = []
all_cloud_predictions = []
n_chunks = math.ceil(image_names_df.shape[0]/chunk_size)
for chunk in range(1, n_chunks + 1):
    start_index = (chunk - 1) * chunk_size
    if chunk == n_chunks:
        end_index = image_names_df.shape[0] - 1
    else:
        end_index = (chunk * chunk_size) - 1
    print("Applying model to images indexed: " + str(start_index) + " to " + str(end_index))

    #Select the relevant sample names
    df_chunk = image_names_df.iloc[start_index:end_index + 1,].copy()
    df_chunk.reset_index(inplace=True, drop=True)

    #Time the model application to this chunk
    application_time_start = time.time()

    #Load this chunk of data
    eval_set_for_precip = Functions.ImageLoader(labels=df_chunk, timesteps=timesteps_for_precip, data_path = images_path, temporal_data_path=temporal_images_path, additional_data_path=additional_images_path, device= device, do_mask_sensor_marks=True, sensor_mark_masks_path= sensor_mark_masks_path)
    dataloader_precip = DataLoader(eval_set_for_precip, batch_size=1, shuffle=False, drop_last=False)
    eval_set_for_cloud = Functions.ImageLoader(labels=df_chunk, timesteps=timesteps_for_cloud, data_path=images_path, temporal_data_path=temporal_images_path,
                                                additional_data_path=additional_images_path, device=device,
                                                do_mask_sensor_marks=True,
                                                sensor_mark_masks_path=sensor_mark_masks_path)
    dataloader_cloud = DataLoader(eval_set_for_cloud, batch_size=1, shuffle=False, drop_last=False)

    #Iterating over each observation in the chunk
    for index, obs in enumerate(iter(dataloader_precip)):
            x_p = obs
            #Normalise the samples with ImageNet mean and standard deviation
            x_p_norm = Functions.scale_and_norm_batch(x_p, len(timesteps_for_precip), device)
            #Predict using the model, and move the output from the GPU if applicable
            predictions_p = precip_model(x_p_norm).cpu().detach().numpy()
            if chunk == 1 and index == 0:
                all_precip_predictions = predictions_p
            else:
                all_precip_predictions = np.concatenate((all_precip_predictions, predictions_p))

    for index, obs in enumerate(iter(dataloader_cloud)):
            x_c = obs
            #Repeat this process, predicting using the cloud model
            x_c_norm = Functions.scale_and_norm_batch(x_c, len(timesteps_for_cloud), device)
            predictions_c = cloud_model(x_c_norm).cpu().detach().numpy()
            if chunk == 1 and index == 0:
                all_cloud_predictions = predictions_c
            else:
                all_cloud_predictions = np.concatenate((all_cloud_predictions, predictions_c))
    application_time_end = time.time()
    application_time = application_time_end - application_time_start
    print("Time for application to chunk: " + str(np.round(application_time, 2)) + "s")

#Write out the predictions as columns of the image names dataframe
image_names_df["precipitation_prediction"] = all_precip_predictions
image_names_df["obs_cloud_prediction"] = all_cloud_predictions
image_names_df.to_excel(outputs_save_path + image_names_path.split("/")[-2] + ".xlsx")

