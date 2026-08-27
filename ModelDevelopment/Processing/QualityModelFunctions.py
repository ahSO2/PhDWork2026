import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
import sys
sys.path.append("C:/Users/ggp24ash/PycharmProjects/PhDWork2026/")
import VolcDictionaryWithCorrectClears

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

def get_pretrained_resnet18_model_only():
    model = models.resnet18(weights = "IMAGENET1K_V1")
    #Fix the weights so thay are not trained
    for param in model.parameters():
        param.requires_grad = False
    return model
class TripleBranchedModel(torch.nn.Module):
    def __init__(self):
        super(TripleBranchedModel, self).__init__()
        pretrained_resnet18 = get_pretrained_resnet18_model_only()
        #Branch t=0
        self.t0_conv1 = pretrained_resnet18.conv1
        self.t0_bn1 = pretrained_resnet18.bn1
        self.t0_relu = pretrained_resnet18.relu
        self.t0_maxpool = pretrained_resnet18.maxpool
        self.t0_layer1 = pretrained_resnet18.layer1
        self.t0_layer2 = pretrained_resnet18.layer2
        self.t0_layer3 = pretrained_resnet18.layer3
        self.t0_layer4 = pretrained_resnet18.layer4

        self.m1_conv1 = pretrained_resnet18.conv1
        self.m1_bn1 = pretrained_resnet18.bn1
        self.m1_relu = pretrained_resnet18.relu
        self.m1_maxpool = pretrained_resnet18.maxpool
        self.m1_layer1 = pretrained_resnet18.layer1
        self.m1_layer2 = pretrained_resnet18.layer2
        self.m1_layer3 = pretrained_resnet18.layer3
        self.m1_layer4 = pretrained_resnet18.layer4

        self.p1_conv1 = pretrained_resnet18.conv1
        self.p1_bn1 = pretrained_resnet18.bn1
        self.p1_relu = pretrained_resnet18.relu
        self.p1_maxpool = pretrained_resnet18.maxpool
        self.p1_layer1 = pretrained_resnet18.layer1
        self.p1_layer2 = pretrained_resnet18.layer2
        self.p1_layer3 = pretrained_resnet18.layer3
        self.p1_layer4 = pretrained_resnet18.layer4

        #Then concatenate
        self.t0_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.m1_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.p1_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(nn.Flatten(),
                nn.Linear(512 * 3, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid())

    def forward(self, x):
        #This model contains redundant definitions for the
        #"m1" and "p1" branches (as these weights are not
        #trained, we just pass each timestep data through the
        #equivalent "t0" branch). The redundant definitions
        #were mistakenly not removed before training of the
        #classifiers. Though this has no purpose, and
        #will be corrected in any subsequent versions, it
        #should not affect results (and doesn't affect the
        #parameter count).

        #[obsevation, timestep, channel, height, width]
        branch_t0 = self.t0_conv1(x[:,1,:,:,:])
        branch_t0 = self.t0_bn1(branch_t0)
        branch_t0 = self.t0_relu(branch_t0)
        branch_t0 = self.t0_maxpool(branch_t0)
        branch_t0 = self.t0_layer1(branch_t0)
        branch_t0 = self.t0_layer2(branch_t0)
        branch_t0 = self.t0_layer3(branch_t0)
        branch_t0 = self.t0_layer4(branch_t0)
        branch_t0 = self.t0_avgpool(branch_t0)

        branch_m1 = self.t0_conv1(x[:, 0, :, :, :])
        branch_m1 = self.t0_bn1(branch_m1)
        branch_m1 = self.t0_relu(branch_m1)
        branch_m1 = self.t0_maxpool(branch_m1)
        branch_m1 = self.t0_layer1(branch_m1)
        branch_m1 = self.t0_layer2(branch_m1)
        branch_m1 = self.t0_layer3(branch_m1)
        branch_m1 = self.t0_layer4(branch_m1)
        branch_m1 = self.t0_avgpool(branch_m1)

        branch_p1 = self.t0_conv1(x[:, 2, :, :, :])
        branch_p1 = self.t0_bn1(branch_p1)
        branch_p1 = self.t0_relu(branch_p1)
        branch_p1 = self.t0_maxpool(branch_p1)
        branch_p1 = self.t0_layer1(branch_p1)
        branch_p1 = self.t0_layer2(branch_p1)
        branch_p1 = self.t0_layer3(branch_p1)
        branch_p1 = self.t0_layer4(branch_p1)
        branch_p1 = self.t0_avgpool(branch_p1)

        comb_output = self.fc(torch.cat((branch_m1, branch_t0, branch_p1), dim=1))
        return comb_output

def get_triple_branched_resnet18(device):
    model = TripleBranchedModel()
    return model.to(device)

def map_yes_no_to_binary(value):
    if value == "Yes":
        return 1
    if value == "No":
        return 0

def convert_outcome_to_binary(labels_df, column_name):
    new_column = labels_df[column_name].apply(map_yes_no_to_binary)
    labels_df[column_name + "_Yes"] = new_column
    return labels_df

def read_data(bandA_list, bandB_list, temporal_array, timesteps_provided):
    x_precip = []
    x_cloud = []

    timesteps_for_precip = [-60, 0, 60]
    timesteps_for_cloud = [-10, 0, 10]

    for index in range(0, len(bandA_list)):
        # Array to store data for this sample.
        # [observation_index][timestep_index, channel, 486, 648]
        this_obs_x_precip = np.zeros((3, 3, 486, 648))
        this_obs_x_cloud = np.zeros((3, 3, 486, 648))

        step_count = -1
        for step in timesteps_for_precip:
            step_count += 1
            if step == 0:
                this_obs_x_precip[step_count, 0, :, :] = bandA_list[index]
                this_obs_x_precip[step_count, 1, :, :] = bandB_list[index]
            else:
                index_in_provided = timesteps_provided.index(step)
                this_obs_x_precip[step_count, 0, :, :] = temporal_array[index, index_in_provided, 0, :, :]
                this_obs_x_precip[step_count, 1, :, :] = temporal_array[index, index_in_provided, 1, :, :]

        step_count = -1
        for step in timesteps_for_cloud:
            step_count += 1
            if step == 0:
                this_obs_x_cloud[step_count, 0, :, :] = bandA_list[index]
                this_obs_x_cloud[step_count, 1, :, :] = bandB_list[index]
            else:
                index_in_provided = timesteps_provided.index(step)
                this_obs_x_cloud[step_count, 0, :, :] = temporal_array[index, index_in_provided, 0, :, :]
                this_obs_x_cloud[step_count, 1, :, :] = temporal_array[index, index_in_provided, 1, :, :]

        x_precip.append(this_obs_x_precip)
        x_cloud.append(this_obs_x_cloud)
    return x_precip, x_cloud
class ImageLoader(Dataset):
    def __init__(self, device, indexes, bandA_list, bandB_list, temporal_available, temporal_array, timesteps_provided):

        x_precip, x_cloud = read_data(bandA_list, bandB_list, temporal_array, timesteps_provided)
        self.image_indexes = indexes
        self.n_timesteps = 3 #Per model
        self.X_P = torch.tensor(np.array(x_precip)).float()
        self.X_C = torch.tensor(np.array(x_cloud)).float()
        self.apply_model = torch.tensor(np.array(temporal_available))

        self.device = device
    def __len__(self):
        # Return number of input observations
        return len(self.X_P)

    def __getitem__(self, index):
        return self.X_P[index].to(self.device), self.X_C[index].to(self.device), self.apply_model[index], self.image_indexes[index]

def scale_and_normalise_image(observation, device):
    '''Scale to [0,1], then normaise using ImageNet norm.'''
    n_timesteps = observation.shape[0]
    observation = observation / 1023
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for timestep in range(0, n_timesteps):
        norm_X_t = normalize(observation[timestep,:,:,:])
        observation[timestep,:,:,:] = norm_X_t
    return observation.to(device)
def scale_and_norm_batch(x, device):
    output_x = x.clone()
    for observation_index in range(0, x.shape[0]):
        output_x[observation_index] = scale_and_normalise_image(x[observation_index], device)
    return output_x
