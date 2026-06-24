import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
import VolcanoesDictionaryForQualityModels

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

def get_single_branched_resnet18(device):
    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(512, 64),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(64, 1),
                             nn.Sigmoid())
    return model.to(device)
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
        #This model contains redundant parameters for the
        #"m1" and "p1" branches (as these weights are not
        #trained, we pass each timestep data through the
        #equivalent "t0" branch. The redundant parameters
        #were mistakenly not removed before training of the
        #final classifiers. Though this is inefficient, and
        #will be corrected in any subsequent versions, it
        #should not affect results.

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

def mask_sensor_marks(image, mask_path, data_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(data_path + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image
def read_data(labels, timesteps, data_path, temporal_data_path, additional_data_path, do_mask_sensor_marks, sensor_mark_data_path, precip_is_labelled=False, cloud_is_labelled=False):
    X = []
    for index in range(0, labels.shape[0]):
        # Array to store data for this sample.
        # [observation_index][timestep_index, channel, 486, 648]
        if len(timesteps) > 1:
            this_obs_x = np.empty((len(timesteps), 3, 486, 648))
        else:
            this_obs_x = np.empty((3, 486, 648))

        dictionary_name = labels['volcano_dictionary_name'][index]
        volcano_dictionary = VolcanoesDictionaryForQualityModels.map_dictionary_name_to_dictionary(dictionary_name)
        sensor_mask_name_A = volcano_dictionary["sensor_marks_mask_A"]
        sensor_mask_name_B = volcano_dictionary["sensor_marks_mask_B"]

        timestep_index = 0
        for timestep_name in timesteps:
            if timestep_name == "image_name":
                path_to_read = data_path
            else:
                path_to_read = temporal_data_path
            #Check if it's actually an additional sample:
            if "labelled" in labels.columns:
                if labels["labelled"][index] == "Additional":
                    path_to_read = additional_data_path

            image_name_A = labels[timestep_name][index]
            image_name_B = labels[timestep_name + "_B"][index]
            image_A = cv2.imread(path_to_read + "/" + image_name_A, -1)
            image_B = cv2.imread(path_to_read + "/" + image_name_B, -1)

            # Mask for sensor marks
            if do_mask_sensor_marks == True:
                image_A = mask_sensor_marks(image_A, sensor_mask_name_A, sensor_mark_data_path)
                image_B = mask_sensor_marks(image_B, sensor_mask_name_B, sensor_mark_data_path)

            if len(timesteps) > 1:
                this_obs_x[timestep_index, 0, :, :] = image_A
                this_obs_x[timestep_index, 1, :, :] = image_B
                this_obs_x[timestep_index, 2, :, :] = np.zeros_like(image_A)
            else:
                this_obs_x[0, :, :] = image_A
                this_obs_x[1, :, :] = image_B
                this_obs_x[2, :, :] = np.zeros_like(image_A)
            timestep_index += 1
        X.append(this_obs_x)
    return X
class ImageLoader(Dataset):
    def __init__(self, labels, timesteps, data_path, temporal_data_path, additional_data_path, device, do_mask_sensor_marks=True, sensor_mark_masks_path=None):

        X = read_data(labels, timesteps=timesteps, data_path=data_path, temporal_data_path=temporal_data_path, additional_data_path=additional_data_path, do_mask_sensor_marks=do_mask_sensor_marks, sensor_mark_data_path=sensor_mark_masks_path)

        self.n_timesteps = len(timesteps)
        self.X = torch.tensor(np.array(X)).float()

        self.device = device
    def __len__(self):
        # Return number of input observations
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].to(self.device)

def scale_and_normalise_image(observation, n_timesteps, device):
    '''Scale to [0,1], then normaise using ImageNet norm.'''
    observation = observation / 1023
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if n_timesteps > 1:
        for timestep in range(0, n_timesteps):
            norm_X_t = normalize(observation[timestep,:,:,:])
            observation[timestep,:,:,:] = norm_X_t
    else:
        norm_X_t = normalize(observation)
        observation = norm_X_t
    return observation.to(device)
def scale_and_norm_batch(x, n_timesteps, device):
    output_x = x.clone()
    for observation_index in range(0, x.shape[0]):
        output_x[observation_index] = scale_and_normalise_image(x[observation_index], n_timesteps, device)
    return output_x
