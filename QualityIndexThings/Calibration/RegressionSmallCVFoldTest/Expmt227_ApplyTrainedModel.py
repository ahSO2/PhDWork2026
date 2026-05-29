import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.utils import resample
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import v2

full_exp_start = time.time()
########Things to edit
import sys
#sys.path.append("/mnt/parscratch/users/ggp24ash")
sys.path.append("C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/HPCScripts")
import VolcDictionaryWithCorrectClears
#labels_path = "/mnt/parscratch/users/ggp24ash/CrossValidationSplits"
labels_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/UpdatedCorrectedDataframes/ToUploadToHPC"
dataset_name = "FGCloudSet"
#data_path = "/mnt/parscratch/users/ggp24ash/FullDatasetCorrectedWithVolcDict2"
data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2"
#temporal_data_path = "/mnt/parscratch/users/ggp24ash/FullDatasetCorrectedWithVolcDict2Temporal"
temporal_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/FullDatasetCorrectedWithVolcDict2Temporal"

#sensor_mark_masks_path = "/mnt/parscratch/users/ggp24ash/SensorMarkMasks/"
sensor_mark_masks_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/HPCScripts/SensorMarkMasks/"
torch.manual_seed(42)
np.random.seed(42)
do_mask_sensor_marks = True
do_norm = True
data_aug_proportion = 0.5
corrected_predictions_to_add_count = 300
correct_predictions_to_add_count = 0
correct_predictions_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/RecursiveLabellingKilauea/Round3/CorrectPredictionsR2And3FGCloud.xlsx"
#correct_predictions_path = "/mnt/parscratch/users/ggp24ash/RecursiveLabellingKilauea/Round3/CorrectPredictionsR2And3FGCloud.xlsx"
corrected_predictions_path = "C:/Users/ggp24ash/PycharmProjects/MLforQualityClass/ChunkLabelsSet/RecursiveLabellingKilauea/Round3/CorrectedLabelsCombR2And3FGCloud.xlsx"
#corrected_predictions_path = "/mnt/parscratch/users/ggp24ash/RecursiveLabellingKilauea/Round3/CorrectedLabelsCombR2And3FGCloud.xlsx"
additional_data_path = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/PotentialNewDataForRecursiveLabellingKilauea"
#additional_data_path = "/mnt/parscratch/users/ggp24ash/PotentialNewDataForRecursiveLabellingKilauea"
additional_data_path_temporal = "C:/Users/ggp24ash/Documents/Main Datasets/QualityClassification/PotentialNewDataForRecursiveLabellingKilaueaTemporal"
#additional_data_path_temporal = "/mnt/parscratch/users/ggp24ash/PotentialNewDataForRecursiveLabellingKilaueaTemporal"

dropout = 0.2
mod = 1
n_epochs = 20
lr = 0.0008
train_batch_size = 50
valid_batch_size = 10
outputs_save_loc = "C:/Users/ggp24ash/Documents/HPC Outputs/Experiment227/"
timesteps_to_read = ["minus_ten_s_name", "image_name", "plus_ten_s_name"]
########

location_names = ["Cotopaxi"]
#location_names = ["Kilauea"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:" + device)

########## Defining the necessary functions
def mask_sensor_marks(image, mask_path):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(sensor_mark_masks_path + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image

def convert_timestep_naming_convention(df):
    df.rename(columns={'prev_min_name': 'minus_one_min_name',
                       'prev_min_name_B': 'minus_one_min_name_B',
                       'next_min_name': 'plus_one_min_name',
                       'next_min_name_B': 'plus_one_min_name_B',
                       'prev_tensec_name': "minus_ten_s_name",
                       'prev_tensec_name_B': "minus_ten_s_name_B",
                       'next_tensec_name': "plus_ten_s_name",
                       'next_tensec_name_B': "plus_ten_s_name_B"}, inplace=True)
    return df

def read_data(labels, timesteps_to_read, do_mask_sensor_marks, mod=1):
    x = []
    y = []
    n_timesteps = len(timesteps_to_read)
    t0_index = int((n_timesteps - 1) / 2)
    for index in range(0, len(labels['image_name']), mod):
        #print(labels['data_type'][index])
        if labels['data_type'][index] == "original":
            path_to_read = data_path
            path_to_read_temporal = temporal_data_path
        else:
            path_to_read = additional_data_path
            path_to_read_temporal = additional_data_path_temporal
        #print(path_to_read)
        #print(path_to_read_temporal)

        # Array to store data for this sample.
        # [observation_index][timestep_index, channel, 486, 648]
        this_obs_x = np.empty((n_timesteps, 3, 486, 648))
        dictionary_name = labels['volcano_dictionary_name'][index]
        volcano_dictionary = VolcDictionaryWithCorrectClears.map_dictionary_name_to_dictionary(dictionary_name)
        sensor_mask_path_A = volcano_dictionary["sensor_marks_mask_A"]
        sensor_mask_path_B = volcano_dictionary["sensor_marks_mask_B"]

        # read the image
        timestep_index = 0
        for timestep_col_name in timesteps_to_read:
            #print(timestep_col_name)
            image_name_A = labels[timestep_col_name][index]
            image_name_B = labels[timestep_col_name + "_B"][index]
            #print(image_name_A)
            # If the timestep is t=0, then read from the std folder path
            # Otherwise read from the temporal data folder path
            if timestep_index == t0_index:
                image_A = cv2.imread(path_to_read+ "/" + image_name_A, -1)
                image_B = cv2.imread(path_to_read + "/" + image_name_B, -1)
            else:
                #print("Reading temp data")
                image_A = cv2.imread(path_to_read_temporal + "/" + image_name_A, -1)
                image_B = cv2.imread(path_to_read_temporal + "/" + image_name_B, -1)


            # Mask for sensor marks
            if do_mask_sensor_marks == True:

                #print(sensor_mask_path_A)
                #unmasked_image_A = this_image.copy()
                image_A = mask_sensor_marks(image_A, sensor_mask_path_A)
                #unmasked_image_B = this_image_B.copy()
                image_B = mask_sensor_marks(image_B, sensor_mask_path_B)

                #fig, axs = plt.subplots(nrows=1, ncols=2)
                #axs[0].imshow(unmasked_image_A, cmap='gray')
                #axs[0].set_title("Original")
                #axs[1].imshow(this_image, cmap='gray')
                #axs[1].set_title("Infilled")
                #plt.show()

            # [observation_index][timestep_index, channel, 486, 648]
            this_obs_x[timestep_index, 0, :, :] = image_A
            this_obs_x[timestep_index, 1, :, :] = image_B

            #fig, axs = plt.subplots(nrows=1, ncols=n_timesteps)
            #for i in range(0, n_timesteps):
            #    axs[i].imshow(this_obs_x[i, 1, :, :], cmap='gray')
            #    axs[i].set_title("Timestep " + str(i))
            #plt.show()

            # store the observations in x and y
            this_obs_x[timestep_index, 2, :, :] = np.zeros_like(image_A)
            timestep_index += 1
        this_obs_y = labels['cloud_level_numeric'][index]

        x.append(this_obs_x)
        y.append(this_obs_y)
    return x, y

def date_from_image_name(name):
    return name.split("_")[1][:10]

def select_additional_data_to_add(potential_to_add, set_to_exclude):
    #Take in a df of potential new labels
    #Filter out any that are from days in the validation set
    dates_to_omit = list(set(set_to_exclude["date"]))
    print("Dates to omit: ")
    print(dates_to_omit)
    potential_to_add["date"] = potential_to_add["image_name"].apply(date_from_image_name)
    data_to_add = potential_to_add[~potential_to_add["date"].isin(dates_to_omit)]
    #data_to_add.to_excel("TestOfDataToAdd.xlsx")
    #Return the updated df of labels to add
    return data_to_add

def map_yes_no_to_binary(value):
    if value == "Yes":
        return 1
    if value == "No":
        return 0

def map_binary_to_yes_no(value):
    if value == 1:
        return "Yes"
    elif value == 0:
        return "No"

def convert_outcome_to_binary(labels_df, column_name):
    new_column = labels_df[column_name].apply(map_yes_no_to_binary)
    labels_df[column_name + "_Yes"] = new_column
    return labels_df

def map_level_to_numeric(level):
    if level == "No":
        return 0
    elif level == "Minor":
        return 0.25
    elif level == "Not Calc":
        return 0.75
    elif level == "In Calc":
        return 0.875
    elif level == "Very":
        return 1

def convert_level_to_numeric(labels_df, column_name):
    new_column = labels_df[column_name].apply(map_level_to_numeric)
    labels_df[column_name + "_numeric"] = new_column
    return labels_df


class SampleSplitLoader(Dataset):
    def __init__(self, labels, timesteps_to_read, mod, do_norm, do_mask_sensor_marks):
        # Read in the data.
        labels = convert_outcome_to_binary(labels, 'cloud')
        labels = convert_level_to_numeric(labels, "cloud_level")
        x, y = read_data(labels, timesteps_to_read, do_mask_sensor_marks=do_mask_sensor_marks, mod=mod)
        self.n_timesteps = len(timesteps_to_read)
        # x should be a list of the images, converted to tensor
        self.X = torch.tensor(np.array(x)).float()
        self.Y = torch.tensor(np.array(y)).float()

    def __len__(self):
        # Return number of input observations
        return len(self.X)

    def __getitem__(self, index):
        # Fetch one observation.
        # Scale it to [0,1] and optionally normalise it
        #X_to_norm = torch.tensor(np.zeros((3, 486, 648))).float()
        #X_to_norm[0, :, :] = self.X[index][0, :, :] / 1024
        #X_to_norm[1, :, :] = self.X[index][1, :, :] / 1024
        #X_to_norm[2, :, :] = self.X[index][2, :, :] / 1024
        #### Normalisation has been moved to occur in the batch
        #### loader, to allow for normalisation between aug transforms

        return self.X[index].to(device), self.Y[index].to(device)



def get_pretrained_resnet18_model_only():
    model = models.resnet18(weights = "IMAGENET1K_V1")
    #Fix the weights so thay are not trained
    for param in model.parameters():
        param.requires_grad = False
    return model

class TripleBranchedModel(torch.nn.Module):
    def __init__(self):
        super(TripleBranchedModel, self).__init__()
        #I will call the previous timesteps m1, m2 ... for 'minus 1 step'
        #and the next timesteps p1, p2.. for 'plus 1 step'

        #Maybe call the pretrained ResNets here
        pretrained_resnet18 = get_pretrained_resnet18_model_only()

        #Branch t=0 #TODO check this is actually creating a new copy
        self.t0_conv1 = pretrained_resnet18.conv1
        self.t0_bn1 = pretrained_resnet18.bn1
        self.t0_relu = pretrained_resnet18.relu
        self.t0_maxpool = pretrained_resnet18.maxpool
        self.t0_layer1 = pretrained_resnet18.layer1
        self.t0_layer2 = pretrained_resnet18.layer2 #Output should be 128 28x28 oputput images
        self.t0_layer3 = pretrained_resnet18.layer3
        self.t0_layer4 = pretrained_resnet18.layer4

        self.m1_conv1 = pretrained_resnet18.conv1
        self.m1_bn1 = pretrained_resnet18.bn1
        self.m1_relu = pretrained_resnet18.relu
        self.m1_maxpool = pretrained_resnet18.maxpool
        self.m1_layer1 = pretrained_resnet18.layer1
        self.m1_layer2 = pretrained_resnet18.layer2  # Output should be 128 28x28 oputput images
        self.m1_layer3 = pretrained_resnet18.layer3
        self.m1_layer4 = pretrained_resnet18.layer4

        self.p1_conv1 = pretrained_resnet18.conv1
        self.p1_bn1 = pretrained_resnet18.bn1
        self.p1_relu = pretrained_resnet18.relu
        self.p1_maxpool = pretrained_resnet18.maxpool
        self.p1_layer1 = pretrained_resnet18.layer1
        self.p1_layer2 = pretrained_resnet18.layer2  # Output should be 128 28x28 oputput images
        self.p1_layer3 = pretrained_resnet18.layer3
        self.p1_layer4 = pretrained_resnet18.layer4

        #Then concatenate
        self.t0_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Think this pools each of the 512 3x3 convolutional filtered images into 1x1s
        self.m1_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.p1_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(nn.Flatten(),
                nn.Linear(512 * 3, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                #nn.Linear(256, 128),
                nn.Linear(64, 1),
                nn.Sigmoid())

    def forward(self, x):
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
        return comb_output #The output

def get_triple_branched_resnet18():
    #initial_lr = 0.01
    model = TripleBranchedModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    #also return scheduler
    return model.to(device), loss_fn, optimizer

#Defining Transforms
resize_cropper = v2.RandomResizedCrop(size=(486, 648), scale=(0.8,1), ratio=(0.7,0.8))
#Want images to be the same size, but simulate slight changes in zoom and view with the
#scale parameter, then changes in the actual shape of the volcano and the obscurnace with
#the ratio parameter.

hflipper = v2.RandomHorizontalFlip(p=0.5)

perspective_transformer = v2.RandomPerspective(distortion_scale=0.2, p=0.5)
#There's one result that looks black - its just because of the contrast
#Check that band A and B still are registered well after this trasnform - done, all the ones I view look ok

def find_percentile_of_tensor(the_tensor):
    as_array = the_tensor.cpu().numpy()
    masked = np.ma.masked_where(as_array == 0, as_array)
    compressed = np.ma.compressed(masked).flatten()
    perc_val = np.percentile(compressed, 99)
    return perc_val

def relu_tensor(the_tensor):
    as_numpy = the_tensor.cpu().numpy()
    result = np.where(as_numpy < 0, 0, as_numpy)
    return torch.tensor(result).float()

class LightingTransform(torch.nn.Module):
    def forward(self, img):
        #Transform the lighting of the image
        #Add or minus from all the gray levels some amount in
        # [0, minumum gray level]
        random_proportion = random.uniform(-0.5, 0.5) #proportion of min gray level
        #print("Random proportion" + str(random_proportion))
        new_img = torch.empty_like(img)
        for i in range(0, img.shape[0]): #For each timestep
            min_band_A = find_percentile_of_tensor(img[i,0,:,:])
            #print(min_band_A)
            min_band_B = find_percentile_of_tensor(img[i,1,:,:])
            #print(min_band_B)
            #Using percentiles rather than mins, then fill any negative vals with zero
            shifted_A = img[i, 0,:,:] + random_proportion * min_band_A
            shifted_B = img[i, 1,:,:] + random_proportion * min_band_B
            #Any negative values are filled with zero:
            shifted_A = relu_tensor(shifted_A)
            shifted_B = relu_tensor(shifted_B)

            #Scale the maximum gray level by factor in [0.5, 1.5]

            scale_factor = random.uniform(0.7, 1.3)
            #print("Random scale factor: " + str(scale_factor))
            new_band_A = shifted_A * scale_factor
            new_band_B = shifted_B * scale_factor

            #If either band's values are now outwith the range [0,1023]
            #then scale both bands back down accordingly.

            band_A_ratio = new_band_A.max() /1023
            band_B_ratio = new_band_B.max() /1023

            if band_A_ratio > 1 or band_B_ratio > 1:
                ratio_to_use = max(band_A_ratio, band_B_ratio)
                new_band_A = new_band_A/ratio_to_use
                new_band_B = new_band_B/ratio_to_use

            new_img[i, 0,:,:] = new_band_A
            new_img[i, 1,:,:] = new_band_B
        return new_img.to(device)

lighting_transform = LightingTransform()

class PartialCutoff(torch.nn.Module):
    def forward(self, img):
        #Partially occlude the
        occ_img = img.clone()

        do_transform = random.randint(0,1)

        #Pick a random direction
        corners = ["tl", "bl", "tr", "br"]
        corner = random.choice(corners)
        #print(corner)
        #Pick a random number of pixels
        pixels = random.randint(150,400)
        #print(pixels)
        #Draw a rectangle

        if do_transform == 0:
            pass
        elif corner == "tl":
            occ_img[:,0,0:pixels,0:pixels] = torch.tensor((np.zeros((img.shape[0], pixels,pixels))))
            occ_img[:,1,0:pixels, 0:pixels] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))
        elif corner == "bl":
            occ_img[:, 0, -pixels:, 0:pixels] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))
            occ_img[:, 1, -pixels:, 0:pixels] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))
        elif corner == "tr":
            occ_img[:,0, 0:pixels, -pixels:] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))
            occ_img[:,1, 0:pixels, -pixels:] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))
        elif corner == "br":
            occ_img[:,0, -pixels:, -pixels:] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))
            occ_img[:,1, -pixels:, -pixels:] = torch.tensor((np.zeros((img.shape[0], pixels, pixels))))

        return occ_img.to(device)

partial_occl = PartialCutoff()

def re_zero_band_B_edges(transformed_band_B, original_band_B):
    original_numpy = original_band_B.cpu().numpy()
    transformed_numpy = transformed_band_B.cpu().numpy()

    re_zeroed = np.where(original_numpy == 0, 0, transformed_numpy)
    return torch.tensor(re_zeroed).float().to(device)


selected_transforms = v2.Compose([perspective_transformer,
                         hflipper,
                         resize_cropper,
                         partial_occl,
])


def plot_two_images(left, l_title, right, r_title, save_path = None):
    fig, axs = plt.subplots(ncols=2)
    left_plot = axs[0].imshow(left, cmap='gray')
    right_masked = np.ma.masked_where(right == 0, right)
    right_plot = axs[1].imshow(right_masked, cmap='gray')
    axs[0].set_title(l_title)
    axs[1].set_title(r_title)
    fig.colorbar(left_plot, ax=axs[0], shrink=0.5)
    fig.colorbar(right_plot, ax=axs[1], shrink=0.5)
    if save_path != None:
        fig.savefig(save_path)
        plt.close()
    plt.show()
def augment_batch(x, y, index, n_to_aug):
    batch_size = x.shape[0]
    #randomly select n samples to augment
    indexes_to_aug = np.random.randint(0, batch_size, size=n_to_aug)
    #select the corresponding y vals
    aug_x = x[indexes_to_aug,:,:,:,:].clone()
    x_to_aug = aug_x.clone()
    aug_y = y[indexes_to_aug].clone()
    #For each observation
    for observation_index in range(0, aug_x.shape[0]):

        transformed_img = lighting_transform(aug_x[observation_index])
        #Where the original band B image was zero, infill with zeroes
        #transformed_img[1,:,:] = re_zero_band_B_edges(transformed_img[1,:,:], x[observation_index][1,:,:])

        transformed_img = selected_transforms(transformed_img)
        transformed_img = scale_and_normalise_image(transformed_img)

        #band_A = transformed_img[:, 0, :, :]
        #band_B = transformed_img[1,:,:]

        #random_number = random.randint(1,30)
        #if random_number == 1:
        #    plot_two_images(x_to_aug[observation_index,0,:,:].cpu().numpy(), "Original", band_A.cpu().numpy(), "Transformed", save_path = None)
        #    plot_two_images(x_to_aug[observation_index, 1, :, :].cpu().numpy(), "Original", band_B.cpu().numpy(),"Transformed", save_path = None)

        aug_x[observation_index] = transformed_img
    return aug_x, aug_y


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def scale_and_normalise_image(observation):
    '''Scale to [0,1], then normaise using ImageNet norm.'''
    n_timesteps = observation.shape[0]
    observation = observation / 1023

    if do_norm == True:
        for timestep in range(0, n_timesteps):
            norm_X_t = normalize(observation[timestep,:,:,:])
            observation[timestep,:,:,:] = norm_X_t
            #plt.imshow(norm_X[1,:,:])
            #plt.colorbar()
            #plt.show()
        return observation.to(device)
def scale_and_norm_batch(x):
    output_x = x.clone()
    for observation_index in range(0, x.shape[0]):
        output_x[observation_index] = scale_and_normalise_image(x[observation_index])
    return output_x


def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y.view(y.shape[0], 1))
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == (torch.transpose(y.view(1, len(y)), 0, 1) >= 0.5)
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def eval_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    loss = loss_fn(prediction, y.view(y.shape[0], 1))
    return loss.item()

######### Applying the functions to train a model at each location
for location in location_names:
    torch.cuda.empty_cache()
    print("Reading data from: " + labels_path)
    print("For location: " + location)
    train = pd.read_excel(labels_path + "/" + dataset_name + "CVSplitWithout" + location + "Train.xlsx")
    valid = pd.read_excel(labels_path + "/" + dataset_name + "CVSplitWithout" + location + "Test.xlsx")
    test = pd.read_excel(labels_path + "/" + dataset_name[:-3] + "CrossValidTestSet" + location + ".xlsx")

    train["data_type"] = ["original"] * train.shape[0]
    valid["data_type"] = ["original"] * valid.shape[0]
    test["data_type"] = ["original"] * test.shape[0]

    print("Total samples in this fold:")
    print(int(train.shape[0]) + int(valid.shape[0]) + int(test.shape[0]))


    if corrected_predictions_to_add_count != 0:
        potential_corrected_additional_labels = pd.read_excel(corrected_predictions_path)
        potential_corrected_additional_labels = convert_timestep_naming_convention(potential_corrected_additional_labels)
        print("Potential corrected additional labels:" + str(potential_corrected_additional_labels.shape[0]))
        #select ones that are not from days in valid set
        corrected_additional_labels = select_additional_data_to_add(potential_corrected_additional_labels, set_to_exclude=valid)
        corrected_additional_labels["data_type"] = ["additional"] * corrected_additional_labels.shape[0]
        print("Number selected that were not from day in valid set: " + str(corrected_additional_labels.shape[0]))
        if corrected_predictions_to_add_count == "all":
            print("Adding all potential additional data.")
        else:
            corrected_additional_labels = resample(corrected_additional_labels, replace=True, n_samples=corrected_predictions_to_add_count)
            print("Adding " + str(corrected_additional_labels.shape[0]) + " samples of corrected data.")
        #and add to train set
        train = pd.concat([train, corrected_additional_labels], axis=0)
        train = train.reset_index(drop=True)
        #train.to_excel("ExampleTrainWithAddData.xlsx")

    if correct_predictions_to_add_count != 0:
        potential_correct_additional_labels = pd.read_excel(correct_predictions_path)
        potential_correct_additional_labels["cloud"] = potential_correct_additional_labels["cloud"].copy().apply(map_binary_to_yes_no).to_list()
        potential_correct_additional_labels = convert_timestep_naming_convention(potential_correct_additional_labels)
        print("Potential correct additional labels:" + str(potential_correct_additional_labels.shape[0]))
        #select ones that are not from days in valid set
        correct_additional_labels = select_additional_data_to_add(potential_correct_additional_labels, set_to_exclude=valid)
        correct_additional_labels["data_type"] = ["additional"] * correct_additional_labels.shape[0]
        print("Number selected that were not from day in valid set: " + str(correct_additional_labels.shape[0]))
        if correct_predictions_to_add_count == "all":
            print("Adding all potential additional data.")
        else:
            correct_additional_labels = resample(correct_additional_labels, replace=True, n_samples=correct_predictions_to_add_count)
            print("Adding " + str(correct_additional_labels.shape[0]) + " samples of correct prediction data.")
        #and add to train set
        train = pd.concat([train, correct_additional_labels], axis=0)
        train = train.reset_index(drop=True)
        train.to_excel("ExampleTrainWithAddData.xlsx")

    #Setup structures to store outputs
    metrics_df = pd.DataFrame()

    #train_set = SampleSplitLoader(labels=train, timesteps_to_read=timesteps_to_read, mod=mod, do_norm=do_norm,
    #                              do_mask_sensor_marks=do_mask_sensor_marks)
    print("Loading valid set")
    valid_set = SampleSplitLoader(labels=valid, timesteps_to_read=timesteps_to_read, mod=mod, do_norm=do_norm,
                                  do_mask_sensor_marks=do_mask_sensor_marks)
    print("Loading test set")
    test_set = SampleSplitLoader(labels=test, timesteps_to_read=timesteps_to_read, mod=mod, do_norm=do_norm,
                                 do_mask_sensor_marks=do_mask_sensor_marks)

    #print("Number of training observations:" + str(len(train_set)))
    print("Number of validation observations (seen locs):" + str(len(valid_set)))
    print("Number of test observations (unseen loc):" + str(len(test_set)))

    #train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_set, batch_size=valid_batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=valid_batch_size, shuffle=True, drop_last=False)

    location_start = time.time() #Timer for training at this location
    model, loss_fn, optimizer = get_triple_branched_resnet18()
    model.load_state_dict(torch.load("C:/Users/ggp24ash/Documents/HPC Outputs/Experiment227/Cotopaxi_epoch150.pth", weights_only=True))
    model.eval()

    valid_targets = []
    valid_predictions = []
    test_targets = []
    test_predictions = []

    for index, batch in enumerate(iter(valid_dataloader)):
        x, y = batch
        x = scale_and_norm_batch(x)
        prediction = model(x)
        valid_targets = valid_targets + y.detach().cpu().numpy().tolist()
        valid_predictions = valid_predictions + prediction[:,0].detach().cpu().numpy().tolist()

    valid_results_df = pd.DataFrame()
    valid_results_df["target"] = valid_targets
    valid_results_df["prediction"] = valid_predictions
    valid_results_df.to_excel(outputs_save_loc + "ValidSetPredictions.xlsx")

    for index, batch in enumerate(iter(test_dataloader)):
        x, y = batch
        x = scale_and_norm_batch(x)
        prediction = model(x)
        test_targets = test_targets + y.detach().cpu().numpy().tolist()
        test_predictions = test_predictions + prediction[:,0].detach().cpu().numpy().tolist()

    test_results_df = pd.DataFrame()
    test_results_df["target"] = test_targets
    test_results_df["prediction"] = test_predictions
    test_results_df.to_excel(outputs_save_loc + "TestSetPredictions.xlsx")




    '''
    # Train and evaluate the ML model
    train_losses = []
    valid_losses = []
    test_losses = []
    train_accuracies = []
    valid_accuracies = []
    test_accuracies = []

    eps = n_epochs
    n_to_aug = int(round(train_batch_size * data_aug_proportion, 0))
    print("Augmenting " + str(n_to_aug) + " images per batch.")
    for epoch in range(eps):
        print(f"epoch {epoch + 1}/" + str(eps))
        epoch_train_losses = []
        epoch_valid_losses = []
        epoch_test_losses = []
        epoch_train_accuracies = []
        epoch_valid_accuracies = []
        epoch_test_accuracies = []

        for index, batch in enumerate(iter(train_dataloader)):
            # print(f"Processing batch {index}")
            x, y = batch

            if n_to_aug == 0:
                x_train = scale_and_norm_batch(x)
                y_train = y.clone()
            else:
                aug_x, aug_y = augment_batch(x, y, index, n_to_aug)
                x_norm = scale_and_norm_batch(x)
                x_train = torch.cat((x_norm, aug_x))
                y_train = torch.cat((y, aug_y))

            batch_loss = train_batch(x_train, y_train, model, optimizer, loss_fn)
            epoch_train_losses.append(batch_loss)
            is_correct = accuracy(x_train, y_train, model)
            epoch_train_accuracies.extend(is_correct)
        epoch_loss = np.array(epoch_train_losses).mean()
        print("Epoch training loss:" + str(epoch_loss))
        epoch_train_accuracy = np.mean(epoch_train_accuracies)
        print("Epoch training accuracy:" + str(epoch_train_accuracy))

        for index, batch in enumerate(iter(valid_dataloader)):
            x, y = batch
            x = scale_and_norm_batch(x)
            val_is_correct = accuracy(x, y, model)
            epoch_valid_accuracies.extend(val_is_correct)
            batch_valid_loss = eval_loss(x, y, model, loss_fn)
            epoch_valid_losses.append(batch_valid_loss)
        epoch_valid_loss = np.array(epoch_valid_losses).mean()
        print("Epoch validation loss:" + str(epoch_valid_loss))
        epoch_valid_accuracy = np.mean(epoch_valid_accuracies)
        print("Epoch validation accuracy:" + str(epoch_valid_accuracy))

        for index, batch in enumerate(iter(test_dataloader)):
            x, y = batch
            x = scale_and_norm_batch(x)
            test_is_correct = accuracy(x, y, model)
            epoch_test_accuracies.extend(test_is_correct)
            batch_test_loss = eval_loss(x, y, model, loss_fn)
            epoch_test_losses.append(batch_test_loss)
        epoch_test_loss = np.array(epoch_test_losses).mean()
        print("Epoch test loss:" + str(epoch_test_loss))
        epoch_test_accuracy = np.mean(epoch_test_accuracies)
        print("Epoch test accuracy:" + str(epoch_test_accuracy))


        train_losses.append(epoch_loss)
        valid_losses.append(epoch_valid_loss)
        test_losses.append(epoch_test_loss)
        train_accuracies.append(epoch_train_accuracy)
        valid_accuracies.append(epoch_valid_accuracy)
        test_accuracies.append(epoch_test_accuracy)

        if ((epoch + 1) % 10) == 0:
            this_model_save_path = outputs_save_loc + location + "_epoch" + str(epoch + 1) + ".pth"
            torch.save(model.state_dict(), this_model_save_path)

    location_end = time.time()
    location_runtime = location_end - location_start
    print("For " + location + "training runtime is: " + str(location_runtime))

    epoch_indexes = np.arange(1, len(train_accuracies) + 1, 1)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    train_accuracy_plot = axs[0].plot(epoch_indexes, train_accuracies, label="Train")
    valid_accuracy_plot = axs[0].plot(epoch_indexes, valid_accuracies, label="Valid")
    test_accuracy_plot = axs[0].plot(epoch_indexes, test_accuracies, label="Test")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Prediction Accuracy")
    axs[0].legend()
    axs[0].set_title("Accuracy")
    train_loss_plot = axs[1].plot(epoch_indexes, train_losses, label = "Train")
    valid_loss_plot = axs[1].plot(epoch_indexes, valid_losses, label = "Valid")
    test_loss_plot = axs[1].plot(epoch_indexes, test_losses, label="Test")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("BCE Loss")
    axs[1].legend()
    axs[1].set_title("Loss")
    plt.suptitle("Training Metrics: " + location)
    #plt.show()
    plt.savefig(outputs_save_loc + location + "_TrainValidAcc.png")
    plt.close()

    # Save the accuracies and losses to a csv
    this_fold_outputs_df = pd.DataFrame()
    this_fold_outputs_df["TrainAccuracies"] = train_accuracies
    this_fold_outputs_df["TrainLosses"] = train_losses
    this_fold_outputs_df["ValidAccuracies"] = valid_accuracies
    this_fold_outputs_df["ValidLosses"] = valid_losses
    this_fold_outputs_df["TestAccuracies"] = test_accuracies
    this_fold_outputs_df["TestLosses"] = test_losses
    this_fold_outputs_df.to_excel(outputs_save_loc + location + "_Metrics.xlsx")

full_exp_end = time.time()
full_runtime = full_exp_end - full_exp_start
print("Full runtime: " + str(full_runtime))
'''



