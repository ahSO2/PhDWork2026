from mpl_toolkits.basemap import Basemap
import bisect
from BackgroundMethods import *
import cv2
from datetime import datetime, timedelta
import geonum
from geonum import BASEMAP_AVAILABLE
import QualityModelFunctions
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
import pandas as pd
import random
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader

def show(image, cmap="gray", title=None):
    plt.imshow(image, cmap=cmap)
    plt.colorbar()
    if title != None:
        plt.title(title)
    plt.show()


def map_image_name_to_time(image_name):
    date_str = image_name.split("_")[0][0:10]
    time_str = image_name.split("_")[0][11:17]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    return datetime_obj

def map_spectrometer_timestamp_to_datetime(timestamp):
    date_str = timestamp[0:10]
    time_str = timestamp[-8:]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[3:5]), int(time_str[6:]))
    return datetime_obj

def shutter_speed_from_band_img_name(image_name):
    return int(image_name.split("_")[3][:-2])

def mask_sensor_marks(image, mask):
    '''Take in a float image, create an 8-bit copy, and calculate the masked values.
    Infill the float image using these calculated values.'''
    scale_factor = 255/np.max(image)
    image_8bit = (image * scale_factor).astype(np.uint8)
    infilled_8bit = cv2.inpaint(image_8bit, mask, 5, cv2.INPAINT_TELEA)
    infilled_rescaled = infilled_8bit / scale_factor
    image = np.where(mask > 0, infilled_rescaled, image)
    return image

def correct_sample(bandA, bandB, dark_name_A, dark_name_B, vin_mask_A, vin_mask_B, reg_trans, smm_A, smm_B):
    '''Apply corrections to the image pair with this index (in the whole sequence).'''

    #Dark subtract
    dark_img_A = cv2.imread(dark_name_A, -1)
    dark_img_B = cv2.imread(dark_name_B, -1)
    bandA = bandA.astype(np.float32) - dark_img_A
    bandB = bandB.astype(np.float32) - dark_img_B


    # Vignette correct
    bandA = np.divide(bandA, vin_mask_A)
    bandB = np.divide(bandB, vin_mask_B)

    #Register bandB
    bandB = cv2.warpPerspective(bandB, reg_trans, (bandB.shape[1], bandB.shape[0]))

    # Mask sensor marks where applicable
    if isinstance(smm_A, np.ndarray):
        bandA = mask_sensor_marks(bandA, smm_A)
    if isinstance(smm_B, np.ndarray):
        bandB = mask_sensor_marks(bandB, smm_B)

    return bandA, bandB

def calculate_AA(bandA_images, bandB_images, backgrounds_A, backgrounds_B):
    '''Given lists of 310 and 330nm image samples, and corresponding background estimations,
    calulate the apparent absorbance and return in the form of a 3D array. Mask out any areas of
    the image or background image equal to zero, (and return zero value for these pixels)
    to avoid error in log function.'''
    array_A = np.array(bandA_images)
    array_B = np.array(bandB_images)
    masked_array_A = np.ma.masked_where(array_A == 0, array_A)
    masked_array_B = np.ma.masked_where(array_B == 0, array_B)
    bg_array_A = np.array(backgrounds_A)
    bg_array_B = np.array(backgrounds_B)
    masked_backgrounds_A = np.ma.masked_where(bg_array_A == 0, bg_array_A)
    masked_backgrounds_B = np.ma.masked_where(bg_array_B == 0, bg_array_B)
    tau_A = -1 * np.ma.log(np.ma.divide(masked_array_A, masked_backgrounds_A))
    tau_B = -1 * np.ma.log(np.ma.divide(masked_array_B, masked_backgrounds_B))
    AA_seq = tau_A - tau_B

    return AA_seq

class Sequence():
    '''Class representing a sequence of data to be processed.'''
    def __init__(self):
        self.volc_dict = None
        self.chunk_indexes = []
        self.previous_indexes = []
        self.batch_bandA = []
        self.batch_bandB = []
        self.spectra_exists = []
        self.spectra = []
        self.bgs_A = []
        self.bgs_B = []



    def set_volcano_dictionary(self, dictionary):
        self.volc_dict = dictionary

    def initialise_and_match_full_sequence(self, directory_path, correct=True):
        '''Read all sample names from the given directory. Produce a list
        of band A samples, matched to corresponding bandB samples.
        If correct is True, match names of files for dark and vignette correction,
        then initialse the registration transform, using the data from the volcano dictionary.
        '''
        self.image_directory = directory_path
        if self.volc_dict == None and correct == True:
            print("ERROR: Volcano dictionary needed for data correction.")

        print("Matching image pairs in directory:")
        all_filenames = os.listdir(directory_path)
        all_band_A_names = list(filter(lambda file_name: ("fltrA" in file_name and ".png" in file_name), all_filenames))
        print("Total band A samples:" + str(len(all_band_A_names)))
        all_band_B_names = list(filter(lambda file_name: ("fltrB" in file_name and ".png" in file_name), all_filenames))

        all_band_B_times = []
        for band_B_name in all_band_B_names:
            date_time = map_image_name_to_time(band_B_name)
            all_band_B_times.append(date_time)

        self.bandA_names = []
        self.bandB_names = []
        self.times = []

        #For every band A image, check if there is a matching band B image
        #If so, save the sample pair to the sequence and note the time
        for band_A_name in all_band_A_names:
            date_time = map_image_name_to_time(band_A_name)
            if date_time in all_band_B_times:
                match_index = all_band_B_times.index(date_time)
                band_B_name = all_band_B_names[match_index]
                self.bandA_names.append(band_A_name)
                self.bandB_names.append(band_B_name)
                self.times.append(date_time)
                # TODO Note: if any more lists of values for the whole sequence are added at this stage, remember to remove them
                # using del below (during the reading of correction data stage) if there is no matching dark image.

        print("Matching band A/B pairs found: " + str(len(self.bandA_names)))

        self.correct = correct
        if correct == True:
            print("Loading correction data for this sequence:")
            #If we do want to correct the data as it is read in, load the information to do so here.
            ##Read and match dark images
            self.dark_path_A = self.volc_dict["dark_path_A"]
            self.dark_path_B = self.volc_dict["dark_path_B"]
            self.available_dark_names_A = os.listdir(self.dark_path_A)
            self.available_dark_ss_A = []
            self.available_dark_names_B = os.listdir(self.dark_path_B)
            self.available_dark_ss_B = []
            self.dark_sig_figs = self.volc_dict["sig_figs_for_dark_ss"]

            for dark_name in self.available_dark_names_A:
                #print(dark_name)
                shutter_speed = int(dark_name.split("_")[3][:-2])
                #print(shutter_speed)
                self.available_dark_ss_A.append(round(shutter_speed, self.dark_sig_figs))
            for dark_name in self.available_dark_names_B:
                #print(dark_name)
                shutter_speed = int(dark_name.split("_")[3][:-2])
                #print(shutter_speed)
                self.available_dark_ss_B.append(round(shutter_speed, self.dark_sig_figs))

            #For each image pair, select the matching dark images for band A and B
            #If a matching dark image doesn't exist, drop the image pair
            self.matched_dark_names_A = []
            self.matched_dark_names_B = []
            pairs_to_remove = []
            for pair_index in range(0, len(self.bandA_names)):

                band_A_ss = round(shutter_speed_from_band_img_name(self.bandA_names[pair_index]), self.dark_sig_figs)
                band_B_ss = round(shutter_speed_from_band_img_name(self.bandB_names[pair_index]), self.dark_sig_figs)
                matching_dark_A = False
                matching_dark_B = False
                if band_A_ss in self.available_dark_ss_A:
                    matching_dark_A = True
                if band_B_ss in self.available_dark_ss_B:
                    matching_dark_B = True
                if matching_dark_A == False or matching_dark_B == False:
                    print("Removing image: " + str(self.bandA_names[pair_index]))
                    print("as there is no matching dark image for one or both of bands A or B.")
                    pairs_to_remove.append(pair_index)
                else:
                    matching_dark_index_A = self.available_dark_ss_A.index(band_A_ss)
                    matching_dark_index_B = self.available_dark_ss_B.index(band_B_ss)
                    self.matched_dark_names_A.append(self.available_dark_names_A[matching_dark_index_A])
                    self.matched_dark_names_B.append(self.available_dark_names_B[matching_dark_index_B])

            #Remove the pairs which didn't have matching dark images for one or both bands:
            for index_to_delete in pairs_to_remove:
                del self.bandA_names[index_to_delete]
                del self.bandB_names[index_to_delete]
                del self.times[index_to_delete]
            print(str(len(pairs_to_remove)) + " pairs dropped due to no match for dark images.")
            print(len(self.bandA_names))

            #Vignette correction data
            #Assuming that clears are already dark subtracted
            self.clear_path_A = self.volc_dict["clear_sky_path_A"]
            self.clear_path_B = self.volc_dict["clear_sky_path_B"]
            self.create_vinette_masks()

            #Registration transform:
            self.reg_trans_matrix = cv2.getPerspectiveTransform(self.volc_dict['registration_points_B'], self.volc_dict['registration_points_A'])

            try:
                self.flank_mask = cv2.imread(self.volc_dict["flank_mask_path"], -1)
            except:
                print("ERROR in reading flank mask.")

            #Masks for permanent marks on sensor to be infilled
            self.sensor_mark_mask_path_A = self.volc_dict["sensor_marks_mask_A"]
            self.sensor_mark_mask_path_B = self.volc_dict["sensor_marks_mask_A"]
            if self.sensor_mark_mask_path_A != "None":
                self.smm_A = cv2.imread(self.sensor_mark_mask_path_A, -1)
            else:
                self.smm_A = None
            if self.sensor_mark_mask_path_B != "None":
                self.smm_B = cv2.imread(self.sensor_mark_mask_path_B, -1)
            else:
                self.smm_B = None


        else:
            print("NOTE: No correction data is being loaded for this image sequence.")

    def match_timestep_images(self):
        '''For each pair of images that has been matched by the read-in function,
        identify the indexes of the pairs which represent the requested timesteps
        from that sample, and store as a list.'''
        self.timestep_indicies = []
        self.all_timestep_data_available = []

        #For every image pair, identify which pair index corresponds to the +-10s and +-1min timesteps
        for pair_time in self.times:
            this_pair_timestep_indices = []
            # Calculate the difference between this time and all available times
            for ts in self.requested_timesteps:
                target_time = pair_time + timedelta(seconds=ts)
                diffs_from_target = [np.abs(t - target_time) for t in self.times]
                minimum_diff_index = diffs_from_target.index(min(diffs_from_target))
                min_diff = diffs_from_target[minimum_diff_index].seconds
                diff_to_ts = pair_time - self.times[minimum_diff_index]

                if min_diff <= 10 and diff_to_ts.seconds != 0: #If the timestep is within 10s of the one requested, and is not zero (relevant for the 10s step)
                    this_pair_timestep_indices.append(minimum_diff_index)
                else:
                    this_pair_timestep_indices.append(np.nan)
            self.timestep_indicies.append(this_pair_timestep_indices)
            if np.nan in this_pair_timestep_indices:
                self.all_timestep_data_available.append(False)
            else:
                self.all_timestep_data_available.append(True)
    def create_vinette_masks(self):
        # Calculate the vignette masks
        print("Creating vignette masks.")
        clear_img_A = cv2.imread(self.clear_path_A, -1).astype(np.float32)
        self.vin_mask_A = np.divide(clear_img_A, np.max(clear_img_A))
        clear_img_B = cv2.imread(self.clear_path_B, -1).astype(np.float32)
        self.vin_mask_B = np.divide(clear_img_B, np.max(clear_img_B))
    def read_and_correct_selected_indexes(self, indexes_to_read, correct=True, temporal=True):
        '''For each index in self.chunk_indicies, read and optionally correct the
        corresponding image pair (the existing image list is overwritten).

        It is assumed that the clear sky images linked in the volcano dictionary are dark subtracted.'''

        bandA_outputs = []
        bandB_outputs = []

        if temporal == True:
            # Create an array to store the temporal image samples
            #[chunk size, timestep, band]
            temporal_array = np.empty(shape=(len(indexes_to_read), len(self.requested_timesteps), 2, self.img_shape[0], self.img_shape[1]))
        else:
            self.requested_timesteps = []

        index_in_chunk = -1
        for index_to_read in indexes_to_read:
            index_in_chunk += 1
            for timestep_to_read in [0] + self.requested_timesteps:
                if timestep_to_read == 0:
                    name_A = self.bandA_names[index_to_read]
                    name_B = self.bandB_names[index_to_read]
                else:
                    timestep_index = self.requested_timesteps.index(timestep_to_read)
                    matched_timestep_index = self.timestep_indicies[index_to_read][timestep_index]
                    if math.isnan(matched_timestep_index):
                        name_A = None
                        name_B = None
                    else:
                        name_A = self.bandA_names[matched_timestep_index]
                        name_B = self.bandB_names[matched_timestep_index]

                #If the matching timestep exists
                if name_A is None:
                    bandA = np.empty(shape=self.img_shape)
                    bandA[:] = np.nan
                    bandB = np.empty(shape=self.img_shape)
                    bandB[:] = np.nan
                else:

                    bandA = cv2.imread(self.image_directory + "/" + name_A, -1)
                    bandB = cv2.imread(self.image_directory + "/" + name_B, -1)

                    if correct == True:
                        bandA, bandB = correct_sample(
                            bandA=bandA,
                            bandB=bandB,
                            dark_name_A=self.dark_path_A + "/" + self.matched_dark_names_A[index_to_read],
                            dark_name_B=self.dark_path_B + "/" + self.matched_dark_names_B[index_to_read],
                            vin_mask_A=self.vin_mask_A,
                            vin_mask_B=self.vin_mask_B,
                            reg_trans=self.reg_trans_matrix,
                            smm_A=self.smm_A,
                            smm_B=self.smm_B)


                if timestep_to_read == 0:
                    bandA_outputs.append(bandA)
                    bandB_outputs.append(bandB)
                else:
                    #Append to the array representing the temporal data
                    temporal_array[index_in_chunk, timestep_index, 0, :, :] = bandA
                    temporal_array[index_in_chunk, timestep_index, 1, :, :] = bandB
        return bandA_outputs, bandB_outputs, temporal_array




    def apply_quality_models(self, chunk_size=50):
        '''
        Reading and correcting chunk_size samples at a time, apply the quality classification
        models.
        For each image pair, identify the corresponding pairs as close as possible to the
        timestep required by the models. If no temporal data is available skip applying the
        models to this image pair, and store the prediction as np.nan.
        '''

        # Make use of GPU if available:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = "cpu"
        print("Using device:" + device)
        torch.cuda.empty_cache()

        # Load the model definitions and trained weights
        precip_model = QualityModelFunctions.get_triple_branched_resnet18(device)
        cloud_model = QualityModelFunctions.get_triple_branched_resnet18(device)
        print("Loading trained model weights:")
        precip_model.load_state_dict(torch.load("C:/Users/ggp24ash/PycharmProjects/QualityIndexModels/SavedModelWeights/PrecipitationModel_V2.pth", weights_only=True))
        cloud_model.load_state_dict(torch.load("C:/Users/ggp24ash/PycharmProjects/QualityIndexModels/SavedModelWeights/ObsCloudModel.pth", weights_only=True))
        precip_model.eval()
        cloud_model.eval()

        # Load data in subsets of size "chunk_size", reducing the memory required
        n_chunks = math.ceil(len(self.bandA_names) / chunk_size)
        for chunk in range(1, n_chunks + 1):
            start_index = (chunk - 1) * chunk_size
            if chunk == n_chunks:
                end_index = len(self.bandA_names) - 1
            else:
                end_index = (chunk * chunk_size) - 1
            print("Applying models to image pairs indexed: " + str(start_index) + " to " + str(end_index))
            self.chunk_indexes = np.arange(start_index, end_index + 1).astype(np.uint16)

            # Read and correct this chunk of data
            self.batch_bandA, self.batch_bandB, self.chunk_temporal = self.read_and_correct_selected_indexes(indexes_to_read=self.chunk_indexes, correct=True, temporal=True)
            # Load it as a pytorch tensor
            eval_set = QualityModelFunctions.ImageLoader(device=device, indexes=self.chunk_indexes, bandA_list=self.batch_bandA, bandB_list=self.batch_bandB, temporal_available=self.all_timestep_data_available, temporal_array=self.chunk_temporal, timesteps_provided=self.requested_timesteps)
            # Set up DataLoader to iterate over samples
            dataloader = DataLoader(eval_set, batch_size=1, shuffle=False, drop_last=False)

            # Iterating over each observation in the chunk
            for index, obs in enumerate(iter(dataloader)):
                x_p, x_c, apply_models, image_index = obs
                if apply_models == True: #If the necessary temporal data was sucessfully matched
                    # Normalise the samples with ImageNet mean and standard deviation
                    x_p_norm = QualityModelFunctions.scale_and_norm_batch(x_p, device)
                    # Predict using the model, and move the output from the GPU if applicable
                    prediction_p = precip_model(x_p_norm).cpu().detach().numpy()[0,:]
                    if chunk == 1 and index == 0:
                        self.all_precip_predictions = prediction_p
                    else:
                        self.all_precip_predictions = np.concatenate((self.all_precip_predictions, prediction_p))

                    # Repeat this process, predicting using the cloud model
                    x_c_norm = QualityModelFunctions.scale_and_norm_batch(x_c, device)
                    prediction_c = cloud_model(x_c_norm).cpu().detach().numpy()[0,:]
                    if chunk == 1 and index == 0:
                        self.all_cloud_predictions = prediction_c
                    else:
                        self.all_cloud_predictions = np.concatenate((self.all_cloud_predictions, prediction_c))
                else:
                    if chunk == 1 and index == 0:
                        self.all_precip_predictions = np.array([np.nan])
                        self.all_cloud_predictions = np.array([np.nan])
                    else:
                        self.all_precip_predictions = np.concatenate((self.all_precip_predictions, np.array([np.nan])))
                        self.all_cloud_predictions = np.concatenate((self.all_cloud_predictions, np.array([np.nan])))

    def iterate(self, b, method="basic", mins=5, correct=True, load_temporal=True):
        '''Read the batch of data necessary for processing of image at index b.
        Typically, this would be the X mins of data surrounding the current image,
        (or the first or last X mins for applicable samples).'''

        #Determine which image indexes we want to load
        print("Current time:")
        current_time = self.times[b]
        print(current_time)
        self.previous_indexes = self.chunk_indexes

        #If we are in the first X/2 mins of the sequence
        if (current_time - self.times[0]).seconds < (mins/2 * 60):
            #Select the frist X mins of data
            self.chunk_end_time = self.times[0] + timedelta(minutes=mins)
            # Now select indicies of the "times" list which are in [start, end]
            self.chunk_indexes = [i for i in range(len(self.times)) if self.times[i] <= self.chunk_end_time]
        elif (self.times[-1] - current_time).seconds < (mins/2 * 60): #We are in the last X/2 mins of the sequence
            self.chunk_start_time = self.times[-1] - timedelta(minutes=mins)
            self.chunk_indexes = [i for i in range(len(self.times)) if self.times[i] >= self.chunk_start_time]
        else: #Take the X mins surrounding the current image:
            self.chunk_start_time =  current_time - timedelta(minutes=mins/2)
            self.chunk_end_time = current_time + timedelta(minutes=mins/2)
            self.chunk_indexes = [i for i in range(len(self.times)) if ((self.times[i] >= self.chunk_start_time) and (self.times[i] <= self.chunk_end_time))]

        chunk_times = [self.times[i] for i in self.chunk_indexes]
        print(chunk_times)

        #Determine overlap with previous batch
        self.overlap = list(set(self.chunk_indexes).intersection(set(self.previous_indexes)))
        self.overlap.sort()
        self.overlap_indexes_in_chunk = [self.previous_indexes.index(i) for i in self.overlap]
        print("Overlap with previous batch:")
        print(self.overlap)

        #Select overlapping images from previous batch (including temporal if applic)
        self.batch_bandA = [self.batch_bandA[i] for i in self.overlap_indexes_in_chunk]
        self.batch_bandB = [self.batch_bandB[i] for i in self.overlap_indexes_in_chunk]
        try:
            self.chunk_temporal
        except:
            pass
        else:
            self.chunk_temporal = self.chunk_temporal[self.overlap_indexes_in_chunk,:,:,:,:]


        self.intermed_chunk_indicies = self.overlap.copy() #Keep track of what indexes we currently have data stored for
        # Load any new images that are needed
        print("New pairs to read:")
        self.new_indexes = list(set(self.chunk_indexes).difference(set(self.previous_indexes)))
        self.new_indexes.sort()
        print([self.times[i] for i in self.new_indexes])
        # For each new index, read the image and slot it into the correct position in
        # the bandA, bandB and temporal data structures, based on the image index.
        print("Loading new pairs:")
        if len(self.new_indexes) != 0:
            new_bandA, new_bandB, new_temporal = self.read_and_correct_selected_indexes(indexes_to_read=self.new_indexes, correct=correct, temporal=load_temporal)

        print("Saving new pairs:")
        if len(self.intermed_chunk_indicies) == 0: #If the batch is empty, then just take the newly calculated image lists
            self.batch_bandA = new_bandA
            self.batch_bandB = new_bandB
            self.chunk_temporal = new_temporal
            self.intermed_chunk_indicies = list(self.new_indexes)
        else: #Otherwise we need to place each sample at the correct index:
            new_items_added = 0
            for ni in self.new_indexes:
                #Place the data
                #[img_pair_index, timestep, channel, h, w]
                temp_array_to_place = np.empty(shape=(1, len(self.requested_timesteps), 2, self.img_shape[0], self.img_shape[1]))
                temp_array_to_place[ 0, :, :, :, :] = new_temporal[new_items_added, :, :, :, :]
                if ni > self.intermed_chunk_indicies[-1]: #If the new image is later in the sequence
                    self.batch_bandA.append(new_bandA[new_items_added])
                    self.batch_bandB.append(new_bandB[new_items_added])
                    self.chunk_temporal = np.concatenate((self.chunk_temporal, temp_array_to_place), axis=0)
                    self.intermed_chunk_indicies.append(ni)
                else:
                    location = bisect.bisect_left(self.intermed_chunk_indicies, ni) #index at which to insert the new data
                    self.batch_bandA.insert(location, new_bandA[new_items_added])
                    self.batch_bandB.insert(location, new_bandB[new_items_added])
                    self.chunk_temporal = np.insert(self.chunk_temporal, location, temp_array_to_place, axis=0)
                    self.intermed_chunk_indicies.insert(location, ni)
                new_items_added += 1

    def view_current_chunk(self, band="A", timesteps=[-10, 10]):
        for index in range(0, len(self.chunk_indexes)):
            if band == "A":
                band = 0
                t0_imgs = self.batch_bandA
                names = self.bandA_names
            elif band == "B":
                band = 1
                t0_imgs = self.batch_bandB
                names = self.bandB_names

            tsi_1 = self.requested_timesteps.index(timesteps[0])
            tsi_2 = self.requested_timesteps.index(timesteps[1])
            fig, axs = plt.subplots(ncols=3)
            axs[0].imshow(self.chunk_temporal[index,tsi_1,band,:,:], cmap="gray")
            axs[1].imshow(t0_imgs[index], cmap="gray")
            axs[2].imshow(self.chunk_temporal[index, tsi_2,band,:,:], cmap="gray")
            axs[1].set_title(names[self.chunk_indexes[index]])
            plt.show()
    def iterate_real_time_OLDVERSION(self, b, chunk_size_m=30):
        '''Return the first 30mins (or specified time period) of the sequence, then
        incrementally return a batch containing the next image and all samples within
        the previous 30mins.

        This function assumes that at least chunk_size mins of consecutive recordings are provided.
        The function is written such that we step forward to include one new image each batch.
        '''

        #Decide which samples to load
        #For b = 0, load the first 30mins of data

        if b == 0:
            print("Loading batch zero:")
            self.batch_start_time = self.times[0]
            self.batch_end_time = self.batch_start_time + timedelta(minutes=chunk_size_m)
            #Now select indicies of the "times" list which are in [start, end]
            self.chunk_indicies = [i for i in range(len(self.times)) if self.times[i] <= self.batch_end_time]

            self.flank_mask = cv2.imread(self.volc_dict["flank_mask_path"], -1)

        else:
            #For every batch after, step forward by one image, and select the previous 30mins of recordings
            self.previous_indicies = self.chunk_indicies #Indicies that were selected in the previous batch
            self.batch_end_time = self.times[self.previous_indicies[-1] + 1]
            self.batch_start_time = self.batch_end_time - timedelta(minutes=chunk_size_m)
            self.chunk_indicies = [i for i in range(len(self.times)) if ((self.times[i] <= self.batch_end_time) and (self.times[i] >= self.batch_start_time))]

        #Correct them (dark, clear and registration)
        if b==0:
            for index_to_read in self.chunk_indicies:
                #print(index_to_read)
                #print("Reading image:" + str(self.bandA_names[index_to_read]))
                bandA = cv2.imread(self.image_directory + "/" + self.bandA_names[index_to_read], -1)
                bandB = cv2.imread(self.image_directory + "/" + self.bandB_names[index_to_read], -1)
                self.batch_bandA.append(bandA)
                self.batch_bandB.append(bandB)
        else:
            print("Moving batch forward by one image.")
            #Read the additional image
            self.batch_bandA.append(cv2.imread(self.image_directory + "/" + self.bandA_names[self.chunk_indicies[-1]], -1))
            self.batch_bandB.append(cv2.imread(self.image_directory + "/" + self.bandB_names[self.chunk_indicies[-1]], -1))
            #Drop samples from the start of the batch
            extra_count = len(self.batch_bandA) - len(self.chunk_indicies)
            if extra_count > 0:
                self.batch_bandA = self.batch_bandA[extra_count:]
                self.batch_bandB = self.batch_bandB[extra_count:]

        if self.correct == True:
            if b == 0:
                #Calculate the vignette masks
                print("Creating vignette masks.")
                clear_img_A = cv2.imread(self.clear_path_A, -1).astype(np.float32)
                self.vin_mask_A = np.divide(clear_img_A, np.max(clear_img_A))
                clear_img_B = cv2.imread(self.clear_path_B, -1).astype(np.float32)
                self.vin_mask_B = np.divide(clear_img_B, np.max(clear_img_B))

                for pair_index in self.chunk_indicies:

                    original_A_to_plot = self.batch_bandA[pair_index].copy()
                    original_B_to_plot = self.batch_bandB[pair_index].copy()
                    self.batch_bandA[pair_index], self.batch_bandB[pair_index] = correct_sample(bandA = self.batch_bandA[pair_index],
                                                                                                bandB = self.batch_bandB[pair_index],
                                                                                                dark_name_A = self.dark_path_A + "/" + self.matched_dark_names_A[pair_index],
                                                                                                dark_name_B = self.dark_path_B + "/" + self.matched_dark_names_B[pair_index],
                                                                                                vin_mask_A = self.vin_mask_A,
                                                                                                vin_mask_B = self.vin_mask_B,
                                                                                                reg_trans = self.reg_trans_matrix,
                                                                                                smm_A=self.smm_A,
                                                                                                smm_B=self.smm_B)


            else:
                #Correct just the last sample
                self.batch_bandA[-1], self.batch_bandB[-1] = correct_sample(bandA=self.batch_bandA[-1],
                                                                                                        bandB=self.batch_bandB[-1],
                                                                                                        dark_name_A = self.dark_path_A + "/" + self.matched_dark_names_A[-1],
                                                                                                        dark_name_B = self.dark_path_B + "/" + self.matched_dark_names_B[-1],
                                                                                                        vin_mask_A = self.vin_mask_A,
                                                                                                        vin_mask_B = self.vin_mask_B,
                                                                                                        reg_trans = self.reg_trans_matrix,
                                                                                                        smm_A=self.smm_A,
                                                                                                        smm_B=self.smm_B)
        remaining_iterations = len(self.bandA_names) - 1 - self.chunk_indicies[-1]
        return self.batch_bandA, self.batch_bandB, remaining_iterations

    ###################### Image Calculator Functions #############################
    def estimate_backgrounds(self, method):
        '''For the current batch of data which has been read by calling the
        "iterate" method and is stored as self.batch_bandA/B, calculate the
        background estimations for every image, using the specified method.'''

        #Select backgrounds from the previous chunk which overlap with the current chunk
        try: #If we are iterating and have calculated overlap with the previous chunk
            self.bgs_A = [self.bgs_A[i] for i in self.overlap_indexes_in_chunk]
            self.bgs_B = [self.bgs_B[i] for i in self.overlap_indexes_in_chunk]
        except: #Otherwise every index is "new"
            self.new_indexes = self.chunk_indexes.copy()

        #If there are new pairs in this chunk, calculate their backgrounds:
        new_bgs_A = []
        new_bgs_B = []
        if len(self.new_indexes) != 0:
            for new_index in self.new_indexes:
                index_in_this_chunk = self.chunk_indexes.index(new_index)
                bgA_new, bgB_new = method(self.batch_bandA[index_in_this_chunk], self.batch_bandB[index_in_this_chunk], self.flank_mask)
                new_bgs_A.append(bgA_new)
                new_bgs_B.append(bgB_new)

        if len(self.bgs_A) == 0: #If this is an entirely distinct chunk from the previous, just save the lists calculated for the new indexes
            self.bgs_A = new_bgs_A
            self.bgs_B = new_bgs_B

        else:
            self.indexes_of_stored_bgs = self.overlap.copy()
            new_items_added = 0
            for ni in self.new_indexes:
                if ni > self.indexes_of_stored_bgs[-1]: #Add the new backgrounds to the end of the relevant lists
                    self.bgs_A.append(new_bgs_A[new_items_added])
                    self.bgs_B.append(new_bgs_B[new_items_added])
                    self.indexes_of_stored_bgs.append(ni)
                else: #Calculate where in the existing list of backgrounds to place the new ones
                    location = bisect.bisect_left(self.indexes_of_stored_bgs, ni)  # index at which to insert the new data
                    self.bgs_A.insert(location, new_bgs_A[new_items_added])
                    self.bgs_B.insert(location, new_bgs_B[new_items_added])
                    self.indexes_of_stored_bgs.insert(location, ni)

                new_items_added += 1

    def calculate_absorbance(self, correction=None):
        '''Calculate the absorbance for the current batch of data, assuming background images have been estimated.
        The result is stored as an array of shape [n_frames, height, width].'''

        existing_AA = True
        try:
            # Select absorbance values from the previous chunk which overlap with the current chunk
            previous_chunk_AA = self.AA.copy()
            self.AA = previous_chunk_AA[self.overlap_indexes_in_chunk, :, :].copy()
            self.AA.mask = previous_chunk_AA[self.overlap_indexes_in_chunk, :, :].mask.copy() #Copy over the mask too
        except:
            existing_AA = False
            self.AA = np.empty(shape=(0, self.img_shape[0], self.img_shape[1]))

        # If there are new pairs in this chunk, calculate their absorbance images:
        if len(self.new_indexes) != 0:
            new_AA_array = np.ma.empty(shape=(len(self.new_indexes), self.img_shape[0], self.img_shape[1]))
            new_calculated = 0
            for new_index in self.new_indexes:
                index_in_this_chunk = self.chunk_indexes.index(new_index) #Find the position of the data for this index in this chunk
                AA_img = calculate_AA(self.batch_bandA[index_in_this_chunk], self.batch_bandB[index_in_this_chunk], self.bgs_A[index_in_this_chunk], self.bgs_B[index_in_this_chunk])
                if correction != None:
                    AA_img = correction(AA_img, self.flank_mask)
                new_AA_array[new_calculated, :, :] = AA_img
                new_AA_array[new_calculated, :, :].mask = AA_img.mask
                new_calculated += 1

        if self.AA.shape[0] == 0: #If this is an entirely distinct chunk from the previous, just save the AA images calculated for the new indexes
            self.AA = new_AA_array
            self.AA.mask = new_AA_array.mask

        else: #Place the new AA images in the correct position in the array
            self.indexes_of_stored_AAs = self.overlap.copy()
            new_items_added = 0
            for ni in self.new_indexes:
                #If the image needs to be joined to the end of the array
                new_AA = np.ma.empty(shape=(1, self.img_shape[0], self.img_shape[1]))
                new_AA[:,:,:] = new_AA_array[new_items_added, :, :]
                new_AA.mask = new_AA_array[new_items_added, :, :].mask
                if ni > self.indexes_of_stored_AAs[-1]:
                    self.AA = np.ma.concatenate([self.AA, new_AA], axis=0)
                    self.indexes_of_stored_AAs.append(ni)
                else:
                    location = bisect.bisect_left(self.indexes_of_stored_AAs, ni)  # index at which to insert the new data
                    np.insert(self.AA, location, new_AA, axis=0)
                    self.indexes_of_stored_AAs.insert(location, ni)

                new_items_added += 1



    def translate_absorbance(self):
        #TODO
        '''Calibrate the absorbance images such that the median in a specific region
        e.g. the flank is zero, using simple subtraction.'''
        pass
    ################################################################################
    def read_spectrometer_data(self, path):
        '''Read the time series of spectrometer measurements from the specified
        file path, and match to the timestamps of the images in this sequence.'''

        cd_df = pd.read_csv(path)
        #For each entry, create a datetime object
        cd_df["datetime"] = cd_df["Time"].apply(map_spectrometer_timestamp_to_datetime)

        #Select only spectrometer readings taken in-sync with camera images
        for time in self.times:
            #Check if there's a corresponding spectrometer reading
            #Save a record of whether the spectrometer data is available as a boolean variable
            #If so, save the column density and associated error
            corr_reading = cd_df[cd_df["datetime"] == time]
            if corr_reading.shape[0] == 0:
                self.spectra_exists.append(False)
                self.spectra.append(np.nan)
            elif corr_reading.shape[0] == 1:
                self.spectra_exists.append(True)
                self.spectra.append(corr_reading["Column density"].item())
            else:
                print("ERROR: Multiple simultaneous spectrometer readings.")

    def select_and_read_indexes_for_spectrometer_FOV_match(self, indexes=None, random_seed = 42):
        '''Assuming that the quality indexes have been calculated and that spectrometer data
        has been read in select image pairs indexes for the FOV match:
        - filter for good quality data
        - select non-zero and non-NAN sprecta values
        - randomly select from these'''
        random.seed(42)

        if indexes == None:
            all_indexes = np.arange(0, len(self.bandA_names))
            good_quality_indexes = [i for i in all_indexes if self.all_precip_predictions[i] < 0.5]
            good_quality_indexes = [i for i in good_quality_indexes if self.all_cloud_predictions[i] < 0.5]
            print(good_quality_indexes)

            if len(good_quality_indexes) > 100:
                self.chunk_indexes = random.sample(good_quality_indexes, k=100)
                self.chunk_indexes.sort()
            else:
                self.chunk_indexes = good_quality_indexes
        else:
            self.chunk_indexes = indexes

        print("Reading data for spectrometer FOV match")

        self.batch_bandA, self.batch_bandB, self.chunk_temporal = self.read_and_correct_selected_indexes(indexes_to_read=self.chunk_indexes, correct=True, temporal=True)


    def find_spectrometer_FOV(self, s=5, plot=True):
        '''Assuming that absorbance images have been calculated and spectrometer data is
        read in, run a cross-correlation to estimate the pixel at which the spectrometer
        is centered.

        Calculations are run for every s-th pixel (simply by selecting that pixel, no
        downsampling or averaging is used).'''

        #Create results array, with extra rows and columns to for ease of filling the downsampled results
        correlation_vis = np.empty((self.AA.shape[1] + s, self.AA.shape[2] + s))

        batch_spectra = [self.spectra[i] for i in self.chunk_indexes]
        for row in range(0, self.AA.shape[1], s):
            for column in range(0, self.AA.shape[2], s):
                absorbance_series = self.AA[:, row, column]
                #Only proceed if no element of the absorbance series is masked out (i.e all absorbance values exist)
                if np.ma.is_masked(absorbance_series) == False:
                    correlation = pearsonr(absorbance_series, batch_spectra).statistic
                    if correlation != np.nan:
                        correlation_vis[row: row+s, column:column+s] = np.ones(shape=(s,s)) * correlation
        #Remove the extra rows of the results array (only used for ease of coding the line above)
        correlation_vis = correlation_vis[:self.AA.shape[1] + 1, :self.AA.shape[2] + 1]

        #Select the pixel with maximum correlation and return the coordinates:
        max_flattened_index = np.nanargmax(correlation_vis)
        n_cols = correlation_vis.shape[1]
        max_row = np.floor(max_flattened_index/n_cols)
        max_column = max_flattened_index - (max_row * n_cols)
        print(max_column)
        max_row = int(max_row)
        max_column = int(max_column)

        if plot == True:
            fig, ax = plt.subplots()
            img = ax.imshow(correlation_vis)
            ax.set_title("Absorbance/Spectrometer CD Correlation R-value")
            circle = plt.Circle((max_column, max_row), radius = 10)
            ax.add_patch(circle)
            fig.colorbar(img, ax=ax)
            plt.show()

            fig2, ax2 = plt.subplots()
            absorbance_to_plot = self.AA[:, max_row, max_column].copy()
            mag_diff = np.sum(batch_spectra) / np.sum(absorbance_to_plot)
            absorbance_to_plot = absorbance_to_plot * mag_diff
            x_vals = np.arange(0, self.AA.shape[0])
            ax2.plot(x_vals, absorbance_to_plot, label="Absorbance")
            ax2.plot(x_vals, batch_spectra, label="Spectrometer CD")
            plt.xlabel("Image count (selected for FOV match)")
            plt.ylabel("AA (Scaled) and Column Density")
            plt.legend()
            plt.show()

        self.spec_CFOV = (max_row, max_column)
        print("Spectrometer CFOV: " + str(self.spec_CFOV))

    def calculate_calibration_curve(self, plot=False):
        '''Assuming that spectrometer readings have been loaded, and that the FOV
        has been determined, calculate the calibration line using a least squares fit.

        Starting with a simple linear fit: column_density = m * absorbance + c
        '''

        absorbance_inputs = self.AA[:,int(self.spec_CFOV[0]), int(self.spec_CFOV[1])]
        spectra_target = self.spectra[self.chunk_indexes[0]:self.chunk_indexes[-1] + 1]
        coeffs = np.polyfit(x=absorbance_inputs, y=spectra_target, deg=1)
        line_fn = np.poly1d(coeffs)
        self.calib_fn = line_fn

        #Plotting
        if plot == True:
            fig, axs = plt.subplots()
            axs.scatter(absorbance_inputs, spectra_target)
            x_range = np.linspace(np.min(absorbance_inputs), np.max(absorbance_inputs))
            line_y_vals = line_fn(x_range)
            axs.plot(x_range, line_y_vals)
            axs.set_xlabel("Apparent Absorbance")
            axs.set_ylabel("Column density (molecules/cm^2)")
            plt.title("Calibration curve")
            plt.show()

    def calibrate_AA(self):
        '''Assuming absorbance images and the calibration curve have been computed,
        calibrate each pixel of the absorbance images, to molecules/cm^2.'''
        self.CD = self.calib_fn(self.AA)

    def convert_molecules_to_mass(self):
        '''Convert SO2 column densities in molecules/cm^2 to kg/m^2.
        Conversion used:
        1 molecule cm^-2
        = 1/6.02e23 moles cm^-2
        = 1/6.02e23 * 64.06 g cm^-2 (Platt and Stutz, 2008, p.12)
        = 1/6.02e23 * 64.06 * 10e-3 kg cm^-2
        = 1/6.02e23 * 64.06 * 10e-3 * 10e4 kg m^-2
        = 1/6.02e22 * 64.06 kg m^-2
        '''
        multiplier = 64.06/(6.02*10e22)
        self.CD = self.CD * multiplier
        show(self.CD[-1,:,:], cmap="YlGnBu_r")

class CameraGeometry():

    def __init__(self, volcano_dictionary, camera_dictionary):
        '''Initialise the camera geometry variables, reading in the relevant information from the
        volcano dictionary.'''

        self.cam_lat = volcano_dictionary['cam_lat'] #Position of the camera
        self.cam_lon = volcano_dictionary['cam_lon']

        self.cam = geonum.GeoPoint(self.cam_lat, self.cam_lon, name="camera", auto_topo_access=True)

        self.cam_height = volcano_dictionary['cam_height'] #Height of the camera above the ground
        if self.cam_height != None:
            self.cam.altitude += self.cam_height
        self.pixel_count = camera_dictionary['pixels']
        self.cam_FOV_angle = camera_dictionary['FOV_angle']
        self.alpha_h = self.cam_FOV_angle[1] / self.pixel_count[1]
        self.alpha_v = self.cam_FOV_angle[0] / self.pixel_count[0]
        print("Horizontal pixel size: " + str(self.alpha_h) + " deg")
        print("Vertical pixel size: " + str(self.alpha_v) + " deg")

        self.ref_lat = volcano_dictionary['ref_lat'] #Position of the reference point in real world coords
        self.ref_lon = volcano_dictionary['ref_lon']
        self.ref = geonum.GeoPoint(self.ref_lat, self.ref_lon, name="ref", auto_topo_access=True)
        self.ref_pixel_coords = volcano_dictionary['ref_pixel_coords'] #POsition of the reference point in the image coordinates

        self.crater_lat = volcano_dictionary['crater_lat']
        self.crater_lon = volcano_dictionary['crater_lon']
        self.crater = geonum.GeoPoint(self.crater_lat, self.crater_lon, name="crater", auto_topo_access=True)


    def calculate_camera_angle(self):
        '''Based on the provided camera and source coordinates, and the pixel index of the vent in the image,
        calculate the azimuth and elevation angle of the camera.'''
        print("Calculating camera angle:")

        self.cam_to_ref = self.ref - self.cam
        self.cam_to_ref.set_anchor(self.cam)

        #Calculate the angle of the vector between the camera position and vent position
        source_elev = self.cam_to_ref.elevation
        source_azim = self.cam_to_ref.azimuth

        #Calculate the angle at which the camera must be pointing (the angle required to bring the secified pixel to the CFOV, plus the angle to bring the CFOV to the source location)
        if self.pixel_count[0] % 2 != 0:
            print("ERROR: Geometry calc assumes image dimensions are divisible by two.")
        elif self.pixel_count[1] % 2 != 0:
            print("ERROR: Geometry calc assumes image dimensions are divisible by two.")
        # How many pixel steps from the CFOV is the reference point?
        angular_steps_h = self.ref_pixel_coords[1] - (self.pixel_count[1]/2) - 0.5
        angular_steps_v = self.ref_pixel_coords[0] - (self.pixel_count[0]/2) - 0.5
        self.cfov_azim = -1 * angular_steps_h * self.alpha_h + source_azim
        self.cfov_elev = angular_steps_v * self.alpha_v + source_elev

        print("Camera CFOV elevation angle: " + str(self.cfov_elev))
        print("Camera CFOV elevation azimuth: " + str(self.cfov_azim))

    def calculate_CFOV_location(self):
        '''
        Calculate the intersection of the CFOV of the camera with the image plane.
        Assume that the image plane is vertical and runs through the crater center, perpendicular to the vector from camera to CFOV.
        '''
        print("Calculating CFOV location on image plane.")
        #Find the azimuth angle of the vector from the camera to the crater
        self.cam_to_crater = self.crater - self.cam
        self.cam_to_crater.set_anchor(self.cam)

        #Create a vector which is on the plane running through the crater center, perfectly vertical and at an angle perpendicular to the
        #long and lat components of the CFOV line (when viewed from a birds eye above).
        if self.cfov_azim >= self.cam_to_crater.azimuth:
            offset_angle = 90 #TODO Does this work if the cfov azim is near 180 or -180?
        else:
            offset_angle = -90
        self.plane_vector = geonum.GeoVector3D(azimuth=self.cfov_azim + offset_angle, elevation=0, dist_hor=self.cam_to_crater.dist_hor/2, name="img_plane_direction")
        self.plane_vector.set_anchor(self.crater)

        #Create a vector extending from the camera, along the CFOV, far beyond the image plane
        self.cfov_dir_vector = geonum.GeoVector3D(azimuth=self.cfov_azim, dist_hor=self.cam_to_crater.dist_hor * 2, elevation=self.cfov_elev)
        self.cfov_dir_vector.set_anchor(self.cam)

        #Then find the horizontal point of intersection of these vectors
        self.horizontal_cam_to_poi = self.cfov_dir_vector.intersect_hor(self.plane_vector) #Vector extending from cam to poi (but only lon and lat components)
        #Calculate the vertical component:
        self.cam_to_poi_dz = np.tan((self.cfov_elev/180) * np.pi) * self.horizontal_cam_to_poi.dist_hor * 1000 #Using trigonometry
        #Add the calculated offsets to the camera position to find the POI
        self.cfov = self.cam.offset(azimuth=self.horizontal_cam_to_poi.azimuth, dist_hor=self.horizontal_cam_to_poi.dist_hor, dist_vert=self.cam_to_poi_dz)
        self.cfov.name="CFOV"

        #Vector from camera extending exactly to CFOV which is used for plotting later
        self.cam_to_cfov = self.cfov - self.cam
        self.cam_to_cfov.set_anchor(self.cam)


    def plot_camera_geometry(self):
        print("Creating plot of camera geometry with geonum package.")

        s = geonum.GeoSetup()
        s.add_geo_points(self.cam, self.crater, self.ref)
        s.set_borders_from_points()
        s.add_geo_vectors(self.cam_to_crater, self.cam_to_cfov, self.plane_vector)

        map2D = s.plot_2d()
        plt.show()

        #map3D = s.plot_3d()

    def map_img_coords_to_world_plane(self, coords_array):
        '''Take in a 3D array of size height x width x 2 which holds Y and X
        coordinates (in pixels) for hxw points.
        Calculate and return the Y and X coordinates of each point
        on the true-size image plane with axes centered at the CFOV.'''

        #TODO ignore any masked points

        #Calculate the height of the CFOV (ignoring the height the altitude of the camera)
        h_cfov = self.cam_to_cfov.dist_hor * np.tan((self.cfov_elev/180) * np.pi)
        print(h_cfov)
        #Check this matches the geonum point! - Yes :)

        #For each point calculate the height above or below the COFV
        vertical_coords = coords_array[:,:,0]
        angles_within_img = (self.pixel_count[0] - vertical_coords) * self.alpha_v
        image_base_angle = self.cfov_elev - self.cam_FOV_angle[0]/2
        angles_with_horiz = angles_within_img + image_base_angle
        #show(angles_with_horiz)
        heights_rel_cfov = self.cam_to_cfov.dist_hor * np.tan((angles_with_horiz/180) * np.pi) - h_cfov
        #show(heights_rel_cfov)

        #Calculate the x-distance of each point from the CFOV
        horizontal_coords = coords_array[:,:,1]
        #Calculate distance along the cfov from plane to camera
        d_cam_cfov = self.cam_to_cfov.magnitude
        #Calculate the angle from the cfov
        horiz_angles_from_cfov = (horizontal_coords - self.pixel_count[1]/2) * self.alpha_h
        #show(horiz_angles_from_cfov)
        #Calculate distances
        horiz_pos_rel_cfov = np.tan((horiz_angles_from_cfov/180) * np.pi) * d_cam_cfov
        #show(horiz_pos_rel_cfov)

        return np.stack(arrays=[heights_rel_cfov, horiz_pos_rel_cfov], axis=-1)

    def calculate_pixel_sizes(self):
        '''Calculate the height, width and area of each pixel in real-world image plane.'''
        #Let pixel coordinates indicate the top left of that pixel
        #X is horixontal direction (index 1), Y is vertical direction (index 0)
        x_vals = np.arange(0, self.pixel_count[1])
        y_vals = np.arange(0, self.pixel_count[0])
        X, Y = np.meshgrid(x_vals, y_vals)

        coords_array = np.stack(arrays=[Y, X], axis=-1)
        world_coords_tl = self.map_img_coords_to_world_plane(coords_array)
        world_coords_br = self.map_img_coords_to_world_plane(coords_array + 1)
        pixel_widths = world_coords_br[:,:,1] - world_coords_tl[:,:,1]
        pixel_heights = world_coords_tl[:,:,0] - world_coords_br[:,:,0]

        show(pixel_widths, title="Pixel Widths")
        show(pixel_heights, title="Pixel Heights")
        pixel_areas = np.multiply(pixel_widths, pixel_heights)
        show(pixel_areas, title="Pixel Areas")

        real_image_size = (world_coords_tl[0,0,0] - world_coords_br[-1,-1,0], world_coords_br[-1,-1,1] - world_coords_tl[0,0,1])
        print(real_image_size)

class IntegrationObject():
    '''Class to represent line or circle to integrate over.
    Any line or circle should be a collection of pixels which
    intersect that line, plus the length of each intersection.
    We also want to know the direction perpendicular to the
    line at every pixel.'''

    def __init__(self):
        pass

class Line(IntegrationObject):
    def __init__(self):
        #Provide the two endpoints

class Circle(IntegrationObject):
    def __init__(self):
        #Provide the center point and radius






























