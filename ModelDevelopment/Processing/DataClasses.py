import cv2
from datetime import datetime, timedelta
import os

def map_image_name_to_time(image_name):
    date_str = image_name.split("_")[0][0:10]
    time_str = image_name.split("_")[0][11:17]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    return datetime_obj

def shutter_speed_from_band_img_name(image_name):
    return int(image_name.split("_")[3][:-2])
class Sequence():
    '''Class representing a sequence of data to be processed.'''
    def __init__(self):
        self.volc_dict = None
        self.previous_indicies = []
        self.batch_bandA = []
        self.batch_bandB = []


    def set_volcano_dictionary(self, dictionary):
        self.volc_dict = dictionary

    def read_and_match(self, directory_path, correct=True):
        '''Read all sample names from the given directory. Procuce a list
        of band A samples, matched to corresponding bandB samples.
        If correct is True, match names of files for dark and vignette correction,
        then initialse the registration transform, using the data from the volcano dictionary.'''
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
                print(pair_index)
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

            #Registration transform:
            self.reg_trans_matrix = cv2.getPerspectiveTransform(self.volc_dict['registration_points_B'], self.volc_dict['registration_points_A'])

            #Masks for permanent marks on sensor to be infilled
            self.sensor_mark_mask_path_A = self.volc_dict["sensor_marks_mask_A"]
            self.sensor_mark_mask_path_B = self.volc_dict["sensor_marks_mask_A"]
        else:
            print("NOTE: No corrections are being applied to this image sequence.")


    def iterate(self, b, chunk_size_m=30):
        '''Return the first 30mins (or specified time period) of the sequence, then
        incrementally return images in range [15mins + dt_b, 30mins + dt_b].
        Where dt_b denotes the timestep difference between the last image
        returned in the previous batch and the next sample.

        This function assumes that at least chunk_size mins of consecutive recordings are provided.
        The function is written such that we step forward to include one new image each batch.
        '''

        #Decide which samples to load
        #For b = 0, load the first 30mins of data

        if b == 0:
            self.batch_start_time = self.times[0]
            self.batch_end_time = self.batch_start_time + timedelta(minutes=chunk_size_m)
            #Now select indicies of the "times" list which are in [start, end]
            self.chunk_indicies = [i for i in range(len(self.times)) if self.times[i] <= self.batch_end_time]

        else:
            #For every batch after, step forward by one image, and select the previous 30mins of recordings
            self.previous_indicies = self.chunk_indicies #Indicies that were selected in the previous batch
            self.batch_end_time = self.times[self.previous_indicies[-1] + 1]
            self.batch_start_time = self.batch_end_time - timedelta(minutes=chunk_size_m)
            self.chunk_indicies = [i for i in range(len(self.times)) if ((self.times[i] <= self.batch_end_time) and (self.times[i] >= self.batch_start_time))]

        #Correct them (dark, clear and registration)
        if b==0:
            for index_to_read in self.chunk_indicies:
                print(index_to_read)
                print("Reading image:" + str(self.bandA_names[index_to_read]))
                bandA = cv2.imread(self.image_directory + "/" + self.bandA_names[index_to_read], -1)
                bandB = cv2.imread(self.image_directory + "/" + self.bandB_names[index_to_read], -1)
                self.batch_bandA.append(bandA)
                self.batch_bandB.append(bandB)
        else:
            print("Iterating batch by one image:")

            overlapping_batch_indicies = [self.chunk_indicies[i] for i in range(0, len(self.chunk_indicies)) if (self.chunk_indicies[i] in self.previous_indicies)]
            #Select the relevant samples which are already read in
            self.batch_bandA = self.batch_bandA[overlapping_batch_indicies[0]: overlapping_batch_indicies[-1] + 1]
            self.batch_bandB = self.batch_bandB[overlapping_batch_indicies[0]: overlapping_batch_indicies[-1] + 1]
            #Then read the additional image
            self.batch_bandA.append(cv2.imread(self.image_directory + "/" + self.bandA_names[self.chunk_indicies[-1]], -1))
            self.batch_bandB.append(cv2.imread(self.image_directory + "/" + self.bandB_names[self.chunk_indicies[-1]], -1))

        if b == 0:
            #Correct all the samples
        else:
            #Correct just the last sample
        #TODO correct the samples (all of the first batch, then just the additional one read in each step).

    def correct_sample(self, sample_index):
        '''Apply corrections to the image pair with this index (in the whole sequence).'''
        pass


