import cv2
from datetime import datetime
import os

def map_image_name_to_time(image_name):
    date_str = image_name.split("_")[0][0:10]
    time_str = image_name.split("_")[0][11:17]
    datetime_obj = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]), int(time_str[:2]), int(time_str[2:4]), int(time_str[4:]))
    return datetime_obj
class Sequence():
    '''Class representing a sequence of data to be processed.'''
    def __init__(self):
        self.volc_dict = None


    def set_volcano_dictionary(self, dictionary):
        self.volc_dict = dictionary

    def read_and_match(self, directory_path, correct=True):
        '''Read all samples from the given directory. Procuce a list
        of band A samples, matched to corresponding bandB samples.
        If correct is True, dark and vignette correct, then register
        the samples, using the data from the volcano dictionary.'''

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
        print("Matching band A/B pairs found: " + str(len(self.bandA_names)))

        self.correct = correct
        if correct == True:
            #If we do want to correct the data as it is read in, load the information to do so here.
            self.dark_names_A = []
            self.dark_names_B = []

            self.dark_sig_figs =

            self.clear_sky_image_A = []
            self.clear_sky_image_B = []


    def iterate(self, n, chunk_size_m=30):
        '''Return the first 30mins (or specified time period) of the sequence, then
        incrementally return images in range [15mins + dt_i, 30mins + dt_i].
        Where dt_i denotes the timestep difference between the last image
        returned and the next sample.'''

        #Decide which samples to load
        #For i = 0, load the first 30mins of data
        self.chunk_indexes = #Select the indexes of the images we want to be in this iteration
        self.active_indexes = #Indexes which were in the previous chunk


        #Correct them (dark, clear and registration)

    def correct_sample(self, sample_index):
        '''Apply corrections to the image pair with this index (in the whole sequence).'''



