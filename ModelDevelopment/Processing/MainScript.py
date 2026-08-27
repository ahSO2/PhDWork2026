import matplotlib.pyplot as plt

from BackgroundMethods import *
from DataClasses import *
import numpy as np


Reventador_2022_dictionary = {"volcano_dictionary_name":"Reventador2022",
                              "volcano_name":"Reventador",
                              "sig_figs_for_dark_ss":2,
                              'dark_path_A': "C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/Darks/Reventador_Band_A",
                              'dark_path_B': "C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/Darks/Reventador_Band_B",
                              'clear_sky_path_A': "C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/SelectedClears/Reventador/2022-04-07T200435_fltrA_1ag_3499986ss_Plume_DarkCorrected.png",
                              'clear_sky_path_B': "C:/Users/ggp24ash/Documents/Main Datasets/fromSharedDrive/SelectedClears/Reventador/2022-04-07T200435_fltrB_1ag_400000ss_Plume_DarkCorrected.png",
                              'registration_points_A': np.float32([[377,438], [120,352], [186,274], [605,408]]),
                              'registration_points_B': np.float32([[372,450], [110,363], [177,283], [608,421]]),
                              'sensor_marks_mask_A': "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/SensorMarkMasks/Reventador_2022A.png",
                              'sensor_marks_mask_B': "None",
                              'flank_mask_path': "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/Reventador2022FlankMask.png"
                              }

geom_data = {'cfov_azim':50, #Azimuth angle of CFOV of camera #TODO Dummy values for writing functions
             'cfov_elev':60,  #Elevation angle of CFOV of camera (including tilt of camera during install/placement)
             'plume_dir_azim': None #Direction of the plume (based on wind direction). If None given, assumed to be perpendicular to camera plane if not given.
             }

#Read in the image names in order, then load, correct and process batches of
#X minutes at a time.
reventador_sequence = Sequence()
reventador_sequence.set_volcano_dictionary(Reventador_2022_dictionary)
reventador_sequence.read_and_match_full_sequence("E:/Reventador/2022/2022-04-24/Seq_2")
reventador_sequence.read_spectrometer_data("E:/Reventador/2022/2022-04-24/Seq_2/Processed_spec_2026-08-19T160055/doas_results_2022-04-24T170000.csv")
#Apply quality models for the whole sequence, and save the predictions to two arrays.
#This is run by loading batches of images at a time, to avoid overwhelming the memory.
#reventador_sequence.apply_quality_models(chunk_size=20)
#print(reventador_sequence.all_precip_predictions)
#print(reventador_sequence.all_cloud_predictions)

#Next, iterate over each image, selecting the set of images to load (based on which data is to be used to calibrate that sample)

#TODO iterate over each image, determining the data to load based on
#TODO the surrounding samples needed for the calibration for that image
for i in range(0, len(reventador_sequence.bandA_names)):
    batchBandA, batchBandB = reventador_sequence.iterate(b=i, method="basic")

#TODO Update the backgrounds, and calculate the absorbance for the new chunk (copying over any that were already calculated in the previous)

#    reventador_sequence.estimate_backgrounds(method=constant_ratio_assumption, b=i_iter)
#    reventador_sequence.calculate_absorbance(b=i_iter)
#    reventador_sequence.find_spectrometer_FOV(s=10, plot=True)
#    reventador_sequence.calculate_calibration_curve()
#    reventador_sequence.calibrate_AA()
#    reventador_sequence.convert_molecules_to_mass()
#    i_iter += 1



