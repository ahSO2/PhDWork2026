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
#Read in the image names in order, then load, correct and process batches of
#X minutes at a time.
reventador_sequence = Sequence()
reventador_sequence.set_volcano_dictionary(Reventador_2022_dictionary)
reventador_sequence.read_and_match("E:/Reventador/2022/2022-04-24/Seq_2")
reventador_sequence.read_spectrometer_data("E:/Reventador/2022/2022-04-24/Seq_2/Processed_spec_2026-08-19T160055/doas_results_2022-04-24T170000.csv")
mins = 5
i_iter = 0 #Batch index
rem=1
while rem > 0: #While the remaining number of batch iterations is greater than zero
    batchBandA, batchBandB, rem = reventador_sequence.iterate(b=i_iter, chunk_size_m=mins)
    print(i_iter)
    print(reventador_sequence.batch_start_time)
    print(reventador_sequence.batch_end_time)
    print(reventador_sequence.chunk_indicies)
    reventador_sequence.estimate_backgrounds(method=constant_ratio_assumption, b=i_iter)
    reventador_sequence.calculate_absorbance(b=i_iter)
    reventador_sequence.find_spectrometer_FOV(s=10, plot=True)
    reventador_sequence.calculate_calibration_curve()
    reventador_sequence.calibrate_AA()
    reventador_sequence.convert_molecules_to_mass()
    i_iter += 1



