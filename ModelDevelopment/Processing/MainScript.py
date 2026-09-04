import matplotlib.pyplot as plt
from BackgroundMethods import *
from DataClasses import *
import matplotlib.pyplot as plt
import numpy as np

############################## Setup variables ####################################
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
                              'flank_mask_path': "C:/Users/ggp24ash/Documents/Main Datasets/PlumeSegmentation/FlankMasks/Reventador2022FlankMask.png",
                              'cam_lat': -0.073421,
                              'cam_lon': -77.617984,
                              'cam_height': None,
                              'ref_lat':-0.0804836, #Coordinates of a reference point in the image used to find the camera angle (here I use the edge of the crater)
                              'ref_lon':-77.6571622,
                              'ref_pixel_coords':(355, 167),#TODO Maybe select a bit more precisely
                              'crater_lat':-0.0806186, #Coordinates of the center of the crater (used to define the distance of the image plane from the camera)
                              'crater_lon':-77.6578386}

camera_dictionary = {'pixels':(486, 648),
                     'FOV_angle':(21, 28)}

geom_data = {'plume_dir_azim': None #Direction of the plume (based on wind direction). If None given, assumed to be perpendicular to camera plane if not given.
             }

background_method = constant_ratio_assumption
spectrometer_CFOV = (55, 305)

######################## Processing ########################################

reventador_sequence = Sequence()
reventador_sequence.img_shape = camera_dictionary['pixels']
reventador_sequence.set_volcano_dictionary(Reventador_2022_dictionary)
reventador_sequence.initialise_and_match_full_sequence("E:/Reventador/2022/2022-04-24/Seq_2")

cam_geom = CameraGeometry(Reventador_2022_dictionary, camera_dictionary)
cam_geom.calculate_camera_angle()
cam_geom.calculate_CFOV_location()
cam_geom.plot_camera_geometry()
cam_geom.calculate_pixel_sizes()

#Apply quality models for the whole sequence, and save the predictions to two arrays.
#This is run by loading batches of images at a time, to avoid overwhelming the memory.
reventador_sequence.requested_timesteps = [10, -10, 60, -60] #Define timesteps to be used when reading temporal data
reventador_sequence.match_timestep_images()
#reventador_sequence.apply_quality_models(chunk_size=20)
#print(reventador_sequence.all_precip_predictions)
#print(reventador_sequence.all_cloud_predictions)

#Identify the spectrometer FOV, based on selected good quality images
reventador_sequence.read_spectrometer_data("E:/Reventador/2022/2022-04-24/Seq_2/Processed_spec_2026-08-19T160055/doas_results_2022-04-24T170000.csv")
if spectrometer_CFOV == None:
    reventador_sequence.select_and_read_indexes_for_spectrometer_FOV_match(indexes=[8, 17, 18, 25, 26, 32, 33, 38, 48, 50, 51, 55, 59, 67, 69, 71, 72, 77, 88, 96, 98, 109, 111, 113, 117, 133, 136, 147, 148, 150, 151, 152, 154, 156, 157, 159, 170, 171, 185, 186, 187, 188, 190, 192, 198, 205, 216, 226, 232, 234, 236, 246, 248, 250, 251, 254, 258, 259, 273, 276, 290, 292, 310, 313, 315, 316, 317, 346, 349, 365, 369, 370, 375, 378, 385, 388, 392, 396, 407, 408, 410, 414, 415, 425, 430, 436, 438, 446, 448, 449, 456, 458, 465, 467, 468, 471, 472, 473, 474, 476])
    #reventador_sequence.select_and_read_indexes_for_spectrometer_FOV_match(indexes=[8, 17, 18, 25])
    reventador_sequence.estimate_backgrounds(method=background_method)
    reventador_sequence.calculate_absorbance(correction=zero_flank)
    reventador_sequence.find_spectrometer_FOV(s=5, plot=True)
else: #Or set a known spectrometer FOV:
    reventador_sequence.spec_CFOV = spectrometer_CFOV


#Iterate over each image, determining the data to load based on
#the surrounding samples needed for the calibration for that image
for i in range(0, len(reventador_sequence.bandA_names)):
    print("Processing image: " + str(i) )
    reventador_sequence.iterate(b=i, method="basic", mins=10)
    #reventador_sequence.view_current_chunk(band="B", timesteps=[-10, 10])
    #Update the backgrounds, and calculate the absorbance for the new chunk (copying over any that were already calculated in the previous)
    reventador_sequence.estimate_backgrounds(method=background_method)
    reventador_sequence.calculate_absorbance(correction=zero_flank)
    reventador_sequence.calculate_calibration_curve(plot=True)
    reventador_sequence.calibrate_AA()
    reventador_sequence.convert_molecules_to_mass()
    if i in [1, len(reventador_sequence.bandA_names)-1]:
        for chunk_i in range(0, len(reventador_sequence.chunk_indexes)):
            img_A = reventador_sequence.batch_bandA[chunk_i]
            img_B = reventador_sequence.batch_bandB[chunk_i]
            bg_A = reventador_sequence.bgs_A[chunk_i]
            bg_B = reventador_sequence.bgs_B[chunk_i]
            AA = reventador_sequence.AA[chunk_i]

            fig, axs = plt.subplots(nrows=2, ncols=3)
            axs[0, 0].imshow(img_A, cmap="gray")
            axs[0, 1].imshow(img_B, cmap="gray")
            axs[0, 2].imshow(AA, cmap="YlGnBu_r", vmin=0)
            axs[1, 0].imshow(bg_A, cmap="gray")
            axs[1, 1].imshow(bg_B, cmap="gray")
            plt.show()



