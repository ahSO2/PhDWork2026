import cv2
import matplotlib.pyplot as plt
import numpy as np

def mask_sensor_marks(image, mask_path, paths_dictionary):
    if mask_path == "None":
        pass
    else:
        mask = cv2.imread(paths_dictionary["sensor_mark_masks_path"] + "/" + mask_path, -1)
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return image
def read_sample(sample_index, df, timesteps, band, volcano_dictionary, paths_dictionary):
    # Create a sequence of timestep images
    sequence = []
    names = []
    batch = df["labelling_batch_name"][sample_index]
    smmn_A = volcano_dictionary["sensor_marks_mask_A"]
    smmn_B = volcano_dictionary["sensor_marks_mask_B"]
    all_plume_mask = None

    for timestep_name in timesteps:
        if timestep_name == "image_name":
            folder_to_read = paths_dictionary["data_path"]
        else:
            folder_to_read = paths_dictionary["data_path_temporal"]
        if band == "B":
            timestep_name = timestep_name + "_B"
        name_to_read = df[timestep_name][sample_index]
        timestep_image = cv2.imread(folder_to_read + "/" + name_to_read, -1)

        if "fltrB" in name_to_read:
            timestep_image = mask_sensor_marks(timestep_image, smmn_B, paths_dictionary)
        else:
            timestep_image = mask_sensor_marks(timestep_image, smmn_A, paths_dictionary)

        sequence.append(timestep_image)
        names.append(name_to_read)

        if "image_name" in timestep_name and band == "A":
            print(name_to_read)
            mask_path = paths_dictionary["segmentation_masks_path"] + batch + "/PlumeAndExpPixels_" + name_to_read.split(".")[0] + ".npy"
            # print("Reading plume mask from: " + mask_path)
            two_channel_mask = np.load(mask_path)  # Manually drawn plume mask (value 1 indicates plume)
            all_plume_mask = two_channel_mask[0, :, :] + two_channel_mask[1, :, :]
            all_plume_mask = np.where(all_plume_mask > 0, 1, 0)

        flank_mask = cv2.imread(volcano_dictionary["flank_mask_path"], -1)

    return sequence, names, all_plume_mask, flank_mask