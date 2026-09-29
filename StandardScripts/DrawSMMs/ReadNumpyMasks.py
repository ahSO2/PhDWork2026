#Script to read in an exported set of draw masks from label studio and
#match them to the correct reerence image.
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

ref_folder = "RefImgs_Batch1_10bit/"
labels_folder = "RawMasks_Batch1/"
save_folder = "SMMs_Batch1/"

starting_index = 254

#############################################################
index = starting_index
all_masks_list = os.listdir(labels_folder)
class str_contains_fn():
    def __init__(self, string):
        self.string = string

    def str_contains(self, input):
        result = self.string in input
        return result

for image_name in os.listdir(ref_folder):
    print("For " + str(image_name))
    print("Relevant masks: ")
    #Read any masks containing "task-index"
    fn_obj = str_contains_fn("task-" + str(index))
    rel_masks = list(filter(fn_obj.str_contains, all_masks_list))

    mask = np.zeros((486, 648))
    for rel_mask in rel_masks:
        this_mask = np.load(labels_folder + rel_mask)
        mask = mask + this_mask

    mask = np.where(mask > 0, 1, 0)
    #Save the mask
    cv2.imwrite(save_folder + image_name, mask)

    index += 1

    plt.imshow(mask)
    plt.colorbar()
    plt.show()