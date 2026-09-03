'''Each background method should intake:
BandA and B image pair (float datatype).
Flank mask.

And output:
Estimated background images (band A and B)
'''
import numpy as np
import matplotlib.pyplot as plt

def constant_ratio_assumption(bandA, bandB, flank_mask):
    return np.ones_like(bandA), np.ones_like(bandB)

def return_image_TEST_FN(bandA, bandB, flank_mask):
    return bandA, bandB


################# Post-AA-calc Corrections #################
def zero_flank(AA, flank_mask):
    '''Subtract the mean value over the flank region from the whole image.'''
    relevant_pixels = np.ma.masked_where((flank_mask==1) & (AA.mask==False), AA)
    relevant_pixels = relevant_pixels.compressed()
    mean_value = np.ma.sum(relevant_pixels)/relevant_pixels.shape[0]
    return AA - mean_value

