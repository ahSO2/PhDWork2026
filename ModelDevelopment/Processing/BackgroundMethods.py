'''Each background method should intake:
BandA and B image pair (float datatype).
Flank mask.

And output:
Estimated background images (band A and B)
'''
import numpy as np

def constant_ratio_assumption(bandA, bandB, flank_mask):
    return np.ones_like(bandA), np.ones_like(bandB)

def return_image_TEST_FN(bandA, bandB, flank_mask):
    return bandA, bandB
