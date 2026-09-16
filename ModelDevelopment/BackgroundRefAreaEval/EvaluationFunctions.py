import matplotlib.pyplot as plt
import numpy as np
def calculate_conf_counts(predicted_mask, true_mask):
    '''The positive (clear-sky) pixels in each mask should be indicated by 1s.'''

    true_p_mask = np.where(true_mask == 1, predicted_mask, 0)
    TP = np.sum(true_p_mask)

    predicted_negative = np.where(predicted_mask == 0, 1, 0)
    true_n_mask = np.where(true_mask == 0, predicted_negative, 0)
    TN = np.sum(true_n_mask)

    false_p_mask = np.where(true_mask == 0, predicted_mask, 0)
    FP = np.sum(false_p_mask)

    false_n_mask = np.where(predicted_mask == 0, true_mask, 0)
    FN = np.sum(false_n_mask)

    return TP, TN, FP, FN

def per_class_precision(TP, TN, FP, FN):
    '''Calculating the precision of positive and negative predictions,
    giving a value of 1 if there are no predictions of that class.'''

    #Precision of prediction of the positive class
    if TP + FP == 0:
        PP = 1
    else:
        PP = TP/(TP + FP)

    #Precision of the negative class
    if TN + FN == 0:
        PN = 1
    else:
        PN = TN/(TN + FN)
    return PP, PN

def per_class_recall(TP, TN, FP, FN):
    '''Calculate the recall of truly positive and negative pixels,
    returning np.nan if there are no pixels of that class.'''

    #Recall of truly positive samples
    if TP + FN == 0:
        RP = np.nan #There are no positive samples to recall
    else:
        RP = TP/(TP + FN)

    #Recall of truly negative samples
    if TN + FP == 0:
        RN = np.nan #There are no negative samples to recall
    else:
        RN = TN/(TN + FP)
    return RP, RN

def intersection_over_union(TP, TN, FP, FN):
    '''Calculate the IOU, returning 1 if both masks are exactly zero.'''
    if TP + FP + FN == 0:
        IOU = 1 #If both masks are zero area, then the prediction is exactly correct
    else:
        IOU = TP/(TP + FP + FN)
    return IOU

def F1_score(P, R):
    if P + R == 0:
        F1 = 0
    else:
        F1 = (2 * P * R)/(P + R)

    return F1




