import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def mean_with_95p_bootstrap(array, rng):
    '''Intakes and passes back a random number generator object (which can be initialised
    outwith this function with a seed). '''
    mean = np.nanmean(array)

    bootstrap_means = []

    # Bootstrap CI
    for b in range(0, 1000):
        selection = rng.choice(array, array.shape[0], replace=True)
        bootstrap_means.append(np.mean(selection))

    lower = np.percentile(bootstrap_means, q=2.5)
    upper = np.percentile(bootstrap_means, q=97.5)
    return np.round(mean, 4), np.round(lower, 4), np.round(upper, 4), rng

def location_balanced_mean(array, locations):

    values_df = pd.DataFrame()
    values_df["value"] = array
    values_df["location"] = locations

    location_means = []
    for location in set(locations):
        location_samples = values_df[values_df["location"] == location]
        location_means.append(np.nanmean(location_samples["value"]))
    return np.mean(location_means)


def location_balanced_mean_with_95p_bootstrap(values, locations, rng):

    balanced_mean = location_balanced_mean(values, locations)

    bootstrap_balanced_means = []

    values_df = pd.DataFrame()
    values_df["value"] = values
    values_df["location"] = locations

    # Bootstrap CI
    for b in range(0, 1000):
        sample = values_df.sample(n=values_df.shape[0], replace=True, random_state=rng)
        bootstrap_balanced_means.append(location_balanced_mean(sample["value"], sample["location"]))
    lower = np.percentile(bootstrap_balanced_means, q=2.5)
    upper = np.percentile(bootstrap_balanced_means, q=97.5)

    return np.round(balanced_mean, 4), np.round(lower, 4), np.round(upper, 4), rng



