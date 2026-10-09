import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

def show(image, colormap="gray", title="None"):
    plt.imshow(image, cmap=colormap)
    if "None" in title:
        pass
    else:
        plt.title(title)
    plt.colorbar()
    plt.show()

def calculate_conf_counts(predicted_mask, true_mask, exclude):
    '''The positive (clear-sky) pixels in each mask should be indicated by 1s.
    Ommit any pixels covered by 1s in the "exclude" mask from the counts.'''

    true_p_mask = np.where(true_mask == 1, predicted_mask, 0)
    true_p_mask = np.where(exclude == 1, 0, true_p_mask)
    TP = np.sum(true_p_mask)

    predicted_negative = np.where(predicted_mask == 0, 1, 0)
    true_n_mask = np.where(true_mask == 0, predicted_negative, 0)
    true_n_mask = np.where(exclude == 1, 0, true_n_mask)
    TN = np.sum(true_n_mask)

    false_p_mask = np.where(true_mask == 0, predicted_mask, 0)
    false_p_mask = np.where(exclude == 1, 0, false_p_mask)
    FP = np.sum(false_p_mask)

    false_n_mask = np.where(predicted_mask == 0, true_mask, 0)
    false_n_mask = np.where(exclude == 1, 0, false_n_mask)
    FN = np.sum(false_n_mask)

    total = true_n_mask + 2*false_n_mask + 3*false_p_mask + 4*true_p_mask
    show(total)

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
    '''Calculate the IOU, returning 1-(FP/total_pixels) if the true mask is exactly zero.'''
    if TP + FN == 0: #If the ground truth mask is zeroes, take the proportion of the negative class which is false-positive
        IOU = 1 - (FP/(FP + TN))
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
        bootstrap_means.append(np.nanmean(selection))

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
    return np.nanmean(location_means)


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


def variance_ratio(bandA, bandB, selected_ref_area, manual_ref_area, flank_mask):
    '''Calculate the rough absorbance, then take the ratio of its standard deviation in the selected
    reference area over the ground truth reference area.'''

    #Mask out where bandB = 0
    inv_flank_mask = np.where(flank_mask == 0, 1, 0)
    bandB_zero_mask = np.where(bandB == 0, 5, 0)
    edge_mask = cv2.blur(bandB_zero_mask, (5, 5))
    edge_mask[:, 0:6] = 1
    edge_mask[:, -5:] = 1
    edge_mask[0:6, :] = 1
    edge_mask[-5:, :] = 1
    edge_mask = edge_mask + inv_flank_mask

    bandA_copy = bandA.copy()
    bandA_copy = np.ma.masked_where(edge_mask>0, bandA_copy)

    # Then calculate the absorbance (without accounting for backgrounds)
    ratio = np.ma.divide(bandA_copy.astype(np.float32), bandB.astype(np.float32))
    ratio = -1 * np.ma.log(ratio)

    selected_values = np.ma.masked_where(selected_ref_area == 0, ratio)
    selected_values.mask = np.where(edge_mask > 0, 1, selected_values.mask)
    selected_values = selected_values.compressed()

    gt_values = np.ma.masked_where(manual_ref_area == 0, ratio)
    gt_values.mask = np.where(edge_mask > 0, 1, gt_values.mask)
    gt_values = gt_values.compressed()

    if selected_values.shape[0] > 0:
        selected_sd = np.std(selected_values)
    else:
        selected_sd = 0
    if gt_values.shape[0] > 0:
        gt_sd = np.std(gt_values)
    else:
        gt_sd = 0

    if gt_sd == 0: #If the ground truth reference area is constant or doesn't have any area
        return 1
    else:
        return min(selected_sd/gt_sd, 1) #Return the proportion of the standard deviation that is represented (1 if the ref area gives an accurate or an overestimation of the background veriability)

def AUC_PR(predictions, ground_truth, exclude, plot=False):
    '''Mask out any pixels which are already masked, plus any indicated with a
    1 in "exclude".'''

    #TODO Mask out pixels in exclude mask
    gt = np.ma.masked_where(exclude, ground_truth)
    pr = np.ma.masked_where(exclude, predictions)

    precisions, recalls, thresholds = precision_recall_curve(y_true=gt.compressed(), y_score=pr.compressed())
    auc_val = auc(recalls, precisions)

    if plot == True:
        cm = 1 / 2.54  # centimeters in inches
        fig, ax = plt.subplots(figsize=(18 * cm, 18 * cm))
        ax.plot(recalls, precisions)
        ax.set_xlabel("Recall", fontsize=14)
        ax.set_ylabel("Precision", fontsize=14)
        plt.show()

    return auc_val