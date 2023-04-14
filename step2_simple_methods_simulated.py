import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.special import erfinv
from scipy.stats import yeojohnson as yj
from copy import deepcopy as COPY
import pdb

from recreate_tables_9_10_11_library import asd_cutoff
from recreate_tables_9_10_11_library import mad_cutoff
from recreate_tables_9_10_11_library import iqr_cutoff
from recreate_tables_9_10_11_library import T2_thresh_data

fake_data = pd.read_csv("fake_data.txt", delimiter = "\t").to_numpy().T
true_outliers = pd.read_csv("fake_data_true_outliers.txt", delimiter = "\t").to_numpy().T
fake_data_yj = np.zeros(fake_data.shape)
for i in range(len(fake_data)):
    fake_data_yj[i] = yj(fake_data[i])[0]

def sd3(x): return(asd_cutoff(x, 3))
outlier_inds_pred_IQR_yj = (np.array([iqr_cutoff(x) for x in fake_data_yj]) == False)
outlier_inds_pred_IQR = (np.array([iqr_cutoff(x) for x in fake_data]) == False)
outlier_inds_pred_ASD_yj = (np.array([asd_cutoff(x, 3) for x in fake_data_yj]) == False)
outlier_inds_pred_ASD = (np.array([asd_cutoff(x, 3) for x in fake_data]) == False)
outlier_inds_pred_MAD_yj = (np.array([mad_cutoff(x, 3) for x in fake_data_yj]) == False)
outlier_inds_pred_MAD = (np.array([mad_cutoff(x, 3) for x in fake_data]) == False)
outlier_inds_pred_T2_yj = np.array([T2_thresh_data(x, 2, sd3) for x in fake_data_yj])
outlier_inds_pred_T2 = np.array([T2_thresh_data(x, 2, sd3) for x in fake_data])
outlier_inds_pred_T3_yj = np.array([T2_thresh_data(x, 3, sd3) for x in fake_data_yj])
outlier_inds_pred_T3 = np.array([T2_thresh_data(x, 3, sd3) for x in fake_data])
outlier_inds_pred_T4_yj = np.array([T2_thresh_data(x, 4, sd3) for x in fake_data_yj])
outlier_inds_pred_T4 = np.array([T2_thresh_data(x, 4, sd3) for x in fake_data])

names = ["IQR", "IQR_yj", "ASD", "ASD_yj", "MAD", "MAD_yj"]
names += ["T2", "T2_yj", "T3", "T3_yj", "T4", "T4_yj"]
inds = [outlier_inds_pred_IQR, outlier_inds_pred_IQR_yj]
inds += [outlier_inds_pred_ASD, outlier_inds_pred_ASD_yj]
inds += [outlier_inds_pred_MAD, outlier_inds_pred_MAD_yj]
inds += [outlier_inds_pred_T2, outlier_inds_pred_T2_yj]
inds += [outlier_inds_pred_T3, outlier_inds_pred_T3_yj]
inds += [outlier_inds_pred_T4, outlier_inds_pred_T4_yj]

if not os.path.exists("simulated_efficacies"):
    os.mkdir("simulated_efficacies")
for name, outlier_inds_pred in zip(names, inds):
    dataset_names, TPR, FPR, precision = np.arange(len(outlier_inds_pred_IQR_yj)), [], [], []
    for i in dataset_names:
        inds_pred, inds_real = outlier_inds_pred[i], true_outliers[i]
        TPR.append(np.sum(inds_pred[inds_real == True])/np.sum(inds_real))
        FPR.append(np.sum(inds_pred[inds_real == False])/np.sum(inds_real == False))
        if np.sum(inds_pred) == 0:
            precision.append(1)
        else:
            precision.append(np.sum(inds_pred[inds_real == True])/np.sum(inds_pred))

    df = pd.DataFrame(np.array([dataset_names, TPR, FPR, precision]).T)
    df.columns = ["simulated dataset", "TPR", "FPR", "precision"]
    df.to_csv("simulated_efficacies/" + name + "_efficacy.txt", sep = "\t", header = True, index = False)

