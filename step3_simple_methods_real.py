import numpy as np
import pandas as pd
from copy import deepcopy as COPY
from sklearn.ensemble import IsolationForest as IF
from matplotlib import pyplot as plt
from scipy.stats import yeojohnson as yj
import os
import pdb

from recreate_tables_9_10_11_library import asd_cutoff
from recreate_tables_9_10_11_library import mad_cutoff
from recreate_tables_9_10_11_library import iqr_cutoff
from recreate_tables_9_10_11_library import T2_thresh_data

real_data_df = pd.read_csv("step0_all_2018_processed.txt", delimiter = "\t")
dataset_names = (real_data_df.columns).to_numpy()
real_data = real_data_df.to_numpy().T
real_data_yj = np.zeros(real_data.shape)
for i in range(len(real_data)):
    real_data_yj[i] = yj(real_data[i])[0]

def sd3(x): return(asd_cutoff(x, 3))
outlier_inds_pred_IQR_yj = [iqr_cutoff(x[np.isnan(x) == False]) == False for x in real_data_yj]
outlier_inds_pred_IQR = [iqr_cutoff(x[np.isnan(x) == False]) == False for x in real_data]
outlier_inds_pred_ASD_yj = [asd_cutoff(x[np.isnan(x) == False], 3) == False for x in real_data_yj]
outlier_inds_pred_ASD = [asd_cutoff(x[np.isnan(x) == False], 3) == False for x in real_data]
outlier_inds_pred_MAD_yj = [mad_cutoff(x[np.isnan(x) == False], 3) == False for x in real_data_yj]
outlier_inds_pred_MAD = [mad_cutoff(x[np.isnan(x) == False], 3) == False for x in real_data]
outlier_inds_pred_T2_yj = [T2_thresh_data(x[np.isnan(x) == False], 2, sd3) for x in real_data_yj]
outlier_inds_pred_T2 = [T2_thresh_data(x[np.isnan(x) == False], 2, sd3) for x in real_data]
outlier_inds_pred_T3_yj = [T2_thresh_data(x[np.isnan(x) == False], 3, sd3) for x in real_data_yj]
outlier_inds_pred_T3 = [T2_thresh_data(x[np.isnan(x) == False], 3, sd3) for x in real_data]
outlier_inds_pred_T4_yj = [T2_thresh_data(x[np.isnan(x) == False], 4, sd3) for x in real_data_yj]
outlier_inds_pred_T4 = [T2_thresh_data(x[np.isnan(x) == False], 4, sd3) for x in real_data]

names = ["IQR", "IQR_yj", "ASD", "ASD_yj", "MAD", "MAD_yj"]
names += ["T2", "T2_yj", "T3", "T3_yj", "T4", "T4_yj"]
inds = [outlier_inds_pred_IQR, outlier_inds_pred_IQR_yj]
inds += [outlier_inds_pred_ASD, outlier_inds_pred_ASD_yj]
inds += [outlier_inds_pred_MAD, outlier_inds_pred_MAD_yj]
inds += [outlier_inds_pred_T2, outlier_inds_pred_T2_yj]
inds += [outlier_inds_pred_T3, outlier_inds_pred_T3_yj]
inds += [outlier_inds_pred_T4, outlier_inds_pred_T4_yj]

if not os.path.exists("real_efficacies"):
    os.mkdir("real_efficacies")
for name, outlier_inds_pred in zip(names, inds):

    dataset_names = (real_data_df.columns).to_numpy()
    used_names, fraction_removed = [], []
    for col_name, inds, x in zip(dataset_names, outlier_inds_pred, real_data):

        if len(np.unique(x)) >= 10:
            used_names.append(col_name)
            fraction_removed.append(np.sum(inds)/len(inds))

    df = pd.DataFrame(np.array([used_names, fraction_removed]).T)
    df.columns = ["dataset", "fraction removed"]
    df.to_csv("real_efficacies/" + name + "_real_efficacies.txt", sep = "\t", header = True, index = False)
