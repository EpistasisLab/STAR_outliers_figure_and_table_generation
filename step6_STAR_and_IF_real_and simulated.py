import numpy as np
import pandas as pd
from copy import deepcopy as COPY
from sklearn.ensemble import IsolationForest as IF
from matplotlib import pyplot as plt
from scipy.stats import yeojohnson as yj
import pdb

#----------------------------------------------------------------------------
# Code for the simulated dataset
#----------------------------------------------------------------------------

true_outliers = pd.read_csv("fake_data_true_outliers.txt", delimiter = "\t").to_numpy().T
pred_outliers = pd.read_csv("fake_data_cleaned_data.txt", delimiter = "\t").to_numpy().T
pred_outliers = np.isnan(pred_outliers)

fake_data = pd.read_csv("fake_data.txt", delimiter = "\t").to_numpy().T
dataset_names, TPR, FPR, TPR2, FPR2 = np.arange(len(fake_data)), [], [], [], []
precision, precision2 = [], []
for inds, inds_STAR, x in zip(true_outliers, pred_outliers, fake_data):

    iforest = IF().fit(x.reshape(-1, 1))
    inds_pred = iforest.predict(x.reshape(-1, 1)) == -1
    TPR.append(np.sum(inds_pred[inds == True])/np.sum(inds))
    FPR.append(np.sum(inds_pred[inds == False])/np.sum(inds == False))
    if np.sum(inds_pred) == 0:
        precision.append(1)
    else:
        precision.append(np.sum(inds_pred[inds == True])/np.sum(inds_pred))


    iforest_scores = iforest._compute_score_samples(x.reshape(-1,1), subsample_features = False)
    num_guesses = np.sum(inds_STAR)
    top_guesses = np.argsort(iforest_scores)[-num_guesses:]
    TPR2.append(np.sum(inds[top_guesses])/np.sum(inds))
    FPR2.append(np.sum(inds[top_guesses] == False)/np.sum(inds == False))
    if np.sum(inds_pred) == 0:
        precision.append(1)
    else:
        precision2.append(np.sum(inds[top_guesses])/len(top_guesses))

df = pd.DataFrame(np.array([dataset_names, TPR, FPR, precision]).T)
df.columns = ["simulated dataset", "TPR", "FPR", "precision"]
df.to_csv("simulated_efficacies/Information_Forest_efficacy.txt", sep = "\t", header = True, index = False)

df = pd.DataFrame(np.array([dataset_names, TPR2, FPR2, precision2]).T)
df.columns = ["simulated dataset", "TPR", "FPR", "precision"]
df.to_csv("simulated_efficacies/Information_Forest_adjusted_efficacy.txt", sep = "\t", header = True, index = False)
    
dataset_names, TPR, FPR, precision = np.arange(len(true_outliers)), [], [], []
for inds, inds_pred in zip(true_outliers, pred_outliers):
    TPR.append(np.sum(inds_pred[inds == True])/np.sum(inds))
    FPR.append(np.sum(inds_pred[inds == False])/np.sum(inds == False))
    if np.sum(inds_pred) == 0:
        precision.append(1)
    else:
        precision.append(np.sum(inds_pred[inds == True])/np.sum(inds_pred))

df = pd.DataFrame(np.array([dataset_names, TPR, FPR, precision]).T)
df.columns = ["simulated dataset", "TPR", "FPR", "precision"]
df.to_csv("simulated_efficacies/STAR_efficacy.txt", sep = "\t", header = True, index = False)

#----------------------------------------------------------------------------
# Code for the real dataset
#----------------------------------------------------------------------------

real_data_df = pd.read_csv("step0_all_2018_processed.txt", delimiter = "\t")
real_data = real_data_df.to_numpy().T
dataset_names = (real_data_df.columns).to_numpy()
used_names, removed_IF = [], []
for name, x in zip(dataset_names, real_data):

    x = x[np.isnan(x) == False]
    if len(np.unique(x)) >= 10:
        used_names.append(name)
        iforest = IF().fit(x.reshape(-1, 1))
        inds_pred = iforest.predict(x.reshape(-1, 1)) == -1
        removed_IF.append(np.sum(inds_pred)/len(x))

df = pd.DataFrame(np.array([used_names, removed_IF]).T)
df.columns = ["dataset", "fraction removed"]
df.to_csv("real_efficacies/IF_real_efficacy.txt", sep = "\t", header = True, index = False)

fname = "step0_all_2018_processed_outlier_info.txt"
STAR_efficacy = pd.read_csv(fname, delimiter = "\t")["percent_inliers"].to_numpy()
df = pd.DataFrame(np.array([used_names, 1 - STAR_efficacy]).T)
df.columns = ["dataset", "fraction removed"]
df.to_csv("real_efficacies/STAR_real_efficacy.txt", sep = "\t", header = True, index = False)

