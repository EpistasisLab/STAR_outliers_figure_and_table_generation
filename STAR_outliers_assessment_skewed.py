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

true_outliers = pd.read_csv("skewed_data_outlier_indices.txt", delimiter = "\t").to_numpy().T
pred_outliers = pd.read_csv("skewed_data_cleaned_data.txt", delimiter = "\t").to_numpy().T
pred_outliers = np.isnan(pred_outliers)

fake_data = pd.read_csv("skewed_data.txt", delimiter = "\t").to_numpy().T
dataset_names, TPR, FPR, TPR2, FPR2 = np.arange(len(fake_data)), [], [], [], []
for inds, inds_STAR, x in zip(true_outliers, pred_outliers, fake_data):

    iforest = IF().fit(x.reshape(-1, 1))
    inds_pred = iforest.predict(x.reshape(-1, 1)) == -1
    TPR.append(np.sum(inds_pred[inds == True])/np.sum(inds))
    FPR.append(np.sum(inds_pred[inds == False])/np.sum(inds == False))

    iforest_scores = iforest._compute_score_samples(x.reshape(-1,1), subsample_features = False)
    num_guesses = np.sum(inds_STAR)
    top_guesses = np.argsort(iforest_scores)[-num_guesses:]
    TPR2.append(np.sum(inds[top_guesses])/np.sum(inds))
    FPR2.append(np.sum(inds[top_guesses] == False)/np.sum(inds == False))

df = pd.DataFrame(np.array([dataset_names, TPR, FPR]).T)
df.columns = ["simulated dataset", "TPR", "FPR"]
df.to_csv("Information_Forest_efficacy_skewed.txt", sep = "\t", header = True, index = False)

df = pd.DataFrame(np.array([dataset_names, TPR2, FPR2]).T)
df.columns = ["simulated dataset", "TPR", "FPR"]
df.to_csv("Information_Forest_efficacy2_skewed.txt", sep = "\t", header = True, index = False)
    
dataset_names, TPR, FPR = np.arange(len(true_outliers)), [], []
for inds, inds_pred in zip(true_outliers, pred_outliers):
    TPR.append(np.sum(inds_pred[inds == True])/np.sum(inds))
    FPR.append(np.sum(inds_pred[inds == False])/np.sum(inds == False))

df = pd.DataFrame(np.array([dataset_names, TPR, FPR]).T)
df.columns = ["simulated dataset", "TPR", "FPR"]
df.to_csv("STAR_efficacy_skewed.txt", sep = "\t", header = True, index = False)

