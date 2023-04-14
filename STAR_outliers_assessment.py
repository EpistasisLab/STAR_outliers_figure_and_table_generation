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
df.to_csv("Information_Forest_efficacy.txt", sep = "\t", header = True, index = False)

df = pd.DataFrame(np.array([dataset_names, TPR2, FPR2]).T)
df.columns = ["simulated dataset", "TPR", "FPR"]
df.to_csv("Information_Forest_efficacy2.txt", sep = "\t", header = True, index = False)
    
dataset_names, TPR, FPR = np.arange(len(true_outliers)), [], []
for inds, inds_pred in zip(true_outliers, pred_outliers):
    TPR.append(np.sum(inds_pred[inds == True])/np.sum(inds))
    FPR.append(np.sum(inds_pred[inds == False])/np.sum(inds == False))

df = pd.DataFrame(np.array([dataset_names, TPR, FPR]).T)
df.columns = ["simulated dataset", "TPR", "FPR"]
df.to_csv("STAR_efficacy.txt", sep = "\t", header = True, index = False)

#----------------------------------------------------------------------------
# Code for the real dataset
#----------------------------------------------------------------------------

def count_IQR_outliers(x, transform = False):
    if transform:
         x = yj(x)[0]
    q25, q75 = np.percentile(x, [25, 75])
    lb = (q25 - 1.5*(q75 - q25))
    ub = (q75 + 1.5*(q75 - q25))
    return((np.sum(x < lb) + np.sum(x > ub))/len(x))

real_data_df = pd.read_csv("all_2018_processed.txt", delimiter = "\t")
real_data = real_data_df.to_numpy().T
dataset_names = (real_data_df.columns).to_numpy()
used_names, removed_IF, removed_IQR, removed_IQR_yj = [], [], [], []
for name, x in zip(dataset_names, real_data):

    x = x[np.isnan(x) == False]
    if len(np.unique(x)) >= 10:
        used_names.append(name)
        iforest = IF().fit(x.reshape(-1, 1))
        inds_pred = iforest.predict(x.reshape(-1, 1)) == -1
        removed_IF.append(np.sum(inds_pred)/len(x))
        removed_IQR.append(count_IQR_outliers(COPY(x)))
        removed_IQR_yj.append(count_IQR_outliers(COPY(x), True))

df = pd.DataFrame(np.array([used_names, removed_IF, removed_IQR, removed_IQR_yj]).T)
df.columns = ["dataset", "IF", "IQR", "IQR_yj"]
df.to_csv("non_STAR_real_data_efficacies.txt", sep = "\t", header = True, index = False)
