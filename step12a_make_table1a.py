from STAR_outliers_library import compute_w
from STAR_outliers_library import estimate_tukey_params
import numpy as np
import pandas as pd
import torch
import pdb
import os
from matplotlib import pyplot as plt
# these models take too long or are too RAM intensive
# from pyod.models.knn import SOS
# from pyod.models.lmdd import LMDD
# from pyod.models.loci import LOCI
# from pyod.models.mo_gaal import MO_GAAL
# from pyod.models.sod import SOD
# from pyod.models.rod import ROD

# Don't want model ensemble 
# from pyod.models.lscp import LSCP
from tqdm import tqdm
from pyod.models.ecod import ECOD 
from pyod.models.copod import COPOD 
from pyod.models.abod import ABOD
from pyod.models.kde import KDE
from pyod.models.sampling import Sampling
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.inne import INNE
from pyod.models.feature_bagging import FeatureBagging as FB
from pyod.models.hbos import HBOS 
from pyod.models.suod import SUOD
from pyod.models.loda import LODA
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from pyod.models.deep_svdd import DeepSVDD
from sklearn.ensemble import IsolationForest as IF

np.random.seed(0)
torch.manual_seed(0)
prefix = "multivariate_checker_output_normal"

N, P = 500000, 20
VAE = VAE(encoder_neurons = [P, int(P/2), int(P/4)], decoder_neurons = [(P/4), int(P/2), P])
CBLOF = CBLOF(n_clusters=10)
AutoEncoder = AutoEncoder(hidden_neurons = [P, int(P/2), int(P/2), int(P)])
models = [ECOD(), COPOD(), KDE(), Sampling(), PCA(), MCD(), OCSVM(), LOF(), COF()]
models += [CBLOF, HBOS(), KNN(), ABOD(), LODA(), SUOD(), VAE, SO_GAAL(), DeepSVDD()]
models += [INNE(), FB(), AutoEncoder]

model_names = ["ECOD", "COPOD", "KDE", "Sampling", "PCA", "MCD", "OCSVM", "LOF", "COF"]
model_names += ["CBLOF", "HBOS", "KNN", "ABOD", "LODA", "SUOD", "VAE", "SO_GAAL", "DeepSVDD"]
model_names += ["INNE", "FB", "AutoEncoder"]
values = [model_names]

if not os.path.isdir(prefix):
    os.mkdir(prefix)

fits = []

X_vec = []
outlier_indices_vec = []
for j in range(10):
    X = np.random.normal(0, 1, N)
    # f is fraction of outliers
    f = 0.01
    outlier_indices = np.random.choice(np.arange(N), int(f*N/P), replace = False)
    outliers = np.random.uniform(np.max(np.abs(X)) + 0.5, np.max(np.abs(X)) + 1.5, int(f*N/P))
    outliers *= np.random.choice([-1, 1], int(f*N/P), replace = True)
    X[outlier_indices] = outliers
    X_vec.append(X.reshape(-1, P))
    outlier_indices_vec.append(outlier_indices)

for m, clf in enumerate(models):

    filename = prefix + "/model" + str(m) + "_outlier_predictions.txt"
    fit = []
    preds = []
    if os.path.isfile(filename):
        preds_df = pd.read_csv(filename, delimiter = "\t", header = None)
    for i in tqdm(range(10)):
            
        X = X_vec[i]
        outlier_indices = outlier_indices_vec[i]
        if not os.path.isfile(filename):
            clf.fit(X)
            outlier_scores = clf.decision_scores_
        else:
            outlier_scores = preds_df[i].to_numpy()

        is_outlier = np.isin(np.arange(N), outlier_indices)
        outlier_row_inds_real = np.where(np.any(is_outlier.reshape(-1, 20), axis = 1))[0]
        num_outliers = len(outlier_row_inds_real)
        outlier_row_inds_pred = np.argsort(outlier_scores)[-num_outliers:]
        recall_num = len(np.intersect1d(outlier_row_inds_real, outlier_row_inds_pred))
        recall = recall_num/num_outliers
        fit.append(recall)
        preds.append(outlier_scores)
    if not os.path.isfile(filename):
        preds_df = pd.DataFrame(np.array(preds).T)
        preds_df.to_csv(filename, sep = "\t", header = False, index = False)
    print(np.mean(fit))
    fits.append(fit)

values.append(np.mean(fits, axis = 1))

table1 = pd.DataFrame(np.array(values).T)
table1.columns = ["model name", "normal TPR"]
table1.to_csv("table1_normal.txt", sep = "\t", header = True, index = False)

