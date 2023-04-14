import numpy as np
import pandas as pd
import os
import pdb
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde as smooth
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import yeojohnson
from tqdm import tqdm
from STAR_outliers_polishing_library import approximate_quantiles
from STAR_outliers_polishing_library import adjust_median_values

# source title: Outlier identification for skewed and/or 
#               heavy-tailed unimodal multivariate distributions

def estimate_tukey_params(W, bound):
    """
    Purpose
    -------
    To estimate tukey parameters for the distribution of deviation scores
    
    
    Parameters
    ----------
    W: the distribution of deviation scores
    bound: the highest percentile on which to fit the tukey parameters
           also, 100 - bound is the corresponding lowest percentile. 
    Returns
    -------
    A: tukey location parameter
    B: tukey scale parameter
    g: tukey skew parameter
    h: tukey tail heaviness parameter
    W_ignored: values of W that are too close to the median.
               very few values are too close in truly continuous data
               most data is not truly continuous and has an overabundance. 
               Note that the tukey only needs to fit the right side of 
               the W distribution, and ignored values are far to the left. 
    """
    # pdb.set_trace()
    
    Q_vec = np.percentile(W, [10, 25, 50, 75, 90])
    W_main = W[np.logical_and(W <= Q_vec[4], W >= Q_vec[0])]
    if len(np.unique(Q_vec)) != 5 or len(np.unique(W_main)) < 30:
        Q_vec = approximate_quantiles(W, [10, 25, 50, 75, 90])
    A = Q_vec[2]
    
    IQR = Q_vec[3] - Q_vec[1]
    SK = (Q_vec[4] + Q_vec[0] - 2*Q_vec[2])/(Q_vec[4] - Q_vec[0])
    T = (Q_vec[4] - Q_vec[0])/(Q_vec[3] - Q_vec[1])
    phi = 0.6817766 + 0.0534282*SK + 0.1794771*T - 0.0059595*(T**2)
    B = (0.7413*IQR)/phi
    Q_vec2 = np.percentile(W, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec2)) != 5 or len(np.unique(W_main)) < 30:
        Q_vec2 = approximate_quantiles(W, [100 - bound, 25, 50, 75, bound])
    zv = norm.ppf(bound/100, 0, 1)
    UHS = Q_vec2[4] - Q_vec2[2]
    LHS = Q_vec2[2] - Q_vec2[0]
    g = (1/zv)*np.log(UHS/LHS)
    y = (W - A)/B        
    Q_vec3 = np.percentile(y, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec3)) != 5 or len(np.unique(W_main)) < 30:
        Q_vec3 = approximate_quantiles(y, [100 - bound, 25, 50, 75, bound])
    Q_ratio = (Q_vec3[4]*Q_vec3[0])/(Q_vec3[4] + Q_vec3[0])
    h = (2/(zv**2))*np.log(-g*Q_ratio)
    if np.isnan(h):
        h = 0   
    return((A, B, g, h))

def compute_w(x):
    """
    Purpose
    -------
    To convert x values into a statistic that quantifies
    deviation from the mean relative to skew and tail heaviness
    
    Parameters
    ----------
    x: numeric input numpy array and the original
       dataset that needs outliers to be removed
    Returns
    -------
    W: a numeric numpy array and the distribution 
       of deviation scores for the x values
    """
    Q_vec = np.percentile(x, [10, 25, 50, 75, 90])
    x_main = x[np.logical_and(x <= Q_vec[4], x >= Q_vec[0])]
    if len(np.unique(Q_vec)) != 5 or len(np.unique(x_main)) < 30:
        Q_vec = approximate_quantiles(x, [10, 25, 50, 75, 90])
    x2 = adjust_median_values(x, Q_vec)
    
    c = 0.7413
    ASO_high = (x2 - Q_vec[2])/(2*c*(Q_vec[3] - Q_vec[2]))
    ASO_low = (Q_vec[2] - x2)/(2*c*(Q_vec[2] - Q_vec[1]))
    ASO = np.zeros(len(x2))
    ASO[x2 >= Q_vec[2]] = ASO_high[x2 >= Q_vec[2]]
    ASO[x2 < Q_vec[2]] = ASO_low[x2 < Q_vec[2]]
    
    W = norm.ppf((ASO + 1E-10)/(np.min(ASO) + np.max(ASO) + 2E-10))
    return(W)

outlier_indices = pd.read_csv("skewed_data_outlier_indices.txt", delimiter = "\t").to_numpy()
fake_data = pd.read_csv("skewed_data.txt", delimiter = "\t").to_numpy()
prefix = "original_method_skewed_data"
if not os.path.exists(prefix):
    os.mkdir(prefix)
if not os.path.exists(prefix + "_untransformed"):
    os.mkdir(prefix + "_untransformed")
for bound in [99]:
    dataset_names, TPR, FPR, correct_cutoffs, cutoffs = np.arange(len(fake_data.T)), [], [], [], []
    for i in tqdm(range(len(dataset_names))):

        x, ind, name = (fake_data.T)[i], (outlier_indices.T)[i], dataset_names[i]
        W, pcutoff = compute_w(x), 0.993
        A, B, g, h = estimate_tukey_params(W, bound)
        z = np.random.normal(0, 1, 200000)
        fitted_TGH  = A + B*(1/(g + 1E-10))*(np.exp((g + 1E-10)*z)-1)*np.exp(h*(z**2)/2)
        cutoff = np.percentile(fitted_TGH, pcutoff*100)
        x_outliers = x[W > cutoff]
        ind_pred = W > cutoff
        TPR.append(np.sum(ind_pred[ind == True])/np.sum(ind == True)*100)
        FPR.append(np.sum(ind_pred[ind == False])/np.sum(ind == False)*100)
        correct_cutoffs.append(np.percentile(W[ind == False], 99.3))
        cutoffs.append(cutoff)

        if bound == 99:
            x_unique, x_counts = np.unique(x, return_counts = True)
            q5, q95 = np.percentile(x, [5, 95])
            x_main = x[np.logical_and(x <= q95, x >= q5)]
            bw_coef = 0.075
            if len(np.unique(x_main)) < 1000:
                bw_coef = 0.3
            if len(np.unique(x_main)) < 100:
                bw_coef = 0.5
            if len(np.unique(x_main)) < 30:
                bw_coef = 0.7
            delta = (np.percentile(W, 99) - np.percentile(W, 1))/2
            xlims = [np.percentile(W, 1) - delta, np.percentile(W, 99) + delta]
            range0 = np.linspace(xlims[0] - delta, xlims[1] + delta, np.max([250, int(len(W)/300)]))
            smooth_TGH = smooth(fitted_TGH, bw_method =  bw_coef)(range0)
            plt.hist([W[ind == False], np.concatenate([W[ind]]*15)], bins = 75, density = True,
                     stacked = True, color = ['blue', 'red'], label = ["data", "outliers (enlarged 15x for emphasis)"],
                     edgecolor = 'black')
            plt.plot(range0, smooth_TGH, "c-", linewidth = 4, label = "fitted tukey distribution")
            plt.plot([cutoff, cutoff], [0, 1], "k-", label = "predicted outlier threshold")
            plt.legend(loc = "upper left", prop={'size': 11})
            plt.xlabel("ASO-probit transformed value", fontsize = 14.5)
            plt.ylabel("probability density", fontsize = 14.5)
            plt.xticks(fontsize=12.5)
            plt.yticks(fontsize=12.5)
            
            plt.title("dataset" + str(name) + "_bound" + str(bound))
            plt.savefig(prefix + "/dataset" + str(name) + "_bound" + str(bound) + ".png")
            plt.clf()

            plt.hist(x, bins = 75, color = 'blue', density = True, edgecolor = 'black')
            plt.savefig(prefix + "_untransformed/dataset" + str(name) + "_bound" + str(bound) + ".png")
            plt.clf()


    df = pd.DataFrame(np.array([dataset_names, TPR, FPR, correct_cutoffs, cutoffs]).T)
    df.columns = ["simulated dataset", "TPR", "FPR", "correct cutoff", "cutoff"]
    df.to_csv("original_method_skewed_data_" + str(bound) + ".txt", sep = "\t", header = True, index = False)

#-----------------------------------------------------------------------------------------------------------------------------
# For STAR_outliers
#-----------------------------------------------------------------------------------------------------------------------------

true_outliers = pd.read_csv("skewed_data_outlier_indices.txt", delimiter = "\t").to_numpy().T
pred_outliers = pd.read_csv("skewed_data_cleaned_data.txt", delimiter = "\t").to_numpy().T
pred_outliers = np.isnan(pred_outliers)

dataset_names, TPR, FPR = np.arange(len(true_outliers)), [], []
for inds, inds_pred in zip(true_outliers, pred_outliers):
    TPR.append(np.sum(inds_pred[inds == True])/np.sum(inds))
    FPR.append(np.sum(inds_pred[inds == False])/np.sum(inds == False))

df = pd.DataFrame(np.array([dataset_names, TPR, FPR]).T)
df.columns = ["simulated dataset", "TPR", "FPR"]
df.to_csv("STAR_efficacy_skewed.txt", sep = "\t", header = True, index = False)

