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

np.random.seed(0)
    
def tukey(A, B, g, h, n):
    z = np.random.normal(0, 1, n)
    t = A + B*(1/(g + 1E-10))*(np.exp((g + 1E-10)*z)-1)*np.exp(h*(z**2)/2)
    return(t)

def simulate_outliers(X, two_sided, stds, num_outliers = 300):
    N, P = len(X[0]), len(X)
    O, mid_mark = num_outliers, int(num_outliers/2)
    p_bounds_sidedness = [[0, 99.5], [0.25, 99.75]]
    outlier_statuses = np.zeros((P, N)).astype(np.bool_)
    for i in range(P):
        p_bounds = p_bounds_sidedness[int(two_sided[i])]
        lb, ub = np.percentile(X[i], p_bounds)
        llb, ulb = lb - np.array([2, 0.5])*stds[i]
        lub, uub = ub + np.array([0.5, 2])*stds[i]
        outlier_inds = np.random.choice(np.arange(N), O, replace = False)
        outlier_statuses[i][outlier_inds] = True
        if two_sided[i]:
            X[i][outlier_inds[0:mid_mark]] = np.random.uniform(llb, ulb, mid_mark)
            X[i][outlier_inds[mid_mark:O]] = np.random.uniform(lub, uub, O - mid_mark)
        else:
            X[i][outlier_inds[0:O]] = np.random.uniform(lub, uub, O)   
    return(X, outlier_statuses)

N = 100000
fake_data = np.zeros((150, N))
multimodal_std = 0
for i in range(0, 50, 1):

    fake_data[i] = np.concatenate([np.random.normal(-3 - 0.6*(i+1), 1, 33333),
                                   np.random.normal(0, 1, 33334),
                                   np.random.normal(3 + 0.6*(i+1), 1, 33333)])
    fake_data[50 + i] = tukey(0, 0.5, 0.015*(i+1), 0, N)
    fake_data[100 + i] = np.random.exponential(1, N)**(1 + 0.03*(i+1))

two_sided = np.array(100*[True] + 50*[False])
stds = np.std(fake_data, axis = 1)
stds[0:50] = stds[0]
fake_data, true_outliers = simulate_outliers(fake_data, two_sided, stds)

pd.DataFrame(fake_data.T).to_csv("skewed_data.txt", sep = "\t", header = True, index = False)
pd.DataFrame(true_outliers.T).to_csv("skewed_data_outlier_indices.txt", sep = "\t", header = True, index = False)
