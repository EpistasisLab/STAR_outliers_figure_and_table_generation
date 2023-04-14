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

def simulate_outliers(X, two_sided, num_outliers = 300):
    N, P, stds = len(X[0]), len(X), np.std(X, axis = 1)
    O, mid_mark = num_outliers, int(num_outliers/2)
    # p_bounds_sidedness = [[0, 99.3], [0.35, 99.65]]
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

fake_data = np.zeros((100, 100000))

# lognormals
fake_data[0] = np.random.lognormal(0, 0.2, 100000)
fake_data[1] = np.random.lognormal(0, 0.4, 100000)
fake_data[2] = np.random.lognormal(0, 0.6, 100000)
fake_data[3] = np.random.lognormal(0, 0.8, 100000)
fake_data[4] = np.random.lognormal(0, 1.0, 100000)
fake_data[5] = np.random.lognormal(0, 1.2, 100000)
fake_data[6] = np.random.lognormal(0, 1.4, 100000)
fake_data[7] = np.random.lognormal(0, 1.6, 100000)
fake_data[8] = np.random.lognormal(0, 1.8, 100000)
fake_data[9] = np.random.lognormal(0, 2.0, 100000)

# exponentials
fake_data[10] = np.random.exponential(1, 100000)
fake_data[11] = np.random.exponential(1, 100000)**1.5
fake_data[12] = np.random.exponential(1, 100000)**2
fake_data[13] = np.random.exponential(1, 100000)**2.5
fake_data[14] = np.random.exponential(1, 100000)**3
fake_data[15] = np.random.exponential(1, 100000)**3.5
fake_data[16] = np.random.exponential(1, 100000)**4
fake_data[17] = np.random.exponential(1, 100000)**4.5
fake_data[18] = np.random.exponential(1, 100000)**5
fake_data[19] = np.random.exponential(1, 100000)**5.5

# powers
fake_data[20] = 1 - np.random.power(2, 100000)
fake_data[21] = 1 - np.random.power(2.5, 100000)
fake_data[22] = 1 - np.random.power(3, 100000)
fake_data[23] = 1 - np.random.power(3.5, 100000)
fake_data[24] = 1 - np.random.power(4, 100000)
fake_data[25] = 1 - np.random.power(4.5, 100000)
fake_data[26] = 1 - np.random.power(5, 100000)
fake_data[27] = 1 - np.random.power(5.5, 100000)
fake_data[28] = 1 - np.random.power(6, 100000)
fake_data[29] = 1 - np.random.power(6.5, 100000)

# discrete distributions (test on discrete data)
fake_data[30] = np.random.poisson(2, 100000)
while len(np.unique(fake_data[30])) < 10:
    fake_data[30] = np.random.poisson(2, 100000)
fake_data[31] = np.random.poisson(4, 100000)
fake_data[32] = np.random.poisson(6, 100000)
fake_data[33] = np.random.poisson(8, 100000)
fake_data[34] = np.random.poisson(10, 100000)
fake_data[35] = np.random.poisson(12, 100000)
fake_data[36] = np.random.poisson(14, 100000)
fake_data[37] = np.random.poisson(16, 100000)
fake_data[38] = np.random.poisson(18, 100000)
fake_data[39] = np.random.poisson(20, 100000)


fake_data[40] = np.random.negative_binomial(5, 0.85, 100000)
while len(np.unique(fake_data[40])) < 10:
    fake_data[40] = np.random.negative_binomial(5, 0.85, 100000)
fake_data[41] = np.random.negative_binomial(5, 0.80, 100000)
fake_data[42] = np.random.negative_binomial(5, 0.75, 100000)
fake_data[43] = np.random.negative_binomial(5, 0.70, 100000)
fake_data[44] = np.random.negative_binomial(5, 0.65, 100000)
fake_data[45] = np.random.negative_binomial(5, 0.60, 100000)
fake_data[46] = np.random.negative_binomial(5, 0.55, 100000)
fake_data[47] = np.random.negative_binomial(5, 0.50, 100000)
fake_data[48] = np.random.negative_binomial(5, 0.45, 100000)
fake_data[49] = np.random.negative_binomial(5, 0.40, 100000)

# tukey distributions assymetric with regular tails
fake_data[50] = tukey(0, 0.5, 0.075, 0, 100000)
fake_data[51] = tukey(0, 0.5, 0.150, 0, 100000)
fake_data[52] = tukey(0, 0.5, 0.225, 0, 100000)
fake_data[53] = tukey(0, 0.5, 0.300, 0, 100000)
fake_data[54] = tukey(0, 0.5, 0.375, 0, 100000)
fake_data[55] = tukey(0, 0.5, 0.450, 0, 100000)
fake_data[56] = tukey(0, 0.5, 0.525, 0, 100000)
fake_data[57] = tukey(0, 0.5, 0.600, 0, 100000)
fake_data[58] = tukey(0, 0.5, 0.675, 0, 100000)
fake_data[59] = tukey(0, 0.5, 0.750, 0, 100000)

# tukey distributions symetric with fat tails
fake_data[60] = tukey(0, 0.5, 0, 0.075, 100000)
fake_data[61] = tukey(0, 0.5, 0, 0.150, 100000)
fake_data[62] = tukey(0, 0.5, 0, 0.225, 100000)
fake_data[63] = tukey(0, 0.5, 0, 0.300, 100000)
fake_data[64] = tukey(0, 0.5, 0, 0.375, 100000)
fake_data[65] = tukey(0, 0.5, 0, 0.450, 100000)
fake_data[66] = tukey(0, 0.5, 0, 0.525, 100000)
fake_data[67] = tukey(0, 0.5, 0, 0.600, 100000)
fake_data[68] = tukey(0, 0.5, 0, 0.675, 100000)
fake_data[69] = tukey(0, 0.5, 0, 0.750, 100000)

# tukey distributions assymetric fat tails
fake_data[70] = tukey(0, 0.5, 0.075, 0.075, 100000)
fake_data[71] = tukey(0, 0.5, 0.150, 0.150, 100000)
fake_data[72] = tukey(0, 0.5, 0.225, 0.225, 100000)
fake_data[73] = tukey(0, 0.5, 0.300, 0.300, 100000)
fake_data[74] = tukey(0, 0.5, 0.375, 0.375, 100000)
fake_data[75] = tukey(0, 0.5, 0.450, 0.450, 100000)
fake_data[76] = tukey(0, 0.5, 0.525, 0.525, 100000)
fake_data[77] = tukey(0, 0.5, 0.600, 0.600, 100000)
fake_data[78] = tukey(0, 0.5, 0.675, 0.675, 100000)
fake_data[79] = tukey(0, 0.5, 0.750, 0.750, 100000)

# finite domain distributions

fake_data[80] = np.random.beta(2, 2, 100000)
fake_data[81] = np.random.beta(2, 2.5, 100000)
fake_data[82] = np.random.beta(2, 3, 100000)
fake_data[83] = np.random.beta(2, 3.5, 100000)
fake_data[84] = np.random.beta(2, 4, 100000)
fake_data[85] = np.random.beta(2, 4.5, 100000)
fake_data[86] = np.random.beta(2, 5, 100000)
fake_data[87] = np.random.beta(2, 5.5, 100000)
fake_data[88] = np.random.beta(2, 6, 100000)
fake_data[89] = np.random.beta(2, 6.6, 100000)


fake_data[90] = np.random.uniform(0, 1, 100000)
fake_data[91] = np.concatenate((np.random.uniform(0.25, 0.75, 50000),
                                np.random.uniform(0, 1, 50000)))
fake_data[92] = np.random.triangular(0, 0.2, 1, 100000)
fake_data[93] = np.random.triangular(0, 0.5, 1, 100000)
fake_data[94] = np.concatenate((np.random.triangular(0, 0.5, 1, 50000),
                                np.random.uniform(0, 1, 50000)))

# multimodal distributions
fake_data[95] = np.concatenate((np.random.normal(-2.5, 1, 50000),
                                np.random.normal(2.5, 1, 50000)))
fake_data[96] = np.concatenate((np.random.normal(-4, 1, 40000),
                                np.random.normal(0, 1, 20000),
                                np.random.normal(4, 1, 40000)))
fake_data[97] = np.concatenate((np.random.triangular(-0.75, -0.5, 0.25, 50000),
                                np.random.triangular(-0.25, 0.5, 0.75, 50000)))
fake_data[98] = np.concatenate((np.random.normal(0, 1, 50000),
                                np.random.triangular(-0.5, 0, 0.5, 50000)))
partial = np.random.normal(0, 1, 90000)
bad_part = partial[partial > 2.6]
fake_data[99] = np.concatenate((partial[partial <= 2.6],
                                np.random.uniform(2.6, 10, 10000 + len(bad_part))))

two_sided = np.array(50*[False] + 50*[True])
fake_data, true_outliers = simulate_outliers(fake_data, two_sided)

df = pd.DataFrame(np.transpose(true_outliers))
df.to_csv("fake_data_true_outliers.txt", sep = "\t", header = True, index = False)

df = pd.DataFrame(np.transpose(fake_data))
df.columns = df.columns + 1
df.to_csv("fake_data.txt", sep = "\t", header = True, index = False)

