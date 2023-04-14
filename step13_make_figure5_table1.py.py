import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import pdb

table1_normal = pd.read_csv("table1_normal.txt", delimiter = "\t")
table1_uniform = pd.read_csv("table1_uniform.txt", delimiter = "\t")
table1 = table1_normal.merge(table1_uniform, on = "model name", how = "inner")
table1.to_csv("table1.txt", sep = "\t", header = True, index = False)

def bootstrap(D, B):
    N = len(D)
    samples = np.random.choice(D, N*B).reshape(N, B)
    means = np.mean(samples, axis = 0)
    CI = np.percentile(means, [2.5, 97.5])
    return(CI)

def inner_join_all(df1, df2):
    return(df1.merge(df2, on = "dataset", how = "inner"))

names = ["T2", "T2_yj", "3SD", "[3] (p = 99)", "STAR_outliers"]
fnames = ["T2_real_efficacies.txt", "T2_yj_real_efficacies.txt"]
fnames += ["ASD_real_efficacies.txt", "original_method_99_real_efficacy_.txt"]
fnames += ["STAR_real_efficacy.txt"]
fnames = ["real_efficacies/" + name for name in fnames]

efficacies = [pd.read_csv(name, delimiter = "\t") for name in fnames]
for i, df in enumerate(efficacies): df.columns = ["dataset", names[i]]
efficacies = reduce(inner_join_all, efficacies)
efficacies = np.abs(efficacies.to_numpy()[:, 1:] - 0.007)*100
CIs = np.array([bootstrap(efficacies[:, i], 1000000) for i in range(len(efficacies[0]))])
pd.DataFrame(CIs).to_csv("performance_CIs.txt", sep = "\t", header = False, index = False)

means = np.mean(efficacies, axis = 0)
inds = np.arange(len(means))
errs = np.abs(means - CIs.T)

plt.figure(figsize = (10, 6))
plt.subplots_adjust(bottom = 0.2)
plt.plot(inds, means, 'o')
plt.ylabel("mean absolute difference from\n 0.7% of data points removed", fontsize=16)
plt.errorbar(inds, means, yerr = errs, fmt='ko', capsize = 10)
plt.xticks(inds, labels = names, rotation = 45, ha="right", fontsize=12)
plt.tick_params(labelsize=12)
plt.ylim(0, 4) 
plt.savefig("figure5.png")
plt.clf()
