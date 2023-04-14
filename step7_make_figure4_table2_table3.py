import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pdb

def F1_bootstrap_CI(file, B):
    F1_data = file[["TPR", "precision"]].to_numpy()
    F1_scores = 2*np.product(F1_data, axis = 1)/np.sum(F1_data, axis = 1)
    F1_scores[np.sum(F1_data, axis = 1) == 0] = 0
    N = len(F1_scores)
    F1_B = np.mean(np.random.choice(F1_scores, (B, N)), axis = 1)
    mu_lb_ub = [np.mean(F1_scores)] + np.percentile(F1_B , [2.5, 97.5]).tolist()
    return(mu_lb_ub)

sim_file_names = np.array(os.listdir("simulated_efficacies"))
sim_names = np.array([name.split("_efficacy")[0] for name in sim_file_names])
sim_file_names = np.array(["simulated_efficacies/" + file for file in sim_file_names])
sim_files = [pd.read_csv(name, delimiter = "\t") for name in sim_file_names]
sim_F1_stats = np.array([F1_bootstrap_CI(file, 100000) for file in sim_files])
sim_F1_means = np.array([stats[0] for stats in sim_F1_stats])
sim_F1_lb = np.array([stats[1] for stats in sim_F1_stats])
sim_F1_ub = np.array([stats[2] for stats in sim_F1_stats])

simple_inds = np.isin(sim_names, ['ASD', 'ASD_yj','IQR', 'IQR_yj', 'MAD', 'MAD_yj'])
T2_inds = np.isin(sim_names, ['T2', 'T2_yj', 'T3', 'T3_yj', 'T4', 'T4_yj'])
original_inds = np.isin(sim_names, ['original_method_' + str(i) for i in np.arange(90, 100)])
IF_inds = np.isin(sim_names, ['Information_Forest_adjusted', 'Information_Forest'])
STAR_inds = np.isin(sim_names, ['STAR'])

tab2_names = [name.replace("ASD", "3SD") for name in sim_names]
tab2_names = [name.replace("Information_Forest_adjusted", "IF-calibrated") for name in tab2_names]
tab2_names = [name.replace("Information_Forest", "IF") for name in tab2_names]
tab2_names = [name.replace("original_method_", "[3] (p = ") for name in tab2_names]
for i in range(len(tab2_names)):
    if "[3]" in tab2_names[i]: tab2_names[i] = tab2_names[i] + ")"

type_names = ["normal", "normal", "IF", "IF", "normal"]
type_names += ["normal", "normal", "normal", "[3]", "[3]"]
type_names += ["[3]", "[3]", "[3]", "[3]", "[3]"]
type_names += ["[3]", "[3]", "[3]", "STAR_outliers", "[9]"]
type_names += ["[9]", "[9]", "[9]", "[9]", "[9]"]

descriptions = ["standard 3SD cutoff"]
descriptions += ["3SD cutoff for yj transformed data"]
descriptions += ["IF calibrated to remove as many outliers as STAR_outliers"]
descriptions += ["IF out of the box model"]
descriptions += ["standard IQR cutoff"]
descriptions += ["standard IQR cutoff for yj transformed data"]
descriptions += ["standard MAD cutoff"]
descriptions += ["standard MAD cutoff for yj transformed data"]
descriptions += ["[3] using 90 and 10 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 91 and 9 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 92 and 8 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 93 and 7 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 94 and 6 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 95 and 5 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 96 and 4 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 97 and 3 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 98 and 2 as outer percentiles for tukey parameter estimation"]
descriptions += ["[3] using 99 and 1 as outer percentiles for tukey parameter estimation"]
descriptions += ["STAR_outliers"]
descriptions += ["T2 with 2 iterations"]
descriptions += ["T2 with 2 iterations for yj transformed data"]
descriptions += ["T2 with 3 iterations"]
descriptions += ["T2 with 3 iterations for yj transformed data"]
descriptions += ["T2 with 4 iterations"]
descriptions += ["T2 with 4 iterations for yj transformed data"]

table2 = pd.DataFrame(np.array([tab2_names, type_names, descriptions]).T).sort_values(by = [1, 0])
table2.columns = ["algorithm", "algorithm type", "description"] 
table2.to_csv("table2.txt", sep = "\t", header = True, index = False)

best_simple = sim_names[simple_inds][np.argmax(sim_F1_means[simple_inds])]
best_T2 = sim_names[T2_inds][np.argmax(sim_F1_means[T2_inds])]
best_original = sim_names[original_inds][np.argmax(sim_F1_means[original_inds])]
best_IF = sim_names[IF_inds][np.argmax(sim_F1_means[IF_inds])]
best_STAR = sim_names[STAR_inds][0]

best_names = np.array([best_simple, best_T2, best_original, best_IF, "T2", best_STAR])
best_inds = np.array([np.where(name == sim_names)[0][0] for name in best_names])
best_names[np.where(np.array(best_names) == 'Information_Forest_adjusted')[0][0]] = 'IF_calibrated'
best_names[np.where(np.array(best_names) == 'original_method_99')[0][0]] = '[3] (p = 99)'
best_names[np.where(np.array(best_names) == 'ASD')[0][0]] = '3SD'
best_files = [sim_files[i] for i in best_inds]

dist_names = ["lognormal", "exponential", "power"]
dist_names += ["poisson", "negative binomial", "tukey-g"]
dist_names += ["tukey-h", "tukey-gh", "beta", "non-standard"]

dist_descriptions = ["lognormal distribution: 10 spread parameter values"]
dist_descriptions += ["exponential random variables drawn and then raised to a power: 10 power values"]
dist_descriptions += ["power distribution: 10 power values"]
dist_descriptions += ["poisson distribution: 10 parameter values"]
dist_descriptions += ["negative binomial distribution: 10 success probabilities"]
dist_descriptions += ["tukey-gh distribution: 10 g values, h = 0"]
dist_descriptions += ["tukey-gh distribution: 10 h values, g = 0"]
dist_descriptions += ["tukey-gh distribution: 10 values, g = h"]
dist_descriptions += ["beta distribution: 10 beta values, alpha = 2"]
dist_descriptions += ["various non-smooth and/or multimodal distributions: 10 different shapes (figure 4)"]

table3 = pd.DataFrame(np.array([dist_names , dist_descriptions]).T)
table3.columns = ["distribution type", "description"] 
table3.to_csv("table3.txt", sep = "\t", header = True, index = False)

lognormal_names = np.round((np.linspace(0.2, 2, 10)), 6).astype(str)
exponential_names = np.round((np.linspace(1, 5.5, 10)), 6).astype(str)
power_names = np.round((np.linspace(2, 6.5, 10)), 6).astype(str)
poisson_names = np.round((np.linspace(2, 20, 10)), 6).astype(str)
NB_names = np.round((np.linspace(0.4, 0.85, 10)), 6).astype(str)
tukey_g_names = np.round((np.linspace(0.075, 0.75, 10)), 6).astype(str)
tukey_h_names = np.round((np.linspace(0.075, 0.75, 10)), 6).astype(str)
tukey_gh_names = np.round((np.linspace(0.075, 0.75, 10)), 6).astype(str)
beta_names = np.round((np.linspace(2, 6.5, 10)), 6).astype(str)
nonstandard_names = np.arange(1, 11).astype(str)

name_sets = [lognormal_names, exponential_names, power_names]
name_sets += [poisson_names, NB_names, tukey_g_names]
name_sets += [tukey_h_names, tukey_gh_names, beta_names, nonstandard_names]

dist_inds = np.arange(100).reshape(10,10)
fig, ax = plt.subplots(nrows = 5, ncols = 4, figsize = (15, 10))
best_data_lineshapes = ["r-", "b-", "g-", "c-", "m-", "k-"]
k = 0
plot_inds = [(0, 0), (0, 1), (0, 2), (0, 3)]
plot_inds += [(1, 0), (1, 1), (1, 2), (1, 3)]
plot_inds += [(2, 0), (2, 1), (2, 2), (2, 3)]
plot_inds += [(3, 0), (3, 1), (3, 2), (3, 3)]
plot_inds += [(4, 0), (4, 1), (4, 2), (4, 3)]
for inds, params, dist_type in zip(dist_inds, name_sets, dist_names):
    k_TPR, k_FPR = plot_inds[k], plot_inds[k + 1]
    k += 2
    for file, name, shape in zip(best_files, best_names, best_data_lineshapes):
        
        file_data = file.loc[inds, ["TPR", "FPR"]].to_numpy()
        TPR, FPR = file_data[:, 0], file_data[:, 1]
        ax[k_TPR].plot(params, TPR, shape, label = name)
        ax[k_FPR].plot(params, FPR, shape, label = name)
        
    ax[k_TPR].set_xticklabels(params, rotation = 45)
    ax[k_FPR].set_xticklabels(params, rotation = 45)
    ax[k_TPR].title.set_text(dist_type + " TPR")
    ax[k_FPR].title.set_text(dist_type + " FPR")
handles, labels = ax[k_TPR].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle('TPR and FPR comparison between methods\n and across distribution types', fontsize = 14)
fig.supxlabel('distribution parameter value', fontsize = 12)
fig.supylabel('TPR/FPR (depending on subplot)', fontsize = 12)
fig.tight_layout(pad = 1.3, h_pad = 0.5, w_pad = 1)
fig.subplots_adjust(bottom = 0.075, top = 0.91, right = 0.95)
plt.savefig("figure4.png")

##################################################################################################
#
# real data analysis
#
##################################################################################################

real_file_names = os.listdir("real_efficacies")
real_names = [name.split("_real")[0] for name in real_file_names]
real_file_names = ["real_efficacies/" + file for file in real_file_names]
real_files = [pd.read_csv(name, delimiter = "\t") for name in real_file_names]
STAR_inds = [pd.read_csv(name, delimiter = "\t") for name in real_file_names]


