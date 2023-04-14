import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
import pdb

values = pd.read_csv("original_method_skewed_data_99.txt", delimiter = "\t").to_numpy()
data1 = np.array([np.linspace(3.6, 33, 50), values[0:50, 1]/100])
data2 = np.array([np.linspace(0.015, 0.75, 50), values[50:100, 1]/100])
data3 = np.array([np.linspace(1.03, 2.5, 50), values[100:150, 1]/100])
original_data = [data1, data2, data3]

values = pd.read_csv("STAR_efficacy_skewed.txt", delimiter = "\t").to_numpy()
data1 = np.array([np.linspace(3.6, 33, 50), values[0:50, 1]])
data2 = np.array([np.linspace(0.015, 0.75, 50), values[50:100, 1]])
data3 = np.array([np.linspace(1.03, 2.5, 50), values[100:150, 1]])
STAR_data = [data1, data2, data3]

x_labels = ["interpeak distance", "tukey skew parameter (g)", "exponent value (a)"]
y_labels = 3*["outlier detection TPR"]
titles = ["outlier cutoff error vs\noutlier detection TPR",
          "distribution skew vs\noutlier detection TPR",
          "monotonic distribution tail heaviness vs\noutlier detection TPR"]
filenames = ["figure 3d", "figure 3e", "figure 3f"]

objects = zip(original_data, STAR_data, x_labels, y_labels, titles, filenames)
for xo, xs, lx, ly, t, f in objects:
    
    plt.plot(xo[0], xo[1], "ro", label = "Verardi & Vermandele method", alpha = 0.5)
    plt.plot(xs[0], xs[1], "co", label = "STAR_outliers", alpha = 0.5)
    plt.legend()
    plt.xlabel(lx, fontsize = 14.5)
    plt.ylabel(ly, fontsize = 14.5)
    plt.xticks(fontsize=12.5)
    plt.yticks(fontsize=12.5)
    plt.legend(loc = "lower left", prop={'size': 11})
    plt.savefig(f + ".png")
    plt.clf()
