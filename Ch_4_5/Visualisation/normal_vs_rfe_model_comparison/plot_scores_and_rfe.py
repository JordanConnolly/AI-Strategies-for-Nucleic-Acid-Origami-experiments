import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy.random import lognormal

# Pandas and Numpy Options
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

file_path_normal = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
                   "magnesium_experiments/machine-learning/results_path/results/" \
                   "ten_percent_nan_removed_experiments/NaN-models/"

FE_path_1 = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
            "magnesium_experiments/machine-learning/results_path/results/" \
            "ten_percent_nan_removed_experiments/NaN-models/"

FE_path_2 = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
            "magnesium_experiments/machine-learning/results_path/results/" \
            "ten_percent_nan_removed_experiments/NaN-models/"

storage_path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
               "magnesium_experiments/machine-learning/results_path/results/" \
               "normal_vs_rfe_model_comparison/normal_vs_rfe_plots"

names = ['r2', 'mae', 'mse', 'rmse', 'medianae']

normal_file = 'Elastic_Net_Regression_10_Percent_NaN_Removed_scores.csv'
file_split = normal_file.split(".")
file_name = file_split[0]
fe_file = 'Perm_Y_Elastic_Net_Regression_10_Percent_NaN_Removed_scores.csv'
fe_file_2 = 'Perm_Y_Elastic_Net_Regression_10_Percent_NaN_Removed_scores.csv'

normal_set = pd.read_csv(file_path_normal + normal_file,
                         names=names, header=None)

fe_set = pd.read_csv(FE_path_1 + fe_file,
                     names=names, header=None)

fe_set_2 = pd.read_csv(FE_path_2 + fe_file_2,
                       names=names, header=None)

for i in names:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    fe = fe_set[i]
    fe_2 = fe_set_2[i]
    # Histogram plot settings
    plt.axvline(normal[0], color='k', linestyle='dashed', linewidth=1, label='regression model score')
    plt.axvline(fe[0], color='k', linewidth=1, label='Select_K regression model score')
    plt.axvline(fe_2[0], color='k', linewidth=1, label='RFE regression model score')

    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    pyplot.xlim([0, 1])
    plt.ylabel("Experiment Count")
    x_ticker = [0.0, 1.1]
    plt.xticks(np.arange(min(x_ticker), max(x_ticker), 0.1))
    plt.title(file_name)

    # store / save the plots
    plt.savefig(storage_path + file_name + "_vs_RFE_" + i + ".png")
    pyplot.show()
