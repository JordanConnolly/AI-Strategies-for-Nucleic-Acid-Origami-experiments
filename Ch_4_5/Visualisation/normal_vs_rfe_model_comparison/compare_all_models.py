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
                   "all_features_used_experiments/elastic_net_model_results/"

file_path_rfe_etr = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
                    "magnesium_experiments/machine-learning/results_path/results/" \
                    "feature_elimination_experiments/rfe-etr-models/"

file_path_select_k = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
                     "magnesium_experiments/machine-learning/results_path/results/" \
                     "feature_elimination_experiments/select-k-models/"

file_path_nan = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
                "magnesium_experiments/machine-learning/results_path/results/" \
                "ten_percent_nan_removed_experiments/NaN-models/"

names = ['r2', 'mae', 'mse', 'rmse', 'medianae']

normal_file = 'Elastic_Net_Regression_Scores.csv'
file_split = normal_file.split(".")
file_name = file_split[0]

fe_file_rfe = 'Elastic_Net_Regression_RFE_ETR_scores.csv'
fe_file_select_k = 'Elastic_Net_Regression_Select_K_scores.csv'
fe_file_nan = 'Elastic_Net_Regression_10_Percent_NaN_Removed_scores.csv'

normal_set = pd.read_csv(file_path_normal + normal_file,
                         names=names, header=None)

fe_set_rfe = pd.read_csv(file_path_rfe_etr + fe_file_rfe,
                         names=names, header=None)

fe_set_select_k = pd.read_csv(file_path_select_k + fe_file_select_k,
                              names=names, header=None)

fe_set_nan = pd.read_csv(file_path_nan + fe_file_nan,
                         names=names, header=None)

for i in names:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    fe_rfe = fe_set_rfe[i]
    fe_select_k = fe_set_select_k[i]
    fe_nan = fe_set_nan[i]

    # # Histogram plot settings
    # # plt.axvline(normal[0], color='k', linestyle='dashed', linewidth=1, label=i + ' regression model score')
    # # plt.axvline(fe_rfe[0], color='k', linewidth=1, label=i + ' RFE regression model score')
    # # plt.axvline(fe_select_k[0], color='k', linewidth=1, label=i + ' Select K regression model score')
    # # plt.axvline(fe_nan[0], color='k', linewidth=1, label=i + ' ten percent nan removed regression model score')
    #
    # plt.bar([1], [normal[0], fe_rfe[0], fe_select_k[0], fe_nan[0]], width=0.1)
    # pyplot.legend(loc='upper right')
    # plt.xlabel("Score Value")
    # pyplot.xlim([0, 1])
    # plt.ylabel("Experiment Count")
    # x_ticker = [-1.0, 1.1]
    # plt.xticks(np.arange(min(x_ticker), max(x_ticker), 0.2))
    # plt.title(file_name)
    #
    # # store / save the plots
    # # plt.savefig(storage_path + file_name + "_vs_RFE_" + i + ".png")
    # pyplot.show()
