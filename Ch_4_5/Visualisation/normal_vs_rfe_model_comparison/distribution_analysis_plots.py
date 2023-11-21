import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy.random import lognormal


# Pandas and Numpy Options
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

storage_path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
            "magnesium_experiments/machine-learning/results_path/results/" \
            "store_plots/"

# Pandas and Numpy Options
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

names = ['r2', 'mae', 'mse', 'rmse', 'medianae']

model_name = "NaN Removed Lasso Regression"

normal_file = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
              "magnesium_experiments/machine-learning/results_path/results/" \
              "ten_percent_nan_removed_experiments/NaN-models/Lasso_Regression_10_Percent_NaN_Removed_scores.csv"

file_perm = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
            "magnesium_experiments/machine-learning/results_path/results/" \
            "/ten_percent_nan_removed_experiments/" \
            "NaN-models/Perm_Y_Lasso_Regression_10_Percent_NaN_Removed_scores.csv"

normal_set = pd.read_csv(normal_file,
                         names=names, header=None)

perm_set = pd.read_csv(file_perm,
                       names=names, header=None)


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh


# Create all graphs
for i in names:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    perm = perm_set[i]

    # apply mad_outlier_removal
    # mad_mask = mad_based_outlier(points=perm)
    # perm.drop(perm[mad_mask].index, inplace=True)
    # print(perm.shape)

    # Histogram plot settings
    pyplot.hist(perm, bins=100, alpha=0.5, label='permutation ' + i)
    plt.axvline(normal[0], color='m', linestyle='dashed', linewidth=2, label='regression model score')
    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    plt.ylabel("Experiment Frequency")
    plt.title(i + " values Regression_model")
    plt.autoscale(enable=True, axis='y', tight=True)

    # store / save the plots
    # plt.savefig(storage_path + "Ridge_Regression_model_vs_Permutation" + i + ".png")
    pyplot.show()


for i in ['r2']:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    perm = perm_set[i]
    # apply mad_outlier_removal
    # mad_mask = mad_based_outlier(points=perm)
    # perm.drop(perm[mad_mask].index, inplace=True)
    # print(perm.shape)

    # apply percentile limit
    # x_limit_set = normal[0] + 1
    # perm_limit_set = perm[perm.between(perm.quantile(0.01), perm.quantile(0.95))]
    # print(perm_limit_set.shape)

    # apply manual scaling of graphs
    # x_limit_set = normal[0] + 1
    print(normal)
    # Histogram plot settings
    pyplot.hist(perm, bins=150, alpha=0.5, label='permutation ' + i, range=(-10, 1))
    plt.axvline(normal[0], color='m', linestyle='dashed', linewidth=2, label='regression model score')
    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    plt.ylabel("Experiment Frequency")
    plt.title(i + " values " + model_name)
    plt.autoscale(enable=True, axis='x', tight=True)

    # store / save the plots
    plt.savefig(storage_path + model_name + "_vs_Permutation_" + i + ".png")
    pyplot.show()

for i in ['mae']:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    perm = perm_set[i]
    # apply mad_outlier_removal
    # mad_mask = mad_based_outlier(points=perm)
    # perm.drop(perm[mad_mask].index, inplace=True)
    # print(perm.shape)

    # apply percentile limit
    # x_limit_set = normal[0] + 1
    # perm_limit_set = perm[perm.between(perm.quantile(0.01), perm.quantile(0.95))]
    # print(perm_limit_set.shape)

    # apply manual scaling of graphs
    # x_limit_set = normal[0] + 1
    print(normal)
    # Histogram plot settings
    pyplot.hist(perm, bins=200, alpha=0.5, label='permutation ' + i, range=(5, 35))
    plt.axvline(normal[0], color='m', linestyle='dashed', linewidth=2, label='regression model score')
    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    plt.ylabel("Experiment Frequency")
    plt.title(i + " values " + model_name)
    plt.autoscale(enable=True, axis='x', tight=True)
    # store / save the plots
    plt.savefig(storage_path + model_name + "_vs_Permutation_" + i + ".png")
    pyplot.show()
#
for i in ['mse']:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    perm = perm_set[i]
    # apply mad_outlier_removal
    # mad_mask = mad_based_outlier(points=perm)
    # perm.drop(perm[mad_mask].index, inplace=True)
    # print(perm.shape)

    # apply percentile limit
    # x_limit_set = normal[0] + 1
    # perm_limit_set = perm[perm.between(perm.quantile(0.01), perm.quantile(0.95))]
    # print(perm_limit_set.shape)

    # apply manual scaling of graphs
    # x_limit_set = normal[0] + 1
    print(normal)
    # Histogram plot settings
    pyplot.hist(perm, bins=150, alpha=0.5, label='permutation ' + i, range=(350, 18000))
    plt.axvline(normal[0], color='m', linestyle='dashed', linewidth=2, label='regression model score')
    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    plt.ylabel("Experiment Frequency")
    plt.title(i + " values " + model_name)
    plt.autoscale(enable=True, axis='x', tight=True)

    # store / save the plots
    plt.savefig(storage_path + model_name + "_vs_Permutation_" + i + ".png")
    pyplot.show()

for i in ['rmse']:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    perm = perm_set[i]
    # apply mad_outlier_removal
    # mad_mask = mad_based_outlier(points=perm)
    # perm.drop(perm[mad_mask].index, inplace=True)
    # print(perm.shape)

    # apply percentile limit
    # x_limit_set = normal[0] + 1
    # perm_limit_set = perm[perm.between(perm.quantile(0.01), perm.quantile(0.95))]
    # print(perm_limit_set.shape)

    # apply manual scaling of graphs
    # x_limit_set = normal[0] + 1
    print(normal)
    # Histogram plot settings
    pyplot.hist(perm, bins=200, alpha=0.5, label='permutation ' + i, range=(14, 100))
    plt.axvline(normal[0], color='m', linestyle='dashed', linewidth=2, label='regression model score')
    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    plt.ylabel("Experiment Frequency")
    plt.title(i + " values " + model_name)
    plt.autoscale(enable=True, axis='x', tight=True)

    # store / save the plots
    plt.savefig(storage_path + model_name + "_vs_Permutation_" + i + ".png")
    pyplot.show()

for i in ['medianae']:
    # Gather data to plot
    print(i)
    i = str(i)
    normal = normal_set[i]
    perm = perm_set[i]
    # apply mad_outlier_removal
    # mad_mask = mad_based_outlier(points=perm)
    # perm.drop(perm[mad_mask].index, inplace=True)
    # print(perm.shape)

    # apply percentile limit
    # x_limit_set = normal[0] + 1
    # perm_limit_set = perm[perm.between(perm.quantile(0.01), perm.quantile(0.95))]
    # print(perm_limit_set.shape)

    # apply manual scaling of graphs
    # x_limit_set = normal[0] + 1
    print(normal)
    # Histogram plot settings
    pyplot.hist(perm, bins=200, alpha=0.5, label='permutation ' + i, range=(0, 14))
    plt.axvline(normal[0], color='m', linestyle='dashed', linewidth=2, label='regression model score')
    pyplot.legend(loc='upper right')
    plt.xlabel("Score Value")
    plt.ylabel("Experiment Frequency")
    plt.title(i + " values " + model_name)
    plt.autoscale(enable=True, axis='x', tight=True)

    # store / save the plots
    plt.savefig(storage_path + model_name + "_vs_Permutation_" + i + ".png")
    pyplot.show()
