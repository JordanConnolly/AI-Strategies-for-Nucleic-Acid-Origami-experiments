import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from textwrap import wrap

plt.style.use('ggplot')  # best so far
# plt.style.use('seaborn-whitegrid')

'''
Create the paths to the appropriate stored files
'''


# PERMUTATION EXPERIMENT

# Directories containing files
cwd = os.getcwd()

cv = "3CV"
# cv = "5CV"

# change the path to the correct file path
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Permutation"

# applied to permutation
# gather permutation final scores

all_permutation_average_final_scores_df = pd.DataFrame(columns=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
reps = 30
for j in range(reps):
    experiment_number = str(j + 1)
    path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Y-Permutation_Validation_Experiments/" \
           "Experiment_Set_2_Magnesium_Permutations/" \
           "Experiment_Set_2_Magnesium_Y_Permutation_" + experiment_number + "/"

    # additional directory paths
    path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_final_scores.csv"
    print(path_final)
    final_scores_permutation_df = pd.read_csv(path_final, names=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
    all_permutation_average_final_scores_df.loc[j] = final_scores_permutation_df.mean()

print(all_permutation_average_final_scores_df)

# import final scores and average for the Experiment Sets, to create plots
# create plots of r2, MAE, RMSE, MSE in a 4 plot configuration, 2,2 MatPlotLib
# Magnesium Experiment Sets:
magnesium_experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"
set_1_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Regression Experiments/" + \
             "Cardinal_Features_Removed/" \
             "Experiment Set 1/" \
             "full_dataset_magnesium/" \
             "80_pearson_corr/" \
             + cv + "/"

set_1_path_final = set_1_path + \
                   "/extra_trees_results/model_final_scores/" + \
                   magnesium_experiment + "_final_scores.csv"

set_2_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Regression Experiments/" + \
             "Cardinal_Features_Removed/" \
             "Experiment Set 2/" \
             "full_dataset_magnesium/" \
             "80_pearson_corr/" \
             + cv + "/"

set_2_path_final = set_2_path + \
                   "/extra_trees_results/model_final_scores/" + \
                   magnesium_experiment + "_final_scores.csv"

set_3_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Regression Experiments/" + \
             "Cardinal_Features_Removed/" \
             "Experiment Set 3/" \
             "bolstered_train_magnesium/" \
             "80_pearson_corr/" \
             + cv + "/"

set_3_path_final = set_3_path + \
                   "/extra_trees_results/model_final_scores/" + \
                   magnesium_experiment + "_final_scores.csv"

# Magnesium with RevNano Sets
with_revnano_set_1_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                           "Cardinal_Features_Removed/" \
                           "Experiment Set 1 RevNano Base Removed/" \
                           "full_dataset_magnesium/" \
                           "80_pearson_corr/" \
                           "removed_features_3CV/"

with_revnano_set_1_path_final = with_revnano_set_1_path + \
                                "/extra_trees_results/model_final_scores/" + \
                                magnesium_experiment + "_final_scores.csv"

# Magnesium with RevNano Sets
with_revnano_set_2_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                           "Cardinal_Features_Removed/" \
                           "Experiment Set 2 RevNano Base Removed/" \
                           "full_dataset_magnesium/" \
                           "80_pearson_corr/" \
                           "removed_features_3CV/"

with_revnano_set_2_path_final = with_revnano_set_2_path + \
                                "/extra_trees_results/model_final_scores/" + \
                                magnesium_experiment + "_final_scores.csv"

# Magnesium WITHOUT RevNano Sets
without_revnano_experiment = "Extra_Trees_RFE_" + cv + "_Stratified_NoRevNano"
without_revnano_set_1_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                               "Cardinal_Features_Removed/" \
                               "Experiment Set 1/" \
                               "full_dataset_magnesium/" \
                               "80_pearson_corr/" \
                               "removed_features_3CV/"

without_revnano_set_1_path_final = without_revnano_set_1_path + \
                                   "/extra_trees_results/model_final_scores/" + \
                                   without_revnano_experiment + "_final_scores.csv"

# Magnesium with RevNano Sets
without_revnano_set_2_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                           "Cardinal_Features_Removed/" \
                           "Experiment Set 2/" \
                           "full_dataset_magnesium/" \
                           "80_pearson_corr/" \
                           "removed_features_3CV/"

without_revnano_set_2_path_final = without_revnano_set_2_path + \
                                   "/extra_trees_results/model_final_scores/" + \
                                   without_revnano_experiment + "_final_scores.csv"

set_path_list = [set_1_path_final, set_2_path_final, set_3_path_final,
                 with_revnano_set_1_path_final, with_revnano_set_2_path_final,
                 without_revnano_set_1_path_final, without_revnano_set_2_path_final]
with_revnano_sets_groups_list = ["Experiment Set 1", "Experiment Set 2", "Experiment Set 3",
                                 "RevNano Experiment Set 1", "RevNano-Baseline Set 1",
                                 "RevNano Experiment Set 2", "RevNano-Baseline Set 2"]
groups_list = ["Experiment Set 1", "Experiment Set 2", "Experiment Set 3"]

all_experiment_sets_df = pd.DataFrame(columns=["r2", "MAE", "RMSE", "MSE", "MedianAE", "group"])
# loop over the sets to create a data frame of experimental sets
for i in range(len(groups_list)):
    path_final = set_path_list[i]
    # combine into data-frame
    final_scores_df = pd.read_csv(path_final, names=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
    all_experiment_sets_df.loc[i] = final_scores_df.mean()

all_experiment_sets_df["group"] = groups_list

# Individual Plot Example
plot_metric = "r2 score"
fig, ax = plt.subplots()
ax.hist(all_permutation_average_final_scores_df[['r2']], bins=20, density=False,
        label="y-permutation Experiments (n=30)")
ax.set_xlabel(plot_metric)
_ = ax.set_ylabel("Probability")
color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
for group in groups_list:
    c = next(color)
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax.axvline(experiment_set[['r2']].values, ls="--", color=c, label=group)
    group_count += 1

title = (plot_metric + " from Magnesium (mM) y-permutation experiments "
         "and Machine Learning Experiment Sets")
plt.title("\n".join(wrap(title, 55)), fontsize=12)
plt.legend(loc=2, prop={'size': 6})
# plt.xlim(0.2, 1)
plt.ylim(0, 8)
plt.show()
plt.cla()
plt.close()

# MAE
plot_metric = "MAE score"
fig, ax = plt.subplots()
ax.hist(all_permutation_average_final_scores_df[['MAE']], bins=20, density=False,
        label="y-permutation Experiments (n=30)")
ax.set_xlabel(plot_metric)
_ = ax.set_ylabel("Probability")
color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
for group in groups_list:
    c = next(color)
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax.axvline(experiment_set[['MAE']].values, ls="--", color=c, label=group)
    group_count += 1

title = (plot_metric + " from Magnesium (mM) y-permutation experiments "
         "and Machine Learning Experiment Sets")
plt.title("\n".join(wrap(title, 55)), fontsize=12)
plt.legend(loc=2, prop={'size': 6})
plt.xlim(0, )
plt.ylim(0, 8)
plt.show()
plt.cla()
plt.close()

# MSE
plot_metric = "MSE score"
fig, ax = plt.subplots()
ax.hist(all_permutation_average_final_scores_df[['MSE']], bins=20, density=False,
        label="y-permutation Experiments (n=30)")
ax.set_xlabel(plot_metric)
_ = ax.set_ylabel("Probability")
color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
for group in groups_list:
    c = next(color)
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax.axvline(experiment_set[['MSE']].values, ls="--", color=c, label=group)
    group_count += 1

title = (plot_metric + " from Magnesium (mM) y-permutation experiments "
         "and Machine Learning Experiment Sets")
plt.title("\n".join(wrap(title, 55)), fontsize=12)
plt.legend(loc=2, prop={'size': 6})
plt.xlim(0, )
plt.ylim(0, 8)
plt.show()
plt.cla()
plt.close()

# RMSE
plot_metric = "RMSE score"
fig, ax = plt.subplots()
ax.hist(all_permutation_average_final_scores_df[['RMSE']], bins=20, density=False,
        label="y-permutation Experiments (n=30)")
ax.set_xlabel(plot_metric)
_ = ax.set_ylabel("Probability")
color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
for group in groups_list:
    c = next(color)
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax.axvline(experiment_set[['RMSE']].values, ls="--", color=c, label=group)
    group_count += 1

title = (plot_metric + " from Magnesium (mM) y-permutation experiments "
         "and Machine Learning Experiment Sets")
plt.title("\n".join(wrap(title, 55)), fontsize=12)
plt.legend(loc=2, prop={'size': 6})
plt.xlim(0, )
plt.ylim(0, 8)
plt.show()
plt.cla()
plt.close()
