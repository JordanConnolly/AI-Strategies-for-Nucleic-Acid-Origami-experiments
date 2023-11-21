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
    all_permutation_average_final_scores_df = pd.concat([all_permutation_average_final_scores_df,
                                                         final_scores_permutation_df])

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
set_groups_list = ["Experiment Set 1", "Experiment Set 2", "Experiment Set 3"]
groups_list =["Experiment Set 2"]

all_experiment_sets_df = pd.DataFrame(columns=["r2", "MAE", "RMSE", "MSE", "MedianAE", "group"])
# loop over the sets to create a data frame of experimental sets
for i in range(len(groups_list)):
    path_final = set_path_list[i]
    # combine into data-frame
    final_scores_df = pd.read_csv(path_final, names=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
    all_experiment_sets_df = pd.concat([all_experiment_sets_df, final_scores_df])

# Multiple Plot Example
# # subplots (multiple 1 plot)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

label_fontsize = 18
global_legend_size = 12

ax1.hist(all_permutation_average_final_scores_df[['r2']], bins=360, density=False,
         label="y-permutation Experiment repetitions (n=900)")
ax1.hist(all_experiment_sets_df[['r2']], bins=10, density=False,
         label="Experiment Set 2 repetitions (n=30)")
ax1.set_xlabel("r2", fontsize=label_fontsize)
_ = ax1.set_ylabel("Probability", fontsize=label_fontsize)
ax1.legend(loc=2, prop={'size': global_legend_size})
ax1.set_xlim([-0.1, 1])
# ax1.set_ylim([0, 5])

ax2.hist(all_permutation_average_final_scores_df[['MAE']], bins=360, density=False,
         label="y-permutation Experiment repetitions (n=900)")
ax2.hist(all_experiment_sets_df[['MAE']], bins=10, density=False,
         label="Experiment Set 2 repetitions (n=30)")
ax2.set_xlabel("MAE", fontsize=label_fontsize)
_ = ax2.set_ylabel("Probability", fontsize=label_fontsize)
ax2.legend(loc=2, prop={'size': global_legend_size})
ax2.set_xlim([0, 4])
# ax2.set_ylim([0, 5])

ax3.hist(all_permutation_average_final_scores_df[['MSE']], bins=360, density=False,
         label="y-permutation Experiment repetitions (n=900)")
ax3.hist(all_experiment_sets_df[['MSE']], bins=10, density=False,
         label="Experiment Set 2 repetitions (n=30)")
ax3.set_xlabel("MSE", fontsize=label_fontsize)
_ = ax3.set_ylabel("Probability", fontsize=label_fontsize)
ax3.legend(loc=2, prop={'size': global_legend_size})
ax3.set_xlim([0, 4])
# ax3.set_ylim([0, 5])

ax4.hist(all_permutation_average_final_scores_df[['RMSE']], bins=360, density=False,
         label="y-permutation Experiment repetitions (n=900)")
ax4.hist(all_experiment_sets_df[['RMSE']], bins=10, density=False,
         label="Experiment Set 2 repetitions (n=30)")
ax4.set_xlabel("RMSE", fontsize=label_fontsize)
_ = ax4.set_ylabel("Probability", fontsize=label_fontsize)
ax4.legend(loc=2, prop={'size': global_legend_size})
ax4.set_xlim([0, 16])
# ax4.set_ylim([0, 5])

title = ("Evaluation metrics for Magnesium (mM) predictions from "
         "Experiment Set 2 y-permutation experiments "
         "and Machine Learning experiments")

fig.suptitle("\n".join(wrap(title, 70)), size=20)
fig.set_dpi(300)
plt.tight_layout()

# plt.draw()
fig.savefig("permutation_comparison_graphs/" +
            "Magnesium (mM) Comparison of Experiment Set 2 Repetitions" + ".png",
            bbox_inches='tight', format='png')
plt.show()
