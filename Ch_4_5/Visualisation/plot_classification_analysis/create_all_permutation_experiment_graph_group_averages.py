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

predictor = "thermal"

predictor_for_titles = "Thermal Profile (Binary)"

classification_columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
all_permutation_average_final_scores_df = pd.DataFrame(columns=classification_columns)
reps = 30
for j in range(reps):
    experiment_number = str(j + 1)
    # path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Y-Permutation_Validation_Experiments/" \
    #        "Experiment_Set_2_" + predictor + "_Permutations_Revnano/" \
    #        "Experiment_Set_2_" + predictor + "_Y_Permutation_RevNano_" + experiment_number + "/"
    path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Y-Permutation_Validation_Experiments/" \
           "Experiment_Set_2_" + predictor + "_Permutations/" \
           "Experiment_Set_2_" + predictor + "_Y_Permutation_" + experiment_number + "/"
    # additional directory paths
    path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_weighted_final_scores.csv"
    print(path_final)
    final_scores_permutation_df = pd.read_csv(path_final, names=classification_columns)
    all_permutation_average_final_scores_df.loc[j] = final_scores_permutation_df.mean()

print(all_permutation_average_final_scores_df)

# import final scores and average for the Experiment Sets, to create plots
# create plots of r2, MAE, RMSE, MSE in a 4 plot configuration, 2,2 MatPlotLib
# Magnesium Experiment Sets:
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"
set_1_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/" \
             "Classification Experiments/" \
             "Experiment Set 1/" \
             "full_dataset_thermal_binary/" \
             "80_pearson_corr/" \
             "removed_features_3CV_subset_balanced/"

set_1_path_final = set_1_path + \
                   "/extra_trees_results/model_final_scores/" + \
                   experiment + "_final_scores.csv"

set_2_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/" \
             "Classification Experiments/" \
             "Experiment Set 2/" \
             "full_dataset_thermal_binary/" \
             "80_pearson_corr/" \
             "removed_features_3CV_subset_balanced/"

set_2_path_final = set_2_path + \
                   "/extra_trees_results/model_final_scores/" + \
                   experiment + "_weighted_final_scores.csv"

set_3_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/" \
             "Classification Experiments/" \
             "Experiment Set 3/" \
             "full_dataset_thermal_binary/" \
             "80_pearson_corr/" \
             "removed_features_3CV_subset_balanced/"

set_3_path_final = set_3_path + \
                   "/extra_trees_results/model_final_scores/" + \
                   experiment + "_final_scores.csv"

# Magnesium with RevNano Sets
with_revnano_set_1_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                           "Cardinal_Features_Removed/" \
                           "Experiment Set 1 RevNano Base Removed/" \
                           "full_dataset_" + predictor + "_profile_binary/" \
                           "80_pearson_corr/" \
                           "features_removed_3CV_subset_balanced/"

with_revnano_set_1_path_final = with_revnano_set_1_path + \
                                "/extra_trees_results/model_final_scores/" + \
                                experiment + "_final_scores.csv"

# Magnesium with RevNano Sets
with_revnano_set_2_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                           "Cardinal_Features_Removed/" \
                           "Experiment Set 2 RevNano Base Removed/" \
                           "full_dataset_" + predictor + "_profile_binary/" \
                           "80_pearson_corr/" \
                           "features_removed_3CV_subset_balanced/"

with_revnano_set_2_path_final = with_revnano_set_2_path + \
                                "/extra_trees_results/model_final_scores/" + \
                                experiment + "_weighted_final_scores.csv"

# Magnesium WITHOUT RevNano Sets
without_revnano_experiment = "Extra_Trees_RFE_" + cv + "_Stratified_NoRevNano"
without_revnano_set_1_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                              "Cardinal_Features_Removed/" \
                              "Experiment Set 1/" \
                              "full_dataset_" + predictor + "_profile_binary/" \
                              "80_pearson_corr/" \
                              "features_removed_3CV_subset_balanced/"


without_revnano_set_1_path_final = without_revnano_set_1_path + \
                                   "/extra_trees_results/model_final_scores/" + \
                                   without_revnano_experiment + "_final_scores.csv"

# Magnesium with RevNano Sets
without_revnano_set_2_path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
                              "Cardinal_Features_Removed/" \
                              "Experiment Set 2/" \
                              "full_dataset_" + predictor + "_profile_binary/" \
                              "80_pearson_corr/" \
                              "features_removed_3CV_subset_balanced/"


without_revnano_set_2_path_final = without_revnano_set_2_path + \
                                   "/extra_trees_results/model_final_scores/" + \
                                   without_revnano_experiment + "_final_scores.csv"

# set_path_list = [set_1_path_final, set_2_path_final, set_3_path_final,
#                  with_revnano_set_1_path_final, with_revnano_set_2_path_final,
#                  without_revnano_set_1_path_final, without_revnano_set_2_path_final]
# with_revnano_sets_groups_list = ["Experiment Set 1", "Experiment Set 2", "Experiment Set 3",
#                                  "RevNano Experiment Set 1", "RevNano-Baseline Set 1",
#                                  "RevNano Experiment Set 2", "RevNano-Baseline Set 2"]
# set_groups_list = ["Experiment Set 1", "Experiment Set 2", "Experiment Set 3"]

set_path_list = [set_2_path_final]
groups_list = ["Experiment Set 2"]

all_experiment_sets_df = pd.DataFrame(columns=classification_columns)
# loop over the sets to create a data frame of experimental sets
for i in range(len(groups_list)):
    path_final = set_path_list[i]
    # combine into data-frame
    final_scores_df = pd.read_csv(path_final, names=classification_columns)
    all_experiment_sets_df.loc[i] = final_scores_df.mean()


final_scores_df = pd.read_csv(set_2_path_final, names=classification_columns)
all_experiment_sets_df["group"] = groups_list


# Multiple Plot Example
# # subplots (multiple 1 plot)
fig, ((ax1, ax2, ax6), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(15, 15))
label_fontsize = 18
global_legend_size = 12

ax6.hist(all_permutation_average_final_scores_df[['Accuracy']], bins=20, density=False,
         label="y-permutation Experiments (n=30)")
ax6.set_visible(False)

ax1.hist(all_permutation_average_final_scores_df[['Accuracy']], bins=20, density=False,
         label="y-permutation Experiments (n=30)")
ax1.set_xlabel("Accuracy score", fontsize=label_fontsize)
_ = ax1.set_ylabel("Probability", fontsize=label_fontsize)
color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
colour_panel = ["blue", "green", "magenta"]
for group in groups_list:
    # c = next(color)
    c = colour_panel[group_count]
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax1.axvline(experiment_set[['Accuracy']].values, ls="--", color=c, label=group, lw=3)
    group_count += 1
ax1.legend(loc=2, prop={'size': global_legend_size})
# ax1.set_xlim([0.20, 1])
# ax1.set_ylim([0, 5])

ax2.hist(all_permutation_average_final_scores_df[['Precision']], bins=20, density=False,
         label="y-permutation Experiments (n=30)")
ax2.set_xlabel("Precision", fontsize=label_fontsize)
_ = ax2.set_ylabel("Probability", fontsize=label_fontsize)
color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
colour_panel = ["blue", "green", "magenta"]
for group in groups_list:
    # c = next(color)
    c = colour_panel[group_count]
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax2.axvline(experiment_set[['Precision']].values, ls="--", color=c, label=group, lw=3)
    group_count += 1
ax2.legend(loc=2, prop={'size': global_legend_size})
# ax2.set_xlim([0.20, 1])
# ax2.set_ylim([0, 5])

ax3.hist(all_permutation_average_final_scores_df[['Recall']], bins=20, density=False,
         label="y-permutation Experiments (n=30)")
ax3.set_xlabel("Recall", fontsize=label_fontsize)
_ = ax3.set_ylabel("Probability", fontsize=label_fontsize)
# color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
colour_panel = ["blue", "green", "magenta"]
for group in groups_list:
    # c = next(color)
    c = colour_panel[group_count]
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax3.axvline(experiment_set[['Recall']].values, ls="--", color=c, label=group, lw=3)
    group_count += 1
ax3.legend(loc=2, prop={'size': global_legend_size})
# ax3.set_xlim([0.20, 1])
# ax3.set_ylim([0, 5])

ax4.hist(all_permutation_average_final_scores_df[['F1_Score']], bins=20, density=False,
         label="y-permutation Experiments (n=30)")
ax4.set_xlabel("F1_Score", fontsize=label_fontsize)
_ = ax4.set_ylabel("Probability", fontsize=label_fontsize)
# color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
colour_panel = ["blue", "green", "magenta"]
for group in groups_list:
    # c = next(color)
    c = colour_panel[group_count]
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax4.axvline(experiment_set[['F1_Score']].values, ls="--", color=c, label=group, lw=3)
    group_count += 1
ax4.legend(loc=2, prop={'size': global_legend_size})
# ax4.set_xlim([0.20, 1])
# ax4.set_ylim([0, 5])

ax5.hist(all_permutation_average_final_scores_df[['ROC_AUC']], bins=20, density=False,
         label="y-permutation Experiments (n=30)")
ax5.set_xlabel("ROC_AUC", fontsize=label_fontsize)
_ = ax5.set_ylabel("Probability", fontsize=label_fontsize)
# color = iter(cm.rainbow(np.linspace(0, 0.7, len(groups_list))))
group_count = 0
colour_panel = ["blue", "green", "magenta"]
for group in groups_list:
    # c = next(color)
    c = colour_panel[group_count]
    experiment_set = all_experiment_sets_df.iloc[group_count, :]
    ax5.axvline(experiment_set[['ROC_AUC']].values, ls="--", color=c, label=group, lw=3)
    group_count += 1
ax5.legend(loc=2, prop={'size': global_legend_size})
# ax4.set_xlim([0.20, 1])
# ax4.set_ylim([0, 5])


title = ("Average (n=30) Evaluation metrics for " + predictor_for_titles +
         " predictions from "
         "Experiment Set 2 y-permutation experiments "
         "and Machine Learning experiments")

fig.suptitle("\n".join(wrap(title, 70)), size=20)
fig.set_dpi(300)
plt.tight_layout()

# plt.draw()
fig.savefig("permutation_comparison_graphs/" + predictor_for_titles +
            " Comparison of Experiment Set 2" + ".png",
            bbox_inches='tight', format='png')
plt.show()
