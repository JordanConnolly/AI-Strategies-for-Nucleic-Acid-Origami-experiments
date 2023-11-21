import pandas as pd
import os
import re
import glob


'''
Create the paths to the appropriate stored files
'''

# Directories containing files
cwd = os.getcwd()

cv = "3CV"
# cv = "5CV"

# change the path to the correct file path
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_NoRevNano"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
#        "Experiment Set 1/" \
#        "full_dataset_thermal_binary/" \
#        "80_pearson_corr/" \
#        + cv

path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
       "Cardinal_Features_Removed/" \
       "Experiment Set 2/" \
       "full_dataset_thermal_profile_binary/" \
       "80_pearson_corr/" \
       "features_removed_3CV_subset_balanced/"

# additional directory paths
path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_final_scores.csv"
path_metadata = path + "/extra_trees_results/model_metadata/"
path_all = path + "/extra_trees_results/model_plots/"
path_pickle = path + "extra_trees_results/model_pickle/"

'''
Add numerical sort for sorting files later
'''

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

'''
Create the table for repetition experiments
'''

reps = 30
for i in range(reps):
    rep = str(i+1)
    print("rep:", rep)
    all_files = sorted(glob.glob(path_metadata + experiment + "_actual_vs_pred_rep_" + rep +
                                 "_fold_*.csv"), key=numerical_sort)
    concat_pd_df = pd.DataFrame()

    for j in all_files:
        print(j)
        fold_cv_df = pd.read_csv(j)
        concat_pd_df = pd.concat([concat_pd_df, fold_cv_df])
    concat_pd_df.reset_index(drop=True, inplace=True)
    concat_pd_df.drop(columns=["Unnamed: 0"], inplace=True)

    concat_pd_df.to_csv(path_all + experiment + "_actual_vs_pred_rep_" + rep + "_best_all_folds.csv")
