import pandas as pd
import os
import re
import glob
from sklearn.metrics import recall_score, f1_score, accuracy_score, \
    precision_score, roc_auc_score, balanced_accuracy_score
import numpy as np

'''
Create the paths to the appropriate stored files
'''

reps = 30

# for exp in range(reps):

# exp_rep = str(exp+1)

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/" \
#        "Y-Permutation_Validation_Experiments/" \
#        "Experiment_Set_2_thermal_Permutations_Revnano/" \
#        "Experiment_Set_2_thermal_Y_Permutation_RevNano_" + str(exp_rep)

# Directories containing files
cwd = os.getcwd()

# cv = "3CV"
cv = "5CV"

# change the path to the correct file path
# experiment = "Extra_Trees_RFE_" + cv + "_Stratified_NoRevNano"
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"
# experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Permutation"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
#        "Experiment Set 2/" \
#        "full_dataset_thermal_binary/" \
#        "80_pearson_corr/removed_features_" \
#        + cv + "_subset_balanced"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
#        "Experiment Set 3/" \
#        "bolstered_train_thermal_profile_binary/" \
#        "80_pearson_corr/removed_features_" \
#        + cv + "_subset_balanced"

path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
       "Cardinal_Features_Removed/" \
       "Experiment Set 2 RevNano Base Removed/" \
       "full_dataset_thermal_profile_binary/" \
       "80_pearson_corr/" \
       "features_removed_" + cv + "_subset_balanced/"

# additional directory paths
path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_weighted_final_scores.csv"
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

all_final_scores_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])

for i in range(reps):
    rep = str(i+1)
    print("rep:", rep)
    all_files = sorted(glob.glob(path_metadata + experiment + "_actual_vs_pred_rep_" + rep +
                                 "_fold_*.csv"), key=numerical_sort)
    new_accuracy_score = []
    new_balanced_accuracy_score = []
    new_recall_score = []
    new_precision_score = []
    new_f1_score = []
    new_rocauc_score = []
    scores = []

    unweighted_accuracy_score = []
    unweighted_balanced_accuracy_score = []
    unweighted_recall_score = []
    unweighted_precision_score = []
    unweighted_f1_score = []
    unweighted_rocauc_score = []
    unweighted_scores = []

    for j in all_files:
        print(j)
        fold_cv_df = pd.read_csv(j)
        fold_cv_df.drop(columns=["Unnamed: 0"], inplace=True)
        pred = fold_cv_df["prediction"].values
        y_test = fold_cv_df["reality"].values

        # Create scoring metrics and append for folds
        accuracy_result = accuracy_score(y_test, pred)
        scores.append(accuracy_result)
        new_accuracy_score.append(accuracy_result)

        # Balanced Accuracy
        balanced_accuracy_result = balanced_accuracy_score(y_test, pred)
        scores.append(balanced_accuracy_result)
        new_balanced_accuracy_score.append(balanced_accuracy_result)

        # Precision Score
        precision_result = precision_score(y_test, pred, average="weighted")
        scores.append(precision_result)
        new_precision_score.append(precision_result)

        # Recall Score
        recall_result = recall_score(y_test, pred, average="weighted")
        scores.append(recall_result)
        new_recall_score.append(recall_result)

        # F1 Score
        f1_result = f1_score(y_test, pred, average="weighted")
        scores.append(f1_result)
        new_f1_score.append(f1_result)

        # ROC AUC Score
        roc_auc_result = roc_auc_score(y_test, pred, average="weighted")
        scores.append(roc_auc_result)
        new_rocauc_score.append(roc_auc_result)

        # # Create scoring metrics and append for folds
        # accuracy_unweighted_result = accuracy_score(y_test, pred)
        # unweighted_scores.append(accuracy_unweighted_result)
        # unweighted_accuracy_score.append(accuracy_unweighted_result)
        #
        # # Balanced Accuracy
        # balanced_accuracy_unweighted_result = balanced_accuracy_score(y_test, pred)
        # unweighted_scores.append(balanced_accuracy_unweighted_result)
        # unweighted_balanced_accuracy_score.append(balanced_accuracy_unweighted_result)
        #
        # # Precision Score
        # precision_unweighted_result = precision_score(y_test, pred)
        # unweighted_scores.append(precision_unweighted_result)
        # unweighted_precision_score.append(precision_unweighted_result)
        #
        # # Recall Score
        # recall_unweighted_result = recall_score(y_test, pred)
        # unweighted_scores.append(recall_unweighted_result)
        # unweighted_recall_score.append(recall_unweighted_result)
        #
        # # F1 Score
        # f1_unweighted_result = f1_score(y_test, pred)
        # unweighted_scores.append(f1_unweighted_result)
        # unweighted_f1_score.append(f1_unweighted_result)
        #
        # # ROC AUC Score
        # roc_auc_unweighted_result = roc_auc_score(y_test, pred)
        # unweighted_scores.append(roc_auc_unweighted_result)
        # unweighted_rocauc_score.append(roc_auc_unweighted_result)

    # Create any necessary averages of scores for nested models and print them
    final_accuracy = np.average(new_accuracy_score)
    final_balanced_accuracy = np.average(new_balanced_accuracy_score)
    final_precision = np.average(new_precision_score)
    final_recall = np.average(new_recall_score)
    final_f1 = np.average(new_f1_score)
    final_roc_auc = np.average(new_rocauc_score)
    print("final accuracy", final_accuracy)
    print("final balanced accuracy", final_balanced_accuracy)
    print("final precision", final_precision)
    print("final recall", final_recall)
    print("final F1", final_f1)
    print("final roc auc", final_roc_auc)

    final_scores = [final_balanced_accuracy, final_precision, final_recall, final_f1, final_roc_auc]
    all_final_scores_df.loc[len(all_final_scores_df)] = final_scores

all_final_scores_df.to_csv(path_final, index=False, header=False)

    # # Create any necessary averages of scores for nested models and print them
    # final_unweighted_accuracy = np.average(unweighted_accuracy_score)
    # final_unweighted_balanced_accuracy = np.average(unweighted_balanced_accuracy_score)
    # final_unweighted_precision = np.average(unweighted_precision_score)
    # final_unweighted_recall = np.average(unweighted_recall_score)
    # final_unweighted_f1 = np.average(unweighted_f1_score)
    # final_unweighted_roc_auc = np.average(unweighted_rocauc_score)
    # print("final_unweighted accuracy", final_unweighted_accuracy)
    # print("final_unweighted balanced accuracy", final_unweighted_balanced_accuracy)
    # print("final_unweighted precision", final_unweighted_precision)
    # print("final_unweighted recall", final_unweighted_recall)
    # print("final_unweighted F1", final_unweighted_f1)
    # print("final_unweighted roc auc", final_unweighted_roc_auc)
