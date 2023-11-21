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
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Permutation"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
#        "Experiment Set 1/" \
#        "full_dataset_thermal_binary/" \
#        "80_pearson_corr/" \
#        + cv

# applied for permutation
reps = 30
for j in range(reps):
    experiment_number = str(j + 1)
    path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Y-Permutation_Validation_Experiments/" \
           "Experiment_Set_2_thermal_Permutations_Revnano/" \
           "Experiment_Set_2_thermal_Y_Permutation_RevNano_" + experiment_number + "/"

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

    # perform it via final score
    # all_files = sorted(glob.glob(path_metadata + experiment +
    #                              "_rep_*_final_score.csv"), key=numerical_sort)
    #
    # print(all_files)
    # if len(all_files) >= 30:
    #     print("correct amount")
    #     concat_pd_df = pd.DataFrame(columns=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
    #     for file in all_files:
    #         print(file)
    #         fold_cv_df = pd.read_csv(file, names=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
    #         list_of_values = fold_cv_df.values.tolist()[0]
    #         concat_pd_df.loc[len(concat_pd_df)] = list_of_values
    #         # Extra_Trees_RFE_3CV_Stratified_Baseline_final_scores
    #     print(concat_pd_df)
    #     concat_pd_df.reset_index(drop=True, inplace=True)
    #     concat_pd_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    #     print(concat_pd_df)
    #     concat_pd_df.to_csv(path_final, index=False, header=False)
    #     # print(concat_pd_df)
    #     # average_of_set_df = pd.DataFrame(columns=["r2", "MAE", "RMSE", "MSE", "MedianAE"])
    #     # average_of_set_df.loc[0] = concat_pd_df.mean()
    #     #
    #     # print(average_of_set_df)
    #     # average_of_set_df.to_csv(path_final + experiment + "_final_scores.csv")

    # perform it via folds
    all_final_scores_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])

    for repetition in range(1, 31):

        all_files = sorted(glob.glob(path_metadata + experiment +
                                     "_rep_" + str(repetition) +
                                     "_fold_*_score.csv"), key=numerical_sort)

        count = 0
        all_fold_scores_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])
        for file in all_files:
            rep_final_score_fold_df = pd.read_csv(file, names=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"])
            list_of_values = rep_final_score_fold_df.values.tolist()[0]
            all_fold_scores_df.loc[count] = list_of_values
            count += 1
        # average it
        all_final_scores_df.loc[repetition-1] = all_fold_scores_df.mean()
    all_final_scores_df.to_csv(path_final, index=False, header=False)

# # Extra_Trees_RFE_3CV_Stratified_Permutation_rep_4_fold_1_score
# # -0.005381652040774283, 3.064624057198624, 14.422109771672394, 3.7972273321445518, 1.9562741888841364
