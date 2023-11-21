import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from textwrap import wrap



# Directories containing files
cwd = os.getcwd()
path = "C:/Users/Big JC/PycharmProjects/PhDCodeOnly/ML_results_to_analyse/" \
           "pearson_corr_removed_models_other_predictors/best_pearson_corr_removed_models/scaffold_molarity/3CV/"
path_metadata = path + "/extra_trees_results/model_metadata/"
path_all = path + "extra_trees_results/model_plots/"
experiment_name = "Extra_Trees_RFE_3CV_Stratified_RevNano"

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def plot_all_predicted_vs_actual(path_to_file):
    all_files = sorted(glob.glob(path_to_file + experiment_name + "_actual_vs_pred_rep_" + "*" + "_fold_" +
                       "*.csv"), key=numerical_sort)
    data_set_file_path = cwd + "/correct_imputation_magnesium_v3_ml_data_set_no_0_25.xlsx"
    original_data_frame = pd.read_excel(data_set_file_path).reset_index()
    y = original_data_frame["Magnesium (mM)"]
    for file in all_files:
        print(file)
        path, filename = os.path.split(file)
        print(filename)
        rep_regex = r"rep_\d*"
        rep_num = re.search(rep_regex, filename)
        fold_regex = r"fold_\d*"
        fold_num = re.search(fold_regex, filename)
        repetition = str(rep_num.group())
        inner_loop_rep = str(fold_num.group())

        real_vs_actual_df = pd.read_csv(file)
        # import df of real and predicted values of folds
        real_list = real_vs_actual_df['reality']
        pred_list = real_vs_actual_df['prediction']

        # Save folds actual vs predicts -- DIAGNOSTIC --
        fig, ax = plt.subplots()
        ax.scatter(real_list, pred_list)
        ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        title = (experiment_name + "_Actual_vs_Predicted_plot_rep_" + repetition + "_fold_" + inner_loop_rep)
        plt.title("\n".join(wrap(title, 44)))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        # plt.savefig(path + experiment_name +
        #             "_Actual_vs_Predicted_plot_rep_" + repetition + "_fold_"
        #             + str(inner_loop_rep) + '.png', bbox_inches="tight")
        plt.show()


def plot_combined_predicted_vs_actual(path_to_file):
    all_files = sorted(glob.glob(path_to_file + experiment_name + "_actual_vs_pred_rep_" + "*" +
                       "best_all_folds.csv"), key=numerical_sort)
    data_set_file_path = cwd + "/correct_imputation_magnesium_v3_ml_data_set_no_0_25.xlsx"
    original_data_frame = pd.read_excel(data_set_file_path).reset_index()
    # Remove all Experiments with Anomalous Mg values
    original_data_frame = original_data_frame[~original_data_frame['Paper Number'].isin(['8', '46', '48',
                                                                                         '59', '71', '85', '96', '98'])]
    # original_data_frame = original_data_frame[~original_data_frame['Paper Number'].isin(['48',
    #                                                                                      ])]
    y = original_data_frame["Magnesium (mM)"]
    for file in all_files:
        print(file)
        path, filename = os.path.split(file)
        print(filename)
        rep_regex = r"rep_\d*"
        rep_num = re.search(rep_regex, filename)
        repetition = str(rep_num.group())

        real_vs_actual_df = pd.read_csv(file)
        # import df of real and predicted values of folds
        real_list = real_vs_actual_df['reality']
        pred_list = real_vs_actual_df['prediction']

        # Save folds actual vs predicts -- DIAGNOSTIC --
        fig, ax = plt.subplots()
        ax.scatter(real_list, pred_list)
        ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        title = ("Extra Trees 5CV Stratified RevNano" + " Actual vs Predicted plot rep " + repetition)
        plt.title("\n".join(wrap(title, 29)))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        # plt.savefig(path + experiment_name +
        #             "_Actual_vs_Predicted_plot_rep_" + repetition + "_fold_"
        #             + str(inner_loop_rep) + '.png', bbox_inches="tight")
        plt.show()


# plot_all_predicted_vs_actual(path_metadata)
plot_combined_predicted_vs_actual(path_all)
