import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import random
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import os
import re
import glob
from textwrap import wrap

# print(plt.style.available)
plt.style.use('ggplot')  # best so far
# plt.style.use('seaborn-whitegrid')

'''
Create the paths to the appropriate stored files
'''

# Directories containing files
cwd = os.getcwd()

# original_data = "/correct_imputation_magnesium_v3_ml_data_set_no_0_25.csv"
original_data = "/subset_1_all_magnesium_0_25_yield_removed_literature_ml_data_set.csv"

cv = "3CV"
# cv = "5CV"

# Title to change for Plot
# change_title = "Dataset 1 Subset 1 Extra Trees RFE " + cv + " Stratified"
change_title = "Experiment Set 1 Extra Trees RFE " + cv + " Stratified"
# change_title = "Increased Training Data Comparison Extra Trees RFE " + cv + " Stratified"

# change the path to the correct file path
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"

path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
       "Experiment Set 1/" \
       "full_dataset_thermal_binary/" \
       "80_pearson_corr/" \
       + cv

# additional directory paths
path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_final_scores.csv"
path_metadata = path + "/extra_trees_results/model_metadata/"
path_all = path + "/extra_trees_results/model_plots/"
path_pickle = path + "extra_trees_results/model_pickle/"

print(path_all)

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

columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
df = pd.read_csv(path_final, names=columns)
df.index += 1
df.sort_values(by=["Accuracy"], inplace=True)
df_low = pd.DataFrame(data=df.iloc[0]).transpose()
df_low_index = df_low.index[0]
df_high = pd.DataFrame(data=df.iloc[-1]).transpose()
df_high_index = df_high.index[0]
df_average = pd.DataFrame(data=df.apply(np.average), columns=["Average"]).transpose()
df_stdev = pd.DataFrame(data=df.apply(np.std), columns=["StDev"]).transpose()
total_result_df = df_low, df_high, df_average, df_stdev
result = pd.concat(total_result_df)
# creates a table for best and worst repetition
print(result)

'''
Create the plots for the best and worst result of repetition experiments
'''


def plot_confusion_matrix(path_to_file, original_data, change_title, original_predictor):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                print(repetition_value[1])
                real_vs_actual_df = pd.read_csv(file)
                # import df of real and predicted values of folds
                real_list = real_vs_actual_df['reality']
                pred_list = real_vs_actual_df['prediction']
                cm = metrics.confusion_matrix(real_list, pred_list)
                print(cm)
                c_report = metrics.classification_report(real_list, pred_list)
                ax = plt.subplot()
                sns.heatmap(cm, annot=True, ax=ax, cbar=False, cmap="YlGnBu", fmt='g')
                # labels, title and ticks
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                title = (original_predictor + change_title + ' Confusion Matrix; Rep: ' + str(repetition_value[1]))
                ax.set_title("\n".join(wrap(title, 42)))
                # plt.savefig("confusion_matrix_rep_" + str(repetition_value[1]) + ".png")
                # plt.cla()
                plt.show()

    elif "xlsx" in ext:
        for file in all_files:
            path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                print(repetition_value[1])
                real_vs_actual_df = pd.read_csv(file)
                # import df of real and predicted values of folds
                real_list = real_vs_actual_df['reality']
                pred_list = real_vs_actual_df['prediction']
                cm = metrics.confusion_matrix(real_list, pred_list)
                print(cm)
                c_report = metrics.classification_report(real_list, pred_list)
                ax = plt.subplot()
                sns.heatmap(cm, annot=True, ax=ax, cbar=False, cmap="YlGnBu", fmt='g')
                # labels, title and ticks
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                title = (original_predictor + change_title + ' Confusion Matrix; Rep: ' + str(repetition_value[1]))
                ax.set_title("\n".join(wrap(title, 42)))
                # plt.savefig("confusion_matrix_rep_" + str(repetition_value[1]) + ".png")
                # plt.cla()
                plt.show()
        return


def plot_roc_auc_curve(path_to_file, original_data, change_title, original_predictor):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                # print(repetition_value[1])
                real_vs_actual_df = pd.read_csv(file)
                # import df of real and predicted values of folds
                real_list = real_vs_actual_df['reality']
                pred_list = real_vs_actual_df['prediction']
                ax = plt.subplot()
                display = metrics.RocCurveDisplay.from_predictions(real_list, pred_list, ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                title = (original_predictor + change_title + ' ROC Curve; Repetition: ' + str(repetition_value[1]))
                ax.set_title("\n".join(wrap(title, 42)))
                plt.show()


plot_confusion_matrix(path_all, original_data, change_title, "Thermal Profile (Binary) ")
# plot_roc_auc_curve(path_all, original_data, change_title, "Thermal Profile (Binary) ")
