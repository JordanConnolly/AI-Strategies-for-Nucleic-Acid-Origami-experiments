import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectKBest, f_regression, chi2
from sklearn import decomposition
from numpy.random import lognormal
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import sklearn
import seaborn as sns
import csv
import shap
import random
import pickle
import os
from textwrap import wrap
from pathlib import Path

# Pandas and Numpy Options
cwd = os.getcwd()
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')


def get_transformer_feature_names(column_transformer):
    """Allows you to get the feature names after they have been
    transformed by the pipeline pre-processing, such as extension of
    categorical variable column names"""
    output_features = []

    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            for j in pipe:
                trans_features = []
                if hasattr(j, 'categories_'):
                    trans_features.extend(j.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)
    return output_features


def concurrent_shuffler(split_list, seed_number_chosen, label):
    value = (split_list.loc[:, label])
    start_list = []
    end_list = []
    end_index = 0
    # list of current magnesium values
    current_list = [value[mag] for mag in range(len(value))]
    # list of next magnesium values
    next_list = [value[mag + 1] for mag in range(len(value) - 1)]
    current_list.insert(len(current_list), 0)
    next_list.insert(len(next_list), 0)
    combined_list = list(zip(current_list, next_list))
    all_index_df = pd.DataFrame()
    for i in range(len(combined_list)):
        list_instance = combined_list[i]
        current_magnesium = list_instance[0]
        next_magnesium = list_instance[1]
        if current_magnesium > next_magnesium:
            start_index = end_index
            end_index = i + 1
            if end_index - start_index > 1:
                start_list.append(start_index)
                end_list.append(end_index)
                df = split_list[start_index:end_index].sample(frac=1, random_state=seed_number_chosen)
                all_index_df = pd.concat([all_index_df, df])
            else:
                df = split_list[start_index:end_index]
                all_index_df = pd.concat([all_index_df, df])
    return all_index_df


def stratified_dataset(data, cv_chosen, label, seed_number, index_required):
    """
    stratified approach is to sort the Y variable, proceed to ensure that the data
    is put forwards to be split in even distribution.
    Step 1: Sort Y variable.
    Step 2: Create n folds before Inner Nested Loop.
    Step 3: Put the n distributions into Inner K Fold.
    Gather the scores and data from the model and see the difference to random dist.
    """
    # sort by y label values
    data_set_sorted = data.sort_values([label], ascending=False)
    data_set_sorted = data_set_sorted.reset_index()
    new_data_set = concurrent_shuffler(data_set_sorted, seed_number_chosen=seed_number, label=label)
    # shuffled data set, now split into cv folds
    names = new_data_set.columns.values
    total_list = []
    data_set_used = new_data_set.drop(columns=['index'], errors='ignore')
    for j in data_set_used.itertuples():
        total_list.append(j)
    total_y_list = []
    # #### SET THIS TO FOLD COUNT #####
    cv_count = cv_chosen
    ###################################
    for cv_chosen in range(cv_count):
        y_list_cv_chosen = total_list[cv_chosen::cv_count]
        total_y_list = (total_y_list + y_list_cv_chosen)
    if index_required:
        final_data_set = pd.DataFrame(total_y_list, columns=names)
        index = final_data_set['index']
    else:
        final_data_set = pd.DataFrame(total_y_list, columns=names)
        index = []
    return final_data_set, index


# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)

# import the data set
data_set_file_path = cwd + '/correct_imputation_magnesium_v3_ml_data_set_no_0_25.xlsx'
data_set = pd.read_excel(data_set_file_path)
# Remove all Experiments with Anomalous Mg values
data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

print(max(data_set["Magnesium (mM)"]))

# data_set_file = data_set.to_csv("correct_imputation_magnesium_v3_ml_data_set_no_0_25.csv")

# which rep is performed
script = Path(__file__).stem
rep = 1
number_of_splits = 3

seed_number = seed_numbers[rep]
final_data_set, index = stratified_dataset(data_set, number_of_splits, 'Magnesium (mM)', seed_number,
                                           index_required=True)
print(max(final_data_set["Magnesium (mM)"]))

###### remove Correlated Features ####
y = final_data_set['Magnesium (mM)']
# drop correlated features
corr_matrix = final_data_set.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
drop_value = 0.80
to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)


# Change the X and Y data, experiment name, scoring used
x = final_data_set.drop(columns=['Magnesium (mM)', 'index', 'Unnamed: 0',
                                 'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                 'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')
# y = final_data_set['Magnesium (mM)']
experiment_name = "Extra_Trees_RFE_3CV_Stratified_Baseline"

print(max(y))


outer_cv = KFold(n_splits=number_of_splits, shuffle=False)

data_splits = list(outer_cv.split(x, y))
# Scores
rep_final = []
scores = []
r2 = []
mae = []
mse = []
rmse = []
median_ae = []

# Actual vs Prediction lists
predictions = []
reality = []
inner_loop_rep = 0

#  Creates data splits
for tr_idx, val_idx in data_splits:
    inner_loop_rep += 1

    # Removes correct answers from New_X, creates train-test sets
    X_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
    X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

    # Store Y values of split -- DIAGNOSTIC --
    y_data = list(zip(y_train, y_test))
    y_data_store = pd.DataFrame(y_train, y_test)

    print("printing y train and test for fold")
    print(max(y_train))
    print(y_test)

