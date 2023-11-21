import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
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
import random
import pickle
import os
from textwrap import wrap
from skrebate import ReliefF

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


def concurrent_shuffler(split_list, seed_number_chosen):
    magnesium_value = (split_list.loc[:, 'Magnesium (mM)'])
    magnesium_list = (list(zip(magnesium_value, magnesium_value.index)))
    start_list = []
    start_index = 0
    end_list = []
    end_index = 0

    # list of current magnesium values
    current_list = [magnesium_value[mag] for mag in range(len(magnesium_value))]
    # list of next magnesium values
    next_list = [magnesium_value[mag + 1] for mag in range(len(magnesium_value) - 1)]

    current_list.insert(len(current_list), 0)
    next_list.insert(len(next_list), 0)
    combined_list = list(zip(current_list, next_list))

    all_index_df = pd.DataFrame()
    for i in range(len(combined_list)):
        list_instance = combined_list[i]
        current_magnesium = list_instance[0]
        next_magnesium = list_instance[1]
        # print(current_magnesium)
        if current_magnesium > next_magnesium:
            start_index = end_index
            end_index = i + 1
            if end_index - start_index > 1:
                # print(start_index, end_index-1)
                start_list.append(start_index)
                end_list.append(end_index)
                df = split_list[start_index:end_index].sample(frac=1, random_state=seed_number_chosen)
                all_index_df = pd.concat([all_index_df, df])
            else:
                df = split_list[start_index:end_index]
                all_index_df = pd.concat([all_index_df, df])
    # print("shuffled index: ", all_index_df.index)
    new_magnesium_value_list = all_index_df.iloc[:, 10]
    # print(all_index_df['Magnesium (mM)'])
    # print(list(zip(magnesium_value, new_magnesium_value_list)))
    # print(all_index_df['Magnesium (mM)'])
    return all_index_df


# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)


data_set_file_path = cwd + '/dot_file_data_set.csv'
data_set = pd.read_csv(data_set_file_path)

reps = 1
for i in range(reps):
    seed_number = seed_numbers[i]
    data_set_drop = data_set.drop(columns=[])
    data_set_sorted = data_set_drop.sort_values(['Magnesium (mM)'], ascending=False)
    data_set_sorted = data_set_sorted.reset_index()
    # print(data_set_sorted.columns)
    # print(data_set_sorted['Magnesium (mM)'])
    new_data_set = concurrent_shuffler(data_set_sorted, seed_number_chosen=seed_number)
    # print(new_data_set['Magnesium (mM)'])

    # apply round robin
    """
    round robin approach is to sort the Y variable, proceed to ensure that the data
    is put forwards to be split in even distribution.
    Step 1: Sort Y variable.
    Step 2: Create 3 folds before Inner Nested Loop.
    Step 3: Put the 3 even distributions into Inner K Fold.
    Gather the scores and data from the model and see the difference to random dist.
    """

    names = new_data_set.columns.values
    total_list = []

    data_set_used = new_data_set.drop(columns=['index'])
    for j in data_set_used.itertuples():
        total_list.append(j)

    total_y_list = []

    #  ### SET THIS TO FOLD COUNT ###  ##
    cv_count = 3
    #  ##############################  ##

    for cv_chosen in range(cv_count):
        y_list_cv_chosen = total_list[cv_chosen::cv_count]
        total_y_list = (total_y_list + y_list_cv_chosen)
    final_data_set = pd.DataFrame(total_y_list, columns=names)

    index = final_data_set['index']

    # Change the X and Y data, experiment name, scoring used
    x = final_data_set.drop(columns=['index', 'Unnamed: 0', 'Magnesium (mM)',
                                     'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                     'Acetic acid (mM)', 'Acetate (mM)'])
    y = final_data_set['Magnesium (mM)']
    print(y)
    experiment_name = "Extra_Trees_3CV_ReliefF_Stratified_RevNano"
    scoring = "r2"

    index_to_save = list(zip(index, y))
    # Actual Machine Learning Script for Regression Problem
    ETR = ExtraTreesRegressor(random_state=seed_number)

    # relief = ReliefF(n_features_to_select=2, n_neighbors=200)
    # Parameter Grid dictionary
    parameters = {}
    # Parameter Grid for ETR / RFR
    # parameters.update({'skr__n_features_to_select': [1, 2]})
    parameters.update({'estimator__n_estimators': [200]})
    parameters.update({'estimator__max_depth': [1]})

    # One-hot pipeline added
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', missing_values=np.NaN)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.NaN)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Stored lists of the numeric and categorical columns using the pandas dtype method.
    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x.select_dtypes(include=['object']).columns

    # Column transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Pipeline is called with model
    rf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('skr', ReliefF(n_features_to_select=1, verbose=True, n_neighbors=200)),
                         ('estimator', ETR)])

    # create folders to store meta data
    # important = (cwd + "/extra_trees_results" + "/model_plots/")
    # final_score_store = (cwd + "/extra_trees_results" + "/model_final_scores/")
    # diagnostic = (cwd + "/extra_trees_results" + "/model_metadata/")
    # pickle_store = (cwd + "/extra_trees_results" + "/model_pickle/"))

    outer_cv = KFold(n_splits=3, shuffle=False)
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

        # Inner CV(hyper-parameter optimisation)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
        est_used = GridSearchCV(estimator=rf, param_grid=parameters, cv=inner_cv, scoring=scoring)
        est_used.fit(X=X_train, y=y_train)
        # Get params for folds model
        fold_params = est_used.best_params_
        # Fit best model of GridSearchCV
        best = est_used.best_estimator_
        best.fit(X_train, y_train)

        # Create prediction and the score of that prediction
        pred = best.predict(X_test)
        score_r2 = r2_score(y_test, pred)
        scores.append(score_r2)
        # MAE
        mae_result = mean_absolute_error(y_test, pred)
        mae.append(mae_result)
        # MSE
        mse_result = mean_squared_error(y_test, pred)
        mse.append(mse_result)
        # RMSE
        rmse_result = np.sqrt(mse_result)
        rmse.append(rmse_result)
        # MedAE
        median_ae_result = median_absolute_error(y_test, pred)
        median_ae.append(median_ae_result)

        # store the fold scores
        fold_score = []
        fold_score.append(score_r2)
        fold_score.append(mae_result)
        fold_score.append(mse_result)
        fold_score.append(rmse_result)
        fold_score.append(median_ae_result)

    # Create any necessary averages of scores for nested models and print them
    final_score = np.average(scores)
    final_mae = np.average(mae)
    final_mse = np.average(mse)
    final_rmse = np.average(rmse)
    final_median_ae = np.average(median_ae)

    rep_final.append(final_score)
    rep_final.append(final_mae)
    rep_final.append(final_mse)
    rep_final.append(final_rmse)
    rep_final.append(final_median_ae)

    print(rep_final)
