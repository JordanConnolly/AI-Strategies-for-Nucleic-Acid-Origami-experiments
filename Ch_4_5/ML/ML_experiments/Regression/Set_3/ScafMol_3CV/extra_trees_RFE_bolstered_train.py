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


feature_selection_rfe = True
best_model_plot = True

# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)

# import the data set
data_set_file_path = cwd + '/subset_1_all_magnesium_0_25_yield_removed_literature_ml_data_set.csv'
data_set = pd.read_csv(data_set_file_path)
# Remove all Experiments with Anomalous Mg values
data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

# which rep is performed
script = Path(__file__).stem
rep = int(script.split("_")[::-1][0])
number_of_splits = 3

# Enter predictor of interest
predictor = 'Staple Molarity (nM)'

# Remove all Experiments with NaN Outcome
data_set = data_set[~data_set[predictor].isin([np.NaN])]

seed_number = seed_numbers[rep]
final_data_set, index = stratified_dataset(data_set, number_of_splits, predictor, seed_number,
                                           index_required=True)

original_y = final_data_set[predictor]

columns_to_drop_list = ['Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
                        'Experiment Number', 'Boric Acid (mM)',
                        'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                        'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                        'average_shortest_path', 'average_clustering_coefficient',
                        'average_degree', 'average_betweenness_centrality',
                        'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                        'graph_reciprocity', 's-metric', 'wiener_index']


# Change the X and Y data, experiment name, scoring used
x = final_data_set.drop(columns=columns_to_drop_list, errors='ignore')

# split the original data set instances and the bolstered instances
data_set_file_path_og = cwd + '/correct_imputation_magnesium_v3_ml_data_set_no_0_25.csv'
data_set_og = pd.read_csv(data_set_file_path_og)
# Remove all Experiments with NaN Outcome
data_set_og = data_set_og[~data_set_og[predictor].isin([np.NaN])]
data_set_og, index_2 = stratified_dataset(data_set_og, number_of_splits, predictor, seed_number,
                                          index_required=True)

# Remove all Experiments with Anomalous Mg values
og_data_set = data_set_og[~data_set_og['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
og_data_set = og_data_set.drop(columns=columns_to_drop_list, errors='ignore')

print("OG DATA SET:", og_data_set.shape)
print("OG DATA COLUMNS:", og_data_set.columns)

bolster_data_set = x[x['Paper Number'] >= 101]

# retrieve y values for the test sets
og_data_set_y = og_data_set[predictor]
bolster_data_set_y = bolster_data_set[predictor]


def corr_drop(ds_given, drop_value):
    # find correlated features to drop
    corr_matrix = ds_given.corr().abs()
    print(ds_given.shape)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
    were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
    return to_drop


# remove correlated values (calculated as if from all instances)
to_drop = corr_drop(x, 0.80)
og_data_set = og_data_set.drop(og_data_set[to_drop], axis=1)
bolster_data_set = bolster_data_set.drop(bolster_data_set[to_drop], axis=1)

# remove y values from X sets
og_data_set = og_data_set.drop(columns=["Paper Number", predictor], errors='ignore')
bolster_data_set = bolster_data_set.drop(columns=["Paper Number", predictor], errors='ignore')

# print the values
print("this is the og data set:", len(og_data_set))
print("this is the bolster data set:", len(bolster_data_set))

# you would then create the train AND test sets from the OG data set.
# you would then create a train only SPLIT of the bolster set (shuffle-split into n=3)
# to do this you could just use the train-test split and combine the tr-val index values, add to the OG.

y = final_data_set[predictor]
experiment_name = "Extra_Trees_RFE_" + str(number_of_splits) + "CV_Stratified_Baseline"
scoring = "r2"

index_to_save = list(zip(index, original_y))

# Actual Machine Learning Script for Regression Problem
ETR = ExtraTreesRegressor(random_state=seed_number, n_jobs=-1)

# Parameter Grid dictionary
parameters = {}
# Parameter Grid for ETR / RFR
parameters.update({'estimator__n_estimators': [10, 100, 200, 500]})
parameters.update({'estimator__max_depth': [None, 1, 2, 3, 4, 5]})

# Parameter Grid for RFE
parameters.update({"rfe__n_features_to_select": [1, 5, 10, 20, 25]})


# # Recursive Elimination for Regression Models
estimator = ExtraTreesRegressor(random_state=seed_number, n_jobs=-1)
print(estimator.get_params().keys())
recurse = RFE(estimator=estimator, step=0.1)  # Set step to 0-1 for percentage of features removed per iter

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
numeric_features = bolster_data_set.select_dtypes(include=['int64', 'float64']).columns
categorical_features = bolster_data_set.select_dtypes(include=['object']).columns

# Column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Pipeline is called with model
rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('rfe', recurse),
                     ('estimator', ETR)])

# create folders to store meta data
important = (cwd + "/extra_trees_results" + "/model_plots/")
final_score_store = (cwd + "/extra_trees_results" + "/model_final_scores/")
diagnostic = (cwd + "/extra_trees_results" + "/model_metadata/")
pickle_store = (cwd + "/extra_trees_results" + "/model_pickle/")

# Save Index for Data Set
index_to_save_df = pd.DataFrame(index_to_save)
index_to_save_df.to_csv(diagnostic + experiment_name + "_rep_" + str(rep) + "_index_values.csv")

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

outer_cv = KFold(n_splits=number_of_splits, shuffle=False)

data_splits = list(outer_cv.split(og_data_set, og_data_set_y))
data_splits_2 = list(outer_cv.split(bolster_data_set, bolster_data_set_y))

for i in range(number_of_splits):
    inner_loop_rep += 1

    # split the og data to create test set, and the train set for combining with bolster
    og_data_split = data_splits[i]
    og_train_idx = og_data_split[0]
    og_test_idx = og_data_split[1]
    og_X_train, og_y_train = og_data_set.iloc[og_train_idx], og_data_set_y.iloc[og_train_idx]
    X_test, y_test = og_data_set.iloc[og_test_idx], og_data_set_y.iloc[og_test_idx]

    # split the bolstered data to combined with the og train set
    bolster_data_split = data_splits_2[i]
    bolster_train_idx = bolster_data_split[0]
    bolster_test_idx = bolster_data_split[1]
    bolster_X_train, bolster_y_train = bolster_data_set.iloc[bolster_train_idx], \
                                       bolster_data_set_y.iloc[bolster_train_idx]
    bolster_X_test, bolster_y_test = bolster_data_set.iloc[bolster_test_idx], \
                                     bolster_data_set_y.iloc[bolster_test_idx]

    # combine the bolstered into one large set to combine
    concat_bolster = pd.concat([bolster_X_train, bolster_X_test])
    concat_bolster_y = pd.concat([bolster_y_train, bolster_y_test])

    # combine the large set with original train data
    X_train = pd.concat([og_X_train, concat_bolster])
    y_train = pd.concat([og_y_train, concat_bolster_y])

    print(len(X_train))
    print(len(y_train))

    print(len(X_test))
    print(len(y_test))

    # Store Y values of split -- DIAGNOSTIC --
    y_data = list(zip(y_train, y_test))
    y_data_store = pd.DataFrame(y_data)
    y_data_store.to_csv(diagnostic + experiment_name + "_rep_" + str(rep) + "_fold_"
                        + str(inner_loop_rep) + "_y_values_split.csv")

    # Inner CV(hyper-parameter optimisation)
    inner_cv = KFold(n_splits=number_of_splits, shuffle=True, random_state=1)
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

    # Store folds best model of inner grid search -- DIAGNOSTIC --
    with open(diagnostic + experiment_name +
              "_best_inner_model_parameters_" + str(rep) + "_fold_"
              + str(inner_loop_rep) + ".txt", "a", newline='') as file:
        file.write(f'{best}')
        file.close()

    # Store the best cross-validation fold estimator -- DIAGNOSTIC --
    with open(pickle_store + experiment_name + "saved_model_" + str(rep) + "_fold_"
              + str(inner_loop_rep) + '.pkl', 'wb'
              ) as fid:
        pickle.dump(best, fid)

    # Save separate score results to files -- DIAGNOSTIC --
    with open(diagnostic + experiment_name + "_rep_" + str(rep) + "_fold_" + str(inner_loop_rep) +
              "_score.csv", "a", newline='') as file:
        writer = csv.writer(file, quoting=0)
        writer.writerow(fold_score)
    file.close()  # Closes text file

    # Create actual vs predictions lists
    pred_list = list(pred)
    real_list = list(y_test)
    predictions += pred_list
    reality += real_list

    fold_actual_vs_pred = list(zip(pred_list, real_list))
    fold_actual_vs_pred_df = pd.DataFrame(fold_actual_vs_pred, columns=['prediction', 'reality'])
    fold_actual_vs_pred_df.to_csv(diagnostic + experiment_name +
                                  "_actual_vs_pred_rep_" + str(rep) + "_fold_" + str(inner_loop_rep) + ".csv")

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

total_actual_vs_pred = list(zip(predictions, reality))
total_actual_vs_pred_df = pd.DataFrame(total_actual_vs_pred, columns=['prediction', 'reality'])
total_actual_vs_pred_df.to_csv(important + experiment_name +
                               "_actual_vs_pred_rep_" + str(rep) + "_best_all_folds.csv")

# Call a search on whole data set to produce the best model
best_outer = GridSearchCV(estimator=rf, param_grid=parameters, scoring=scoring)
best_outer.fit(X=x, y=y)
best_all = best_outer.best_estimator_

# Store the best whole dataset estimator -- WHOLE MODEL --
with open(pickle_store + experiment_name + "saved_model_" + str(rep) + "_all_data" + '.pkl', 'wb') as fid:
    pickle.dump(best_all, fid)

# Store best whole data set grid search parameters -- DIAGNOSTIC --
with open(important + experiment_name +
          "_best_whole_dataset_parameters_" + str(rep) + ".txt", "a", newline='') as file:
    file.write(f'{best_all}')
    file.close()  # Closes text file

# Save separate score results to files
with open(diagnostic + experiment_name + "_rep_" + str(rep) + "_final_score.csv", "a", newline='') as file:
    writer = csv.writer(file, quoting=0)
    writer.writerow(rep_final)
file.close()  # Closes text file

# Save all score results to a file
with open(final_score_store + experiment_name + "_rep_" + str(rep) + "_final_score.csv", "a", newline='') as file:
    writer = csv.writer(file, quoting=0)
    writer.writerow(rep_final)
file.close()  # Closes text file
