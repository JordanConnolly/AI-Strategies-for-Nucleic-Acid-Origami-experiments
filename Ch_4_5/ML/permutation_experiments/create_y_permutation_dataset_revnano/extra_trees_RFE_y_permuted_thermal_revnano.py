import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectKBest, f_regression, chi2
import csv
import shap
import random
import pickle
from sklearn.metrics import plot_roc_curve
import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
from datetime import datetime
startTime = datetime.now()


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


# Pandas and Numpy Options
cwd = os.getcwd()
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

# which rep is performed
script = Path(__file__).stem
rep = int(script.split("_")[::-1][0])

# get config file for permutation algorithm experiments
hyper_parameters_dict = {}
with open('permutation_algorithm_config_used.json') as json_file:
    parsed = json.load(json_file)
    # print(json.dumps(parsed, indent=4, sort_keys=True))
    for i in parsed.get("ml_hyper_params"):
        hyper_parameters_dict = i
    json_file.close()

list_of_hyper_params = ["description", "dataset_filename", "splits", "experiment_name", "predictor", "scoring"]

# get the hyper_parameters from loaded json config file
data_set_file_path = hyper_parameters_dict.get("dataset_filename")
number_of_splits = hyper_parameters_dict.get("splits")
experiment_name = hyper_parameters_dict.get("experiment_name")
predictor = hyper_parameters_dict.get("predictor")
scoring = hyper_parameters_dict.get("scoring")

# store the hyper_parameters locally
storage_path = "./extra_trees_results/"
with open(storage_path + 'permutation_algorithm_config_used.json', "w") as json_file:
    json_file.write(json.dumps(hyper_parameters_dict, indent=4, sort_keys=False))
    json_file.close()

feature_selection_rfe = True
best_model_plot = True

# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)

# import the data set
data_set = pd.read_csv(cwd + data_set_file_path)
seed_number = seed_numbers[rep]

# split the data
x = data_set.drop(columns=['y', "Unnamed: 0", "index"], errors='ignore')
y = data_set['y']

# Begin the Machine Learning Script
# Actual Machine Learning Script for Regression Problem
ETC = ExtraTreesClassifier(random_state=seed_number, n_jobs=-1)

# Parameter Grid dictionary
parameters = {}
# Parameter Grid for ETR / RFR
parameters.update({'estimator__n_estimators': [10, 100, 200, 500]})
parameters.update({'estimator__max_depth': [None, 1, 2, 3, 4, 5]})
# Parameter Grid for RFE
parameters.update({'estimator__class_weight': ["balanced"]})
parameters.update({"rfe__n_features_to_select": [1, 5, 10, 20, 25]})

# # Recursive Elimination for Regression Models
estimator = ExtraTreesClassifier(random_state=seed_number, n_jobs=-1)
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
numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x.select_dtypes(include=['object']).columns

# Column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Pipeline is called with model
rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('rfe', recurse),
                     ('estimator', ETC)])

# create folders to store meta data
important = (cwd + "/extra_trees_results/model_plots/")
final_score_store = (cwd + "/extra_trees_results/model_final_scores/")
diagnostic = (cwd + "/extra_trees_results/model_metadata/")
pickle_store = (cwd + "/extra_trees_results/model_pickle/")

outer_cv = StratifiedKFold(n_splits=number_of_splits, shuffle=False)
data_splits = list(outer_cv.split(x, y))
# Scores
rep_final = []
scores = []
score_accuracy = []
score_precision = []
score_recall = []
score_f1 = []
score_roc_auc = []

# Actual vs Prediction lists
predictions = []
reality = []
inner_loop_rep = 0

# ROC AUC CURVE
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# Confusion Matrix
y_actual_list = []
y_pred_list = []

#  Creates data splits
for tr_idx, val_idx in data_splits:
    inner_loop_rep += 1

    # Removes correct answers from New_X, creates train-test sets
    X_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
    X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

    # Store Y values of split -- DIAGNOSTIC --
    y_data = list(zip(y_train, y_test))
    y_data_store = pd.DataFrame(y_train, y_test)
    # y_data_store.to_csv(diagnostic + experiment_name + "_rep_" + str(i + 1) + "_fold_"
    #                     + str(inner_loop_rep) + "_y_values_split.csv")

    # Inner CV(hyper-parameter optimisation)
    inner_cv = StratifiedKFold(n_splits=number_of_splits, shuffle=True, random_state=1)
    est_used = GridSearchCV(estimator=rf, param_grid=parameters, cv=inner_cv, scoring=scoring)
    est_used.fit(X=X_train, y=y_train)
    # Get params for folds model
    fold_params = est_used.best_params_
    # Fit best model of GridSearchCV
    best = est_used.best_estimator_
    best.fit(X_train, y_train)

    # Create prediction and the score of that prediction
    pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)

    # Create scoring metrics and append for folds
    accuracy_result = accuracy_score(y_test, pred)
    scores.append(accuracy_result)
    score_accuracy.append(accuracy_result)
    # Precision Score
    precision_result = precision_score(y_test, pred)
    scores.append(precision_result)
    score_precision.append(precision_result)
    # Recall Score
    recall_result = recall_score(y_test, pred)
    scores.append(recall_result)
    score_recall.append(recall_result)
    # F1 Score
    f1_result = f1_score(y_test, pred)
    scores.append(f1_result)
    score_f1.append(f1_result)
    # ROC AUC Score
    roc_auc_result = roc_auc_score(y_test, pred)
    scores.append(roc_auc_result)
    score_roc_auc.append(roc_auc_result)

    # Store the fold scores
    fold_score = [accuracy_result, precision_result, recall_result, f1_result, roc_auc_result]
    print(fold_score)

    # Create actual vs predictions lists
    pred_list = list(pred)
    real_list = list(y_test)
    predictions += pred_list
    reality += real_list

    # Store folds best model of inner grid search -- DIAGNOSTIC --
    with open(diagnostic + experiment_name +
              "_best_inner_model_parameters_" + str(rep) + "_fold_"
              + str(inner_loop_rep) + ".txt", "a", newline='') as file:
        file.write(f'{best}')
        file.close()

    # Store the best cross-validation fold estimator -- DIAGNOSTIC --
    # with open(pickle_store + experiment_name + "saved_model_" + str(rep) + "_fold_"
    #           + str(inner_loop_rep) + '.pkl', 'wb'
    #           ) as fid:
    #     pickle.dump(best, fid)

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
final_accuracy = np.average(score_accuracy)
final_precision = np.average(score_precision)
final_recall = np.average(score_recall)
final_f1 = np.average(score_f1)
final_roc_auc = np.average(score_roc_auc)
print("Final Score:", final_accuracy)

# Sum averages into a "final score"
rep_final = [final_accuracy, final_precision, final_recall, final_f1, final_roc_auc]
print("All Final Scores:", rep_final)

total_actual_vs_pred = list(zip(predictions, reality))
total_actual_vs_pred_df = pd.DataFrame(total_actual_vs_pred, columns=['prediction', 'reality'])

# # Call a search on whole data set to produce the best model
# clf_all = GridSearchCV(estimator=rf, param_grid=parameters, scoring=scoring)
# clf_all.fit(X=x, y=y)
# best_all = clf_all.best_estimator_
#
# # Store the best whole dataset estimator -- WHOLE MODEL --
# with open(pickle_store + experiment_name + "saved_model_" + str(rep) + "_all_data" + '.pkl', 'wb') as fid:
#     pickle.dump(best_all, fid)

# # Store best whole data set grid search parameters -- DIAGNOSTIC --
# with open(important + experiment_name +
#           "_best_whole_dataset_parameters_" + str(rep) + ".txt", "a", newline='') as file:
#     file.write(f'{best_all}')
#     file.close()  # Closes text file

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
