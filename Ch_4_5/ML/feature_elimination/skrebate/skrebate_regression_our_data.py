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

'''
Usage Information:
https://epistasislab.github.io/scikit-rebate/using/
'''

# Pandas and Numpy Options
cwd = os.getcwd()
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

data_set_file_path = cwd + '/dot_file_data_set.csv'
data_set = pd.read_csv(data_set_file_path)

x = data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
                           'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                           'Acetic acid (mM)', 'Acetate (mM)'])
y = data_set['Magnesium (mM)']

# Pre-process data
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

# Get ReliefF
relief = ReliefF(n_features_to_select=5, verbose=True, n_neighbors=200)

# Use Algorithm
ETR = ExtraTreesRegressor(random_state=42)

parameters = {}
# Parameter Grid for ETR / RFR
# parameters.update({'skr__n_features_to_select': [1, 2]})
parameters.update({'estimator__n_estimators': [200]})
parameters.update({'estimator__max_depth': [1]})

# Pipeline is called with model
rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('skr', relief),
                     ('estimator', ETR)])
scoring = 'r2'

# Use Cross_val_score
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

