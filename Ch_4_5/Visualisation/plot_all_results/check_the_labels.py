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

from datetime import datetime
startTime = datetime.now()

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


feature_selection_rfe = True
best_model_plot = True

# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)

# import the data set
# data_set_file_path = cwd + '/correct_imputation_magnesium_v3_ml_data_set_no_0_25.csv'
# data_set_file_path = cwd + '/subset_1_all_literature_high_cardinal_removed_ml_data_set.csv'
# data_set_file_path = cwd + '/dot_file_data_set.csv'
# data_set_file_path = cwd + '/dot_file_data_set_bolstered.csv'
data_set_file_path = cwd + '/100_literature_ml_data_set.csv'
data_set = pd.read_csv(data_set_file_path)

# remove 1 / 13 instances of thermal profile that are truly isothermal
data_set = data_set[~data_set['Thermal Profile'].isin(['Isothermal-without-initial-denaturation'])]

# which rep is performed
script = Path(__file__).stem
number_of_splits = 3

# Enter predictor of interest
predictor = 'Thermal Profile'

# Remove all Experiments with NaN Outcome
final_data_set = data_set[~data_set[predictor].isin([np.NaN])]

# ordinal encode target variable
label_encoder = LabelEncoder()
y = final_data_set[predictor]
y = label_encoder.fit_transform(y)
final_data_set['y'] = y
original_class_labels = list(label_encoder.classes_)
print(f"0 label is: {original_class_labels[0]}\n1 label is: {original_class_labels[1]}")
