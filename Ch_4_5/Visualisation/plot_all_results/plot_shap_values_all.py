import random
import pandas as pd
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import os
import re
import glob
import matplotlib.pyplot as plt
from textwrap import wrap

random.seed(3)
reps = 90
random_seed_list = []
for i in range(reps):
    random_seed_list.append(random.randrange(1000))

# Directories containing files
cwd = os.getcwd()
path = "C:/Users/Big JC/PycharmProjects/PhDCodeOnly/ML_results_to_analyse/" \
       "RevNanoStratified/extra_trees_results/model_metadata/"
experiment_name = "Extra_Trees_3CV_Stratified_RevNano"
chosen_value = 29


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


numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


shap_files = sorted(glob.glob(path + experiment_name + "_SHAP_values_for_plot_" + "*" + "_fold_" +
                              "*.csv"), key=numerical_sort)
print(len(shap_files))
# import data set used values
data_set_file_path = cwd + "/dot_file_data_set.csv"
data_set = pd.read_csv(data_set_file_path).reset_index()
# Remove all Experiments with Anomalous Mg values
data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
# Remove all Experiments with Anomalous Mg values
# data_set = data_set[~data_set['Paper Number'].isin(['48'])]

x = data_set.drop(columns=['Magnesium (mM)', 'index',
                           'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                           'Acetic acid (mM)', 'Acetate (mM)'])

outer_i = 0
for i in range(reps):
    # Transform Categorical and Numerical Features
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

    outer_cv = KFold(n_splits=3, shuffle=True, random_state=random_seed_list[i])
    data_splits = list(outer_cv.split(x))
    #  Creates data splits
    inner_loop_rep = 0
    for tr_idx, val_idx in data_splits:
        inner_loop_rep += 1
        # Removes correct answers from New_X, creates train-test sets
        X_train = x.iloc[tr_idx]
        X_test = x.iloc[val_idx]
        # Create processed X for creating shap plot
        transformed_data = preprocessor.fit_transform(x)
        # Access the shap values created in experiment
        file = shap_files[outer_i]
        outer_i += 1
        print(file)
        path, filename = os.path.split(file)
        print(filename)
        # Import the shap values and feature names from the glob found file
        shap_values_df = pd.read_csv(file)
        shap_values_df = shap_values_df.drop(columns="Unnamed: 0")
        # print(shap_values_df)
        shap_values_columns = shap_values_df.columns
        features_named = shap_values_columns.tolist()
        # print(features_named)
        # Create instances data frame to calculate summary plot with
        x_transformed_columns_outer = get_transformer_feature_names(preprocessor)
        df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
        # print(df_test)
        df_test_shap = df_test[features_named]
        # print(df_test_shap)
        # Apply SHAP
        shap_values = shap_values_df.values
        # print(shap_values)
        if i is chosen_value:
            shap.summary_plot(shap_values, df_test_shap,
                              feature_names=features_named, show=False)
            title = (experiment_name + " shap summary plot experiment " + str(i + 1)
                     + " fold " + str(inner_loop_rep))
            plt.title("\n".join(wrap(title, 44)))
            plt.show()
