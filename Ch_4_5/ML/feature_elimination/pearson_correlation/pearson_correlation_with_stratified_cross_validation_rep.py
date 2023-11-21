import random
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
Custom exploration of removal of features using Pearson Correlation if features are redundant.
/// 
Take Metrics and produce an external plot creating python file;
 X = iteration, y = scoring metric, test line = test metrics, train line = train metrics 
///
'''

# Pandas and Numpy Options
cwd = os.getcwd()
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

# Set Random Seed from list of shuffled seed numbers
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)


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


def pearson_correlation_feature_removal(data, drop_value):
    # drop correlated features
    drop_value = drop_value
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
    were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
    final_data_set = data_set.drop(data_set[to_drop], axis=1)
    return final_data_set


# import the data set
data_set_file_path = cwd + '/dot_file_data_set.csv'
data_set = pd.read_csv(data_set_file_path)
# x = data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
#                            'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
#                            'Acetic acid (mM)', 'Acetate (mM)'])
x = data_set.drop(columns=['Magnesium (mM)'])
y = data_set['Magnesium (mM)']

# Use algorithm
ETR = ExtraTreesRegressor(random_state=42, n_estimators=2000, max_depth=None)


def preprocess_data(train, test):
    # Pre-process data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', missing_values=np.NaN)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.NaN)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Stored lists of the numeric and categorical columns using the pandas dtype method.
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train.select_dtypes(include=['object']).columns

    # Column transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # combine it all together using sci-kit pipeline
    preprocess = Pipeline(steps=[('preprocessor', preprocessor)])

    # apply pre-processing above to train and test
    X_train_prep = preprocess.fit_transform(train)
    X_test_prep = preprocess.transform(test)

    # return the X_train and X_test preprocessed
    return X_train_prep, X_test_prep


# count features to remove
features_removed = 0

# Use cross val
cv_chosen = 3

# Create ability to produce a final data frame
final_data_frame = pd.DataFrame()
all_train_scores = []
all_test_scores = []
removed_features_list = []


outer_feature_importance_cv = []
average_train_score = []
average_test_score = []
reps = 10
features_removed += 1

for i in range(reps):
    seed_number = seed_numbers[i]
    # x = final_data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
    #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
    #                                  'Acetic acid (mM)', 'Acetate (mM)'])
    drop_value_list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    y = data_set['Magnesium (mM)']
    final_data_set = pearson_correlation_feature_removal(data_set, drop_value_list[i])
    x = final_data_set.drop(columns=['Magnesium (mM)'], errors='ignore')

    outer_cv = KFold(n_splits=cv_chosen, shuffle=False)
    data_splits = list(outer_cv.split(x, y))
    print(i)
    for tr_idx, val_idx in data_splits:
        # Create test and train sets for inner nest loop
        X_train, y_train = x.iloc[tr_idx], y[tr_idx]
        X_test, y_test = x.iloc[val_idx], y[val_idx]

        X_train_prep, X_test_prep = preprocess_data(X_train, X_test)

        # fit data to model for first removal
        ETR.fit(X_train_prep, y_train)
        # predict with model
        outer_pred = ETR.predict(X_test_prep)

        # Train Score to Save
        train_score = ETR.score(X_train_prep, y_train)
        average_train_score.append(train_score)
        print(train_score)

        # Test Score to Save
        outer_score = r2_score(outer_pred, y_test)
        average_test_score.append(outer_score)
        print(outer_score)
