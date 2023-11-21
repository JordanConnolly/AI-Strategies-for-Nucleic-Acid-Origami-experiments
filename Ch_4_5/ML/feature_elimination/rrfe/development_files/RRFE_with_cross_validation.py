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

'''
Custom RFE to remove best feature at each iteration,
to explore if features are redundant.

/// LOOP START ///
Nested Loop: Static Hyper-parameters (Extra Trees Regression such as n_estimators = 200, depth = default)
Outer CV -> Quick Iteration with a small CV (CV = 3)
/--> Inner CV -> Training set, Extra Trees, Extract Importance Vectors, Best Feature Removed -
- Extract metadata, such as test and train metrics
/// LOOP END ///

/// 
Take Metrics and produce an external plot creating python file;
 X = iteration, y = scoring metric, test line = test metrics, train line = train metrics 
///
'''


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

data_set_file_path = cwd + '/dot_file_data_set.csv'
data_set = pd.read_csv(data_set_file_path)

x = data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
                           'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                           'Acetic acid (mM)', 'Acetate (mM)'])
y = data_set['Magnesium (mM)']

# # remove 10% of NaN values
# column_length = len(x.columns)
# ten_percent_nan_removed = round((column_length / 100) * 90)
# x = x.dropna(thresh=ten_percent_nan_removed, axis=1)
# print(x.shape)

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
numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x.select_dtypes(include=['object']).columns

# x = x.drop(columns=categorical_features)
# print(x.shape)

# Column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ])

# Use algorithm
ETR = ExtraTreesRegressor(random_state=42, n_estimators=500)
preprocess = Pipeline(steps=[('preprocessor', preprocessor)])


def preprocess_data(train, test):
    X_train_prep = preprocess.fit_transform(train)
    X_test_prep = preprocess.transform(test)
    return X_train_prep, X_test_prep


removed_column_name = []
# count features to remove
features_removed = 0
# Use cross val
cv_chosen = 3

outer_cv = KFold(n_splits=cv_chosen, shuffle=False, random_state=42)
data_splits = list(outer_cv.split(x, y))
outer_feature_importance_cv = []

# Use test-train split
get_X_train, get_X_test, get_y_train, get_y_test = train_test_split(x, y, test_size=0.33, random_state=42)
get_X_train_prep, get_X_test_prep = preprocess_data(get_X_train, get_X_test)
# Gather column names
x_transformed_columns_outer = get_transformer_feature_names(preprocess['preprocessor'])
df_test = pd.DataFrame(get_X_train_prep, columns=x_transformed_columns_outer)
cols = list(df_test.columns)


for tr_idx, val_idx in data_splits:
    print("First model fit")
    # Create test and train sets for inner nest loop
    X_train, y_train = x.iloc[tr_idx], y[tr_idx]
    X_test, y_test = x.iloc[val_idx], y[val_idx]
    # retain the removed columns / features names
    # decide which features are removed and store here for use after each iteration
    X_train_prep, X_test_prep = preprocess_data(X_train, X_test)

    # Gather column names
    x_transformed_columns_outer = get_transformer_feature_names(preprocess['preprocessor'])
    df_test = pd.DataFrame(X_train_prep, columns=x_transformed_columns_outer)
    cols = list(df_test.columns)
    len_cols = len(cols)

    for n in range(0, 1):
        # fit data to model for first removal
        ETR.fit(X_train_prep, y_train)
        # predict with model
        outer_pred = ETR.predict(X_test_prep)
        # score model
        outer_score = r2_score(outer_pred, y_test)
        print(outer_score)
        # Feature Importance
        outer_feature_importance = ETR.feature_importances_
        outer_importance_vectors = pd.Series(outer_feature_importance, index=cols)
        outer_feature_with_max = outer_importance_vectors.idxmax()
        outer_feature_importance_cv.append(outer_importance_vectors)
        features_removed += 1
outer_feature_importance_cv = pd.DataFrame(outer_feature_importance_cv)
outer_feature_importance_cv = outer_feature_importance_cv.mean()
outer_feature_with_max = outer_feature_importance_cv.idxmax()
removed_column_name.append(outer_feature_with_max)
print(removed_column_name)

# backwards loop through all remaining features, removing the most informative per cross-validation
for n_inner in range(0, len(cols)-features_removed):
    inner_feature_importance_cv = []
    for tr_idx, val_idx in data_splits:
        # Create test and train sets for inner nest loop
        X_train, y_train = x.iloc[tr_idx], y[tr_idx]
        X_test, y_test = x.iloc[val_idx], y[val_idx]
        # retain the removed columns / features names
        # decide which features are removed and store here for use after each iteration
        X_train_prep, X_test_prep = preprocess_data(X_train, X_test)
        # Gather column names
        x_transformed_columns_outer = get_transformer_feature_names(preprocess['preprocessor'])
        df_test = pd.DataFrame(X_train_prep, columns=x_transformed_columns_outer)
        cols = list(df_test.columns)

        X_train_prep_df = pd.DataFrame(X_train_prep, columns=x_transformed_columns_outer)
        X_test_prep_df = pd.DataFrame(X_test_prep, columns=x_transformed_columns_outer)
        # errors is set to ignore, which is due to the fact TEST TRAIN SPLIT may not contain columns
        X_train_prep_inner = X_train_prep_df.drop(columns=removed_column_name, axis=1, errors='ignore')
        X_test_prep_inner = X_test_prep_df.drop(columns=removed_column_name, axis=1, errors='ignore')

        # fit data to model for first removal
        ETR.fit(X_train_prep_inner, y_train)
        # predict with model
        outer_pred = ETR.predict(X_test_prep_inner)
        # score model
        outer_score = r2_score(outer_pred, y_test)
        print(outer_score)

        # Feature Importance
        inner_feature_importance = ETR.feature_importances_
        remaining_features = [i for i in x_transformed_columns_outer if i not in removed_column_name]
        inner_importance_vectors = pd.Series(inner_feature_importance, index=remaining_features)
        inner_feature_with_max = inner_importance_vectors.idxmax()
        print(outer_feature_with_max)
        inner_feature_importance_cv.append(inner_importance_vectors)

    inner_feature_importance_cv = pd.DataFrame(inner_feature_importance_cv)
    print(inner_feature_importance_cv)
    inner_feature_importance_cv = inner_feature_importance_cv.mean()
    inner_feature_with_max = inner_feature_importance_cv.idxmax()
    removed_column_name.append(inner_feature_with_max)
    print(removed_column_name)
