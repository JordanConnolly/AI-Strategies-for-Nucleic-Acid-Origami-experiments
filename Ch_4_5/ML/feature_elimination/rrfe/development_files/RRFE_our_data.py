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

# remove 10% of NaN values
column_length = len(x.columns)
ten_percent_nan_removed = round((column_length / 100) * 90)
x = x.dropna(thresh=ten_percent_nan_removed, axis=1)
print(x.shape)

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

x = x.drop(columns=categorical_features)
print(x.shape)

# Column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        # ('cat', categorical_transformer, categorical_features)
        ])

# Use algorithm
ETR = ExtraTreesRegressor(random_state=42, n_estimators=500)
preprocess = Pipeline(steps=[('preprocessor', preprocessor)])

# Use cross val
outer_cv = KFold(n_splits=3, shuffle=False)
data_splits = list(outer_cv.split(x, y))

# Scores
rep_final = []

# Actual vs Prediction lists
predictions = []
reality = []

# iterate over these
num_features = x.shape[1]
idx = np.arange(0, x.shape[1])
columns_remain = []

for features in range(num_features):
    scores = []
    r2 = []
    mae = []
    mse = []
    rmse = []
    median_ae = []
    inner_loop_rep = 0
    columns_remain = []
    all_importance_vectors = pd.DataFrame(columns_remain)
    x = x.iloc[:, idx]
    # Creates data splits
    features_remain = num_features - features
    # print(features_remain)

    for tr_idx, val_idx in data_splits:
        inner_loop_rep += 1

        # Create test and train sets for inner nest loop
        X_train, y_train = x.iloc[tr_idx], y[tr_idx]
        X_test, y_test = x.iloc[val_idx], y[val_idx]

        ETR.fit(X=X_train, y=y_train)

        # Create prediction and the score of that prediction
        pred = ETR.predict(X_test)

        # r2
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
        print("fold score:", fold_score)

        # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
        feature_importance = ETR.feature_importances_
        transformed_data = preprocess.named_steps['preprocessor'].transform(x)
        # Gather column names
        x_transformed_columns_outer = get_transformer_feature_names(preprocess['preprocessor'])
        df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)

        importance_vectors = pd.DataFrame()
        for feature_range in range(features_remain):
            importance_vectors[inner_loop_rep] = feature_importance
        all_importance_vectors = pd.concat([all_importance_vectors, importance_vectors], axis=1)

    print(all_importance_vectors)
    x_transformed = preprocess.fit_transform(x)
    idx_transformed = np.arange(0, x_transformed.shape[1])
    print(idx_transformed.shape)
    average_importance = all_importance_vectors.apply(np.average, axis=1)
    print(average_importance.shape)
    idx = idx_transformed[average_importance < average_importance.max()]
    print(len(idx))
    X_columns_remain = x.iloc[:, idx].columns
    columns_remain.append(X_columns_remain)

# we have a problem here, the importance will not work as the columns do not match
# the columns do not match because different splits will contain different categories
# one hot categories will differ across splits
# Fixes include: Making sure splits are even, removing few samples or create a system to average regardless
