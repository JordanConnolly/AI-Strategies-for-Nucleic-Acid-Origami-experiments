import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

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

# generate regression data set
X, y = make_regression(n_samples=20, n_features=5, noise=0.2)

columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
X = pd.DataFrame(X, columns=columns)
print(X.columns)

# Use algorithm
ETR = ExtraTreesRegressor(random_state=42, n_estimators=500)

# Use Cross val
outer_cv = KFold(n_splits=3, shuffle=False)
data_splits = list(outer_cv.split(X, y))

# Scores
rep_final = []

# Actual vs Prediction lists
predictions = []
reality = []

# iterate over these
num_features = X.shape[1]
idx = np.arange(0, X.shape[1])
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
    importance_vectors = pd.DataFrame(columns_remain)
    X = X.iloc[:, idx]
    # Creates data splits
    features_remain = num_features - features
    print(features_remain)
    for tr_idx, val_idx in data_splits:
        inner_loop_rep += 1

        # Create test and train sets for inner nest loop
        X_train, y_train = X.iloc[tr_idx], y[tr_idx]
        X_test, y_test = X.iloc[val_idx], y[val_idx]

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
        print(feature_importance)

        for feature_range in range(features_remain):
            importance_vectors[inner_loop_rep] = feature_importance

    idx = np.arange(0, X.shape[1])  # create an index array, with the number of features
    average_importance = importance_vectors.apply(np.average, axis=1)
    idx = idx[average_importance < average_importance.max()]
    X_columns_remain = X.iloc[:, idx].columns
    columns_remain.append(X_columns_remain)
