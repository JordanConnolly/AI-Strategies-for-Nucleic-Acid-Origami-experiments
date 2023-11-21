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
# x = data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
#                            'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
#                            'Acetic acid (mM)', 'Acetate (mM)'])
y = data_set['Magnesium (mM)']
x = data_set.drop(columns=['Magnesium (mM)', 'Unnamed: 0',
                           'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                           'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                           'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                           'average_shortest_path', 'average_clustering_coefficient',
                           'average_degree', 'average_betweenness_centrality',
                           'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                           'graph_reciprocity', 's-metric', 'wiener_index'])

# data_set_file_path = cwd + '/correct_imputation_magnesium_v3_ml_data_set_no_0_25.xlsx'
# data_set = pd.read_excel(data_set_file_path)
# data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
# x = data_set.drop(columns=['Magnesium (mM)',
#                            'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
#                            'Acetic acid (mM)', 'Acetate (mM)'])
# y = data_set['Magnesium (mM)']


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

# Use test-train split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


def preprocess_data(train, test):
    X_train_prep = preprocess.fit_transform(train)
    X_test_prep = preprocess.transform(test)
    return X_train_prep, X_test_prep


# retain the removed columns / features names
removed_column_name = []
# count features to remove
features_removed = 0
# decide which features are removed and store here for use after each iteration
X_train_prep, X_test_prep = preprocess_data(X_train, X_test)
# Gather column names
x_transformed_columns_outer = get_transformer_feature_names(preprocess['preprocessor'])
df_test = pd.DataFrame(X_train_prep, columns=x_transformed_columns_outer)
cols = list(df_test.columns)

final_data_frame = pd.DataFrame()
all_train_scores = []
all_test_scores = []
removed_features_list = []
number_of_features_removed_list = []

# iterate over the features
for n_inner in range(0, len(cols) - features_removed):
    # Create data frames of original test-train split, remove columns and fit to model
    X_train_prep_df = pd.DataFrame(X_train_prep, columns=x_transformed_columns_outer)
    X_test_prep_df = pd.DataFrame(X_test_prep, columns=x_transformed_columns_outer)
    X_train_prep_inner = X_train_prep_df.drop(columns=removed_column_name, axis=1, errors='ignore')
    X_test_prep_inner = X_test_prep_df.drop(columns=removed_column_name, axis=1, errors='ignore')

    # Fit inner model
    ETR.fit(X_train_prep_inner, y_train)

    # Train Score to Save
    train_score = ETR.score(X_train_prep_inner, y_train)
    print("Train Score:", train_score)

    # Predict with inner model
    pred = ETR.predict(X_test_prep_inner)

    # Test Score to save
    test_score = r2_score(pred, y_test)
    print("Test Score:", test_score)

    # Feature Importance
    feature_importance = ETR.feature_importances_

    # Using list comprehension to get remaining features
    remaining_features = [i for i in x_transformed_columns_outer if i not in removed_column_name]

    # Gather column names
    df_test = pd.DataFrame(X_train_prep_inner, columns=remaining_features)
    inner_importance_vectors = pd.Series(feature_importance, index=remaining_features)

    # Column / feature name of highest feature importance value
    inner_feature_with_max = inner_importance_vectors.idxmax()

    # Highest feature importance column as an index number
    removed_column_name.append(inner_feature_with_max)

    # Count the features removed
    features_removed += 1
    print("remaining features:", len(remaining_features))
    print("number of features removed:", features_removed)
    print("features removed:", removed_column_name)

    # Add to a data frame for plotting
    all_train_scores.append(train_score)
    all_test_scores.append(test_score)
    removed_features_list.append(features_removed)
    number_of_features_removed_list.append(len(remaining_features))

# Combine all these into a single data frame
final_data_frame['features_name_removed'] = removed_column_name
final_data_frame['number_features_removed'] = removed_features_list
final_data_frame['train_accuracy'] = all_train_scores
final_data_frame['test_accuracy'] = all_test_scores
final_data_frame = final_data_frame.reset_index(drop=True)
print(final_data_frame)

# Plot labels and Title
fig, ax = plt.subplots()
plt.title("0.33% Test-Train Split RRFE applied to Extra Trees No RevNano")
plt.xlabel("Number of Most Informative Features Removed")
plt.ylabel("Scoring Metric (r2 value)")

# Make the plot
ax.plot(final_data_frame['number_features_removed'], final_data_frame['train_accuracy'], label="train accuracy",
        alpha=0.5)
ax.plot(final_data_frame['number_features_removed'], final_data_frame['test_accuracy'], label="test accuracy",
        alpha=0.5)
plt.ylim(top=1.2, bottom=-1)
plt.xlim(0)
# plt.annotate("Granularity: " + str(percentage), xy=(0.05, 0.95), xycoords='axes fraction')
plt.legend(fancybox=True, framealpha=0.3)
plt.show()
plt.cla()

# Plot
# bar_plot_df = pd.DataFrame({'test_accuracy': all_test_scores, 'train_accuracy': all_train_scores})
# bar_plot_df = bar_plot_df.iloc[0:20].sort_index(ascending=True)
# bar_plot_df = bar_plot_df.set_index(removed_column_name[0:20])
# ax2 = bar_plot_df.plot.barh(rot=1)
# ax2.plot()
# plt.title("Top 20 Most Informative Features")
# plt.show()

# Save data frame to excel
final_data_frame.to_csv("Extra_Trees_No_Revnano_33_Test_Train_RRFE_Results.csv")
