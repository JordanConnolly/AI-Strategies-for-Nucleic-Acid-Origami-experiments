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
        final_data_set = pd.DataFrame(total_y_list)
        index = []
    return final_data_set, index


# import the data set
data_set_file_path = cwd + '/dot_file_data_set.csv'
data_set = pd.read_csv(data_set_file_path)
# x = data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
#                            'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
#                            'Acetic acid (mM)', 'Acetate (mM)'])
x = data_set.drop(columns=['Magnesium (mM)', 'Unnamed: 0',
                           'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                           'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                           'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                           'average_shortest_path', 'average_clustering_coefficient',
                           'average_degree', 'average_betweenness_centrality',
                           'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                           'graph_reciprocity', 's-metric', 'wiener_index'])
y = data_set['Magnesium (mM)']

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


def preprocess_data(train, test):
    X_train_prep = preprocess.fit_transform(train)
    X_test_prep = preprocess.transform(test)
    return X_train_prep, X_test_prep


removed_column_name = []
# count features to remove
features_removed = 0
# Use cross val
cv_chosen = 3

# Use test-train split
get_X_train, get_X_test, get_y_train, get_y_test = train_test_split(x, y, test_size=0.33, random_state=42)
get_X_train_prep, get_X_test_prep = preprocess_data(get_X_train, get_X_test)

# Gather column names
x_transformed_columns_outer = get_transformer_feature_names(preprocess['preprocessor'])
df_test = pd.DataFrame(get_X_train_prep, columns=x_transformed_columns_outer)
cols = list(df_test.columns)

# Create ability to produce a final data frame
final_data_frame = pd.DataFrame()
all_train_scores = []
all_test_scores = []
removed_features_list = []
number_of_features_removed_list = []

for n_inner in range(0, len(cols) - features_removed):
    outer_feature_importance_cv = []
    average_train_score = []
    average_test_score = []
    reps = 30
    features_removed += 1
    for i in range(reps):
        seed_number = seed_numbers[i]
        final_data_set, index = stratified_dataset(data_set, 3, 'Magnesium (mM)', seed_number, index_required=True)
        # x = final_data_set.drop(columns=['Unnamed: 0', 'Magnesium (mM)',
        #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
        #                                  'Acetic acid (mM)', 'Acetate (mM)'])
        x = final_data_set.drop(columns=['Magnesium (mM)', 'Unnamed: 0',
                                         'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                         'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                                         'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                         'average_shortest_path', 'average_clustering_coefficient',
                                         'average_degree', 'average_betweenness_centrality',
                                         'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                                         'graph_reciprocity', 's-metric', 'wiener_index'])
        y = final_data_set['Magnesium (mM)']
        outer_cv = KFold(n_splits=cv_chosen, shuffle=False)
        data_splits = list(outer_cv.split(x, y))
        print(i)
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
            len_cols = len(cols)

            X_train_prep_df = pd.DataFrame(X_train_prep, columns=x_transformed_columns_outer)
            X_test_prep_df = pd.DataFrame(X_test_prep, columns=x_transformed_columns_outer)
            # errors is set to ignore, which is due to the fact TEST TRAIN SPLIT may not contain columns
            X_train_prep_outer = X_train_prep_df.drop(columns=removed_column_name, axis=1, errors='ignore')
            X_test_prep_outer = X_test_prep_df.drop(columns=removed_column_name, axis=1, errors='ignore')

            for n in range(0, 1):
                # fit data to model for first removal
                ETR.fit(X_train_prep_outer, y_train)
                # predict with model
                outer_pred = ETR.predict(X_test_prep_outer)

                # Train Score to Save
                train_score = ETR.score(X_train_prep_outer, y_train)
                # print(train_score)
                average_train_score.append(train_score)

                # Test Score to Save
                outer_score = r2_score(outer_pred, y_test)
                average_test_score.append(outer_score)
                # print(outer_score)
                # Feature Importance
                outer_feature_importance = ETR.feature_importances_
                remaining_features = [i for i in x_transformed_columns_outer if i not in removed_column_name]
                outer_importance_vectors = pd.Series(outer_feature_importance, index=remaining_features)
                outer_feature_with_max = outer_importance_vectors.idxmax()
                outer_feature_importance_cv.append(outer_importance_vectors)

                # print("remaining features:", len(remaining_features))
                # print("number of features removed:", features_removed)
                # print("features removed:", removed_column_name)

    outer_feature_importance_cv = pd.DataFrame(outer_feature_importance_cv)
    outer_feature_importance_cv = outer_feature_importance_cv.mean()
    outer_feature_with_max = outer_feature_importance_cv.idxmax()
    removed_column_name.append(outer_feature_with_max)

    # calculate test score of 3CV
    test_score = np.average(average_test_score)
    all_test_scores.append(test_score)
    print(test_score)

    # calculate train score of 3CV
    train_score = np.average(average_train_score)
    all_train_scores.append(train_score)
    print(train_score)
    # Count the features removed
    removed_features_list.append(features_removed)
    print(features_removed)

# Combine all these into a single data frame
final_data_frame['features_name_removed'] = removed_column_name
final_data_frame['number_features_removed'] = removed_features_list
final_data_frame['train_accuracy'] = all_train_scores
final_data_frame['test_accuracy'] = all_test_scores
final_data_frame = final_data_frame.reset_index(drop=True)
print(final_data_frame)

experiment_name = "Extra Trees Stratified No RevNano"
cross_validation = "3CV"

# Plot labels and Title
fig, ax = plt.subplots()
plt.title(cross_validation + " RRFE applied to " + experiment_name)
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
plt.savefig(cross_validation + experiment_name + " RRFE_Results.png")
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
final_data_frame.to_csv(cross_validation + experiment_name + " RRFE_Results.csv")
