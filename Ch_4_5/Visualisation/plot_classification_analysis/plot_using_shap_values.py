import random
import pandas as pd
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import os
import re
import glob
import matplotlib.pyplot as plt
from textwrap import wrap


# Directories containing files
cwd = os.getcwd()
# path = "C:/Users/Big JC/PycharmProjects/PhDCodeOnly/ML_results_to_analyse/" \
#        "thermal_profile_classification_results/thermal_profile_shap_data/make_shap_plots_of_best_worst_results/" \
#        "thermal_subset_2/extra_trees_results/model_metadata/"
# experiment_name = "Extra_Trees_RFE_3CV_Stratified_Baseline"

path = "C:/Users/Big JC/PycharmProjects/PhDCodeOnly/ML_results_to_analyse/" \
       "thermal_profile_classification_results/thermal_profile_no_stratification_15032021/" \
       "3CV_no_stratification/subset_2_class_balanced/extra_trees_results/model_metadata/"
experiment_name = "Extra_Trees_RFE_3CV_Stratified_Baseline"
rep_of_interest = 8
class_of_interest = 0


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
    # list of current outcome of interest (y) values
    current_list = [value[y_value_of_interest] for y_value_of_interest in range(len(value))]
    # list of next outcome of interest (y) values
    next_list = [value[y_value_of_interest + 1] for y_value_of_interest in range(len(value) - 1)]
    current_list.insert(len(current_list), 0)
    next_list.insert(len(next_list), 0)
    combined_list = list(zip(current_list, next_list))
    all_index_df = pd.DataFrame()
    for i in range(len(combined_list)):
        list_instance = combined_list[i]
        current_value = list_instance[0]
        next_value = list_instance[1]
        if current_value > next_value:
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


numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


shap_files = sorted(glob.glob(path + experiment_name + "_SHAP_values_for_plot_" + str(rep_of_interest + 1)
                              + "_fold_*_class_" + str(class_of_interest) + ".csv"), key=numerical_sort)
print(shap_files)
if len(shap_files) <= 0:
    print("shap files configured incorrectly, shap files length:", len(shap_files))
if len(shap_files) > 0:
    print("shap files available:", len(shap_files))


# data_set_file_path = cwd + "/correct_imputation_magnesium_v3_ml_data_set_no_0_25.xlsx"
# data_set = pd.read_excel(data_set_file_path).reset_index()
# Remove all Experiments with Anomalous Mg values
# data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(42).shuffle(seed_numbers)

data_set_file_path = cwd + '/dot_file_data_set.csv'
data_set = pd.read_csv(data_set_file_path)

outer_i = 0
reps_list = [rep_of_interest]
for i in reps_list:
    print(i)
    seed_number = seed_numbers[i]
    print(seed_number)

    # Enter predictor of interest
    predictor = 'Thermal Profile'

    # Remove all Experiments with NaN Outcome
    data_set = data_set[~data_set[predictor].isin([np.NaN])]

    # ordinal encode target variable
    label_encoder = LabelEncoder()
    y = data_set[predictor]
    y = label_encoder.fit_transform(y)
    data_set['y'] = y

    final_data_set, index = stratified_dataset(data_set, 3, 'Magnesium (mM)', seed_number, index_required=True)

    y = final_data_set['y']
    # Change the X and Y data, experiment name, scoring used
    x = final_data_set.drop(columns=['y'], errors='ignore')

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
    rf = Pipeline(steps=[('preprocessor', preprocessor)])

    # create folders to store meta data
    important = (cwd + "/extra_trees_results" + "/model_plots/")

    # create the splits of the model
    outer_cv = KFold(n_splits=3, shuffle=False)
    data_splits = list(outer_cv.split(x, y))

    inner_loop_rep = 0

    for tr_idx, val_idx in data_splits:
        inner_loop_rep += 1
        # Removes correct answers from New_X, creates train-test sets
        X_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
        X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

        # Inner Fold Shuffle
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)

        inner_data_splits = list(inner_cv.split(X_train, y_train))

        # Create processed X for creating shap plot
        transformed_data_1 = preprocessor.fit_transform(X_train)
        transformed_data = preprocessor.transform(X_test)

        # Access the shap values created in experiment
        file = shap_files[outer_i]
        outer_i += 1
        print(file)
        path, filename = os.path.split(file)
        print(filename)


        # Import the shap values and feature names from the glob found file
        shap_values_df = pd.read_csv(file)
        shap_values_df = shap_values_df.drop(columns="Unnamed: 0")

        shap_values_columns = shap_values_df.columns
        features_named = shap_values_columns.tolist()
        # print(shap_values_columns)

        # Create instances data frame to calculate summary plot with
        x_transformed_columns_outer = get_transformer_feature_names(preprocessor)
        df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
        print(df_test.columns)
        df_test_shap = df_test[features_named]
        # print(df_test_shap)
        # Apply SHAP
        shap_values = shap_values_df.values
        # print(shap_values)

        shap.summary_plot(shap_values, df_test_shap,
                          feature_names=features_named, show=False)
        title = (experiment_name + " shap summary plot experiment " + str(i + 1)
                 + " fold " + str(inner_loop_rep) + " class:" + str(class_of_interest))
        plt.title("\n".join(wrap(title, 44)))
        plt.show()
