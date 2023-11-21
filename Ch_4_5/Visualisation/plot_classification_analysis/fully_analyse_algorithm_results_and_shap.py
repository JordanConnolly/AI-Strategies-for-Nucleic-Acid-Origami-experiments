import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import random
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold
import os
import re
import glob
from textwrap import wrap

# for re-evals
from sklearn.metrics import recall_score, f1_score, accuracy_score, \
    precision_score, roc_auc_score, balanced_accuracy_score


# print(plt.style.available)
plt.style.use('ggplot')  # best so far
# plt.style.use('seaborn-whitegrid')

'''
Create the paths to the appropriate stored files
'''

# Directories containing files
cwd = os.getcwd()

# Change these
# original_data = "/dot_file_data_set.csv"
# original_data = "/dot_file_data_set_bolstered.csv"
# original_data = "/correct_imputation_magnesium_v3_ml_data_set_no_0_25.csv"
# original_data = "/subset_1_all_literature_high_cardinal_removed_ml_data_set.csv"
original_data = "/100_literature_ml_data_set.csv"

predictor_of_interest = "Thermal Profile"
number_of_splits = 5

cv = str(number_of_splits) + "CV"

# Title to change for Plot
# change_title = "Experiment Set 2 Extra Trees RFE " + cv + " Stratified Baseline"
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_NoRevNano"  # no revnano

# change_title = "Experiment Set 3 Extra Trees RFE " + cv + " Stratified with RevNano Features"
# experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"  # revnano

# change_title = "Experiment Set 1 Extra Trees RFE " + cv + " Stratified"
# experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
#        "Experiment Set 2/" \
#        "full_dataset_thermal_binary/" \
#        "80_pearson_corr/" \
#        + "removed_features_5CV_subset_balanced/"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Classification Experiments/" + \
#        "Experiment Set 3/" \
#        "bolstered_train_thermal_profile_binary/" \
#        "80_pearson_corr/" \
#        + "removed_features_3CV_subset_balanced/"

path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
       "Cardinal_Features_Removed/" \
       "Experiment Set 2/" \
       "full_dataset_thermal_profile_binary/" \
       "80_pearson_corr/" \
       "features_removed_5CV_subset_balanced/"

#  RevNano Base Removed

# create figures directory
# 'mkdir' creates a directory in current directory.
try:
    os.mkdir(path + "created_figs")
except:
    print("already created")

# additional directory paths
# path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_final_scores.csv"

path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_weighted_final_scores.csv"
path_metadata = path + "/extra_trees_results/model_metadata/"
path_all = path + "/extra_trees_results/model_plots/"
path_pickle = path + "extra_trees_results/model_pickle/"

'''
Add numerical sort for sorting files later
'''

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


'''
Create the table for repetition experiments
'''

columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
df = pd.read_csv(path_final, names=columns)
df.index += 1
df.sort_values(by=["Accuracy"], inplace=True)
df_low = pd.DataFrame(data=df.iloc[0]).transpose()
df_low_index = df_low.index[0]
df_high = pd.DataFrame(data=df.iloc[-1]).transpose()
df_high_index = df_high.index[0]
df_average = pd.DataFrame(data=df.apply(np.average), columns=["Average"]).transpose()
df_stdev = pd.DataFrame(data=df.apply(np.std), columns=["StDev"]).transpose()
total_result_df = df_low, df_high, df_average, df_stdev
result = pd.concat(total_result_df)
# creates a table for best and worst repetition
print(result)

'''
Create the plots for the best and worst result of repetition experiments
'''


def plot_confusion_matrix(path_to_file, original_data, change_title, original_predictor, save_path):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                print(repetition_value[1])
                real_vs_actual_df = pd.read_csv(file)
                # import df of real and predicted values of folds
                real_list = real_vs_actual_df['reality']
                pred_list = real_vs_actual_df['prediction']
                cm = metrics.confusion_matrix(real_list, pred_list)
                print(cm)
                c_report = metrics.classification_report(real_list, pred_list)
                ax = plt.subplot()
                sns.heatmap(cm, annot=True, ax=ax, cbar=False, cmap="YlGnBu", fmt='g')
                # labels, title and ticks
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                title = (original_predictor + change_title + ' Confusion Matrix Repetition ' + str(repetition_value[1]))
                ax.set_title("\n".join(wrap(title, 42)))
                plt.savefig(save_path + "created_figs/confusion_matrix_rep_" + str(repetition_value[1]) + ".png",
                            bbox_inches="tight")
                # plt.cla()
                plt.show()

    elif "xlsx" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                print(repetition_value[1])
                real_vs_actual_df = pd.read_csv(file)
                # import df of real and predicted values of folds
                real_list = real_vs_actual_df['reality']
                pred_list = real_vs_actual_df['prediction']
                cm = metrics.confusion_matrix(real_list, pred_list)
                print(cm)
                c_report = metrics.classification_report(real_list, pred_list)
                ax = plt.subplot()
                sns.heatmap(cm, annot=True, ax=ax, cbar=False, cmap="YlGnBu", fmt='g')
                # labels, title and ticks
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                title = (original_predictor + change_title + ' Confusion Matrix Repetition: ' + str(repetition_value[1]))
                ax.set_title("\n".join(wrap(title, 42)))
                plt.savefig(save_path + "created_figs/confusion_matrix_rep_" + str(repetition_value[1]) + ".png",
                            bbox_inches="tight")
                # plt.cla()
                plt.show()
        return


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
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')


def create_revnano_classification_model_shap_plots_from_pickle(path_to_file, original_data, change_title, original_variable,
                                                               number_of_splits, save_path):
    # import the PKL (pickle) files for the experiments
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # if original file is required
    data_set_file_path = cwd + original_data

    # Set Random Seed
    seed_numbers = list(range(1, 1000))
    random.Random(42).shuffle(seed_numbers)

    # import the data set
    data_set = pd.read_csv(data_set_file_path)

    # Remove all Experiments with NaN Outcome
    data_set = data_set[~data_set[original_variable].isin([np.NaN])]

    for file in all_files:
        path, filename = os.path.split(file)
        rep_regex = r"rep_\d*"
        rep_num = re.search(rep_regex, filename)
        repetition = str(rep_num.group())
        repetition_value = repetition.split("_")
        if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
            print(repetition_value[1])
            # which rep is performed
            rep = repetition_value[1]

            # #### remove Correlated Features ####
            predictor = original_variable
            feature_selection_rfe = True
            best_model_plot = True

            # Set Random Seed
            seed_numbers = list(range(1, 1000))
            random.Random(42).shuffle(seed_numbers)

            # remove 1 / 13 instances of thermal profile that are truly isothermal
            data_set = data_set[~data_set['Thermal Profile'].isin(['Isothermal-without-initial-denaturation'])]

            # Remove all Experiments with NaN Outcome
            final_data_set = data_set[~data_set[predictor].isin([np.NaN])]

            # ordinal encode target variable
            label_encoder = LabelEncoder()
            y = final_data_set[predictor]
            y = label_encoder.fit_transform(y)
            final_data_set['y'] = y
            ###### remove Correlated Features ####
            y = final_data_set['y']

            x_stored = final_data_set[['nodes', 'edges',
                                       'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                       'average_shortest_path', 'average_clustering_coefficient',
                                       'average_degree', 'average_betweenness_centrality',
                                       'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                                       'graph_reciprocity', 's-metric', 'wiener_index']]

            final_data_set = final_data_set.drop(columns=['nodes', 'edges',
                                                          'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                                          'average_shortest_path', 'average_clustering_coefficient',
                                                          'average_degree', 'average_betweenness_centrality',
                                                          'average_closeness_centrality', 'graph_assortivity',
                                                          'graph_diameter',
                                                          'graph_reciprocity', 's-metric', 'wiener_index'])
            # drop correlated features
            corr_matrix = final_data_set.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            drop_value = 0.80
            to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
            were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
            final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

            # Change the X and Y data, experiment name, scoring used
            x = final_data_set.drop(columns=['y', 'Thermal Profile', 'Peak Temperature (oC)',
                                             'Base Temperature (oC)', 'Temperature Ramp (s)',
                                             'index', 'Unnamed: 0',
                                             'Buffer Name', 'Scaffold Name',
                                             'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                             'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')
            x = pd.merge(x, x_stored, left_index=True, right_index=True)

            experiment_name = experiment

            outer_cv = StratifiedKFold(n_splits=number_of_splits, shuffle=False)
            data_splits = list(outer_cv.split(x, y))
            inner_loop_rep = 0

            #  Creates data splits
            for tr_idx, val_idx in data_splits:
                inner_loop_rep += 1

                # Removes correct answers from New_X, creates train-test sets
                X_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
                X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

                # pickle folder location path
                inner_pickle_path = path_pickle + experiment_name + "saved_model_" + \
                                    str(rep) + "_fold_" + str(inner_loop_rep) + ".pkl"

                # unpickle the pickled file
                infile = open(inner_pickle_path, 'rb')
                best = pickle.load(infile)
                infile.close()

                # Explain Shap Importance -- DIAGNOSTIC --
                explain = shap.TreeExplainer(best.named_steps['estimator'])
                transformer = best.named_steps['preprocessor']

                # May need to use toarray to work depending on num of features
                try:
                    transformed_data = transformer.transform(X_test)
                    print(X_test.shape)
                    print(transformed_data.shape)
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
                except:
                    transformed_data = transformer.transform(X_test)
                    transformed_data = transformed_data.toarray()
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)

                feature_selection_rfe = True

                # IF NO FEATURE ELIMINATION USED -> transformed df only
                # df_test_shap = transformed_data

                if feature_selection_rfe is True:
                    rfe_named = best.named_steps['rfe']
                    mask = rfe_named.support_
                    x_column_numpy = np.array(x_transformed_columns_outer)[mask]
                    df_test_shap = df_test[x_column_numpy]

                    # Apply SHAP
                    shap_values = explain.shap_values(df_test_shap)

                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_column_numpy, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values[0], df_test_shap,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()

                else:
                    df_test_shap = transformed_data
                    shap_values = explain.shap_values(df_test_shap)
                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values[0], df_test_shap,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()


def create_classification_model_shap_plots_from_pickle(path_to_file, original_data, change_title, original_variable,
                                                       number_of_splits, save_path):
    # import the PKL (pickle) files for the experiments
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # if original file is required
    data_set_file_path = cwd + original_data

    # Set Random Seed
    seed_numbers = list(range(1, 1000))
    random.Random(42).shuffle(seed_numbers)

    # import the data set
    data_set = pd.read_csv(data_set_file_path)

    # Remove all Experiments with NaN Outcome
    data_set = data_set[~data_set[original_variable].isin([np.NaN])]

    for file in all_files:
        path, filename = os.path.split(file)
        rep_regex = r"rep_\d*"
        rep_num = re.search(rep_regex, filename)
        repetition = str(rep_num.group())
        repetition_value = repetition.split("_")
        if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
            print(repetition_value[1])
            # which rep is performed
            rep = repetition_value[1]

            # #### remove Correlated Features ####
            predictor = original_variable
            feature_selection_rfe = True
            best_model_plot = True

            # Set Random Seed
            seed_numbers = list(range(1, 1000))
            random.Random(42).shuffle(seed_numbers)

            # remove 1 / 13 instances of thermal profile that are truly isothermal
            data_set = data_set[~data_set['Thermal Profile'].isin(['Isothermal-without-initial-denaturation'])]

            # Remove all Experiments with NaN Outcome
            final_data_set = data_set[~data_set[predictor].isin([np.NaN])]

            # ordinal encode target variable
            label_encoder = LabelEncoder()
            y = final_data_set[predictor]
            y = label_encoder.fit_transform(y)
            final_data_set['y'] = y

            y = final_data_set['y']

            ###### remove Correlated Features ####
            corr_matrix = final_data_set.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_value = 0.80
            to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
            were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
            final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

            # Change the X and Y data, experiment name, scoring used
            columns_to_drop_list = ['y', 'Thermal Profile', 'Peak Temperature (oC)',
                                    'Base Temperature (oC)', 'Temperature Ramp (s)',
                                    'index', 'Unnamed: 0',
                                    'Buffer Name', 'Scaffold Name',
                                    'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                    'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                                    'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                    'average_shortest_path', 'average_clustering_coefficient',
                                    'average_degree', 'average_betweenness_centrality',
                                    'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                                    'graph_reciprocity', 's-metric', 'wiener_index']

            # Change the X and Y data, experiment name, scoring used
            x = final_data_set.drop(columns=columns_to_drop_list, errors='ignore')

            experiment_name = experiment

            outer_cv = StratifiedKFold(n_splits=number_of_splits, shuffle=False)
            data_splits = list(outer_cv.split(x, y))
            inner_loop_rep = 0

            #  Creates data splits
            for tr_idx, val_idx in data_splits:
                inner_loop_rep += 1

                # Removes correct answers from New_X, creates train-test sets
                X_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
                X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

                # pickle folder location path
                inner_pickle_path = path_pickle + experiment_name + "saved_model_" + \
                                    str(rep) + "_fold_" + str(inner_loop_rep) + ".pkl"

                # unpickle the pickled file
                infile = open(inner_pickle_path, 'rb')
                best = pickle.load(infile)
                infile.close()

                # Explain Shap Importance -- DIAGNOSTIC --
                explain = shap.TreeExplainer(best.named_steps['estimator'])
                transformer = best.named_steps['preprocessor']

                # May need to use toarray to work depending on num of features
                try:
                    transformed_data = transformer.transform(X_test)
                    print(X_test.shape)
                    print(transformed_data.shape)
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
                except:
                    transformed_data = transformer.transform(X_test)
                    transformed_data = transformed_data.toarray()
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)

                feature_selection_rfe = True

                # IF NO FEATURE ELIMINATION USED -> transformed df only
                # df_test_shap = transformed_data

                if feature_selection_rfe is True:
                    rfe_named = best.named_steps['rfe']
                    mask = rfe_named.support_
                    x_column_numpy = np.array(x_transformed_columns_outer)[mask]
                    df_test_shap = df_test[x_column_numpy]

                    # Apply SHAP
                    shap_values = explain.shap_values(df_test_shap)

                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_column_numpy, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values[0], df_test_shap,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()

                else:
                    df_test_shap = transformed_data
                    shap_values = explain.shap_values(df_test_shap)
                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values[0], df_test_shap,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()

            if best_model_plot is True:
                # pickle folder location path
                outer_pickle_path = path_pickle + experiment_name + "saved_model_" + \
                                    str(rep) + "_all_data.pkl"

                # unpickle the pickled file
                infile = open(outer_pickle_path, 'rb')
                best_all = pickle.load(infile)
                infile.close()

                # Explain Shap Importance -- OUTER --
                explain = shap.TreeExplainer(best_all.named_steps['estimator'])
                transformer = best_all.named_steps['preprocessor']

                # May need to use toarray to work depending on num of features
                try:
                    transformed_data = transformer.transform(x)
                    print(transformed_data.shape)
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
                except:
                    transformed_data = transformer.transform(x)
                    transformed_data = transformed_data.toarray()
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)

                if feature_selection_rfe is True:
                    rfe_named = best_all.named_steps['rfe']
                    mask = rfe_named.support_
                    x_column_numpy = np.array(x_transformed_columns_outer)[mask]
                    df_test_fe = df_test[x_column_numpy]
                    columns_used_names = df_test_fe.columns

                    # Feature Importance for whole data fitted set best model -- IMPORTANT --
                    feature_importance = best_all.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(columns_used_names, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)
                    # Explain Shap Importance -- IMPORTANT --
                    explain = shap.TreeExplainer(best_all.named_steps['estimator'])
                    shap_values = explain.shap_values(df_test_fe)

                    # plots whole data set
                    shap.summary_plot(shap_values[0], df_test_fe,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()

                else:
                    # Feature Importance for whole data fitted set best model -- IMPORTANT --
                    feature_importance = best_all.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)
                    # Explain Shap Importance -- IMPORTANT --
                    explain = shap.TreeExplainer(best_all.named_steps['estimator'])
                    shap_values = explain.shap_values(df_test)

                    # plots whole data set
                    shap.summary_plot(shap_values[0], df_test,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()


def create_bolstered_classification_model_shap_plots_from_pickle(path_to_file, original_data, change_title, original_variable,
                                                                 number_of_splits, save_path):
    # import the PKL (pickle) files for the experiments
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # if original file is required
    data_set_file_path = cwd + original_data

    # Set Random Seed
    seed_numbers = list(range(1, 1000))
    random.Random(42).shuffle(seed_numbers)

    # import the data set
    data_set = pd.read_csv(data_set_file_path)

    # Remove all Experiments with NaN Outcome
    data_set = data_set[~data_set[original_variable].isin([np.NaN])]

    for file in all_files:
        path, filename = os.path.split(file)
        rep_regex = r"rep_\d*"
        rep_num = re.search(rep_regex, filename)
        repetition = str(rep_num.group())
        repetition_value = repetition.split("_")
        if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
            print(repetition_value[1])
            # which rep is performed
            rep = repetition_value[1]

            # #### remove Correlated Features ####
            predictor = original_variable
            feature_selection_rfe = True
            best_model_plot = True

            # Set Random Seed
            seed_numbers = list(range(1, 1000))
            random.Random(42).shuffle(seed_numbers)

            # import the data set
            data_set_file_path = cwd + '/subset_1_all_literature_high_cardinal_removed_ml_data_set.csv'
            data_set = pd.read_csv(data_set_file_path)

            # remove 1 / 13 instances of thermal profile that are truly isothermal
            data_set = data_set[~data_set['Thermal Profile'].isin(['Isothermal-without-initial-denaturation'])]

            # Enter predictor of interest
            predictor = 'Thermal Profile'

            # Remove all Experiments with NaN Outcome
            final_data_set = data_set[~data_set[predictor].isin([np.NaN])]

            # ordinal encode target variable
            label_encoder = LabelEncoder()
            y = final_data_set[predictor]
            y = label_encoder.fit_transform(y)
            final_data_set['y'] = y

            # Change the X and Y data, experiment name, scoring used
            columns_to_drop_list = ['Peak Temperature (oC)',
                                    'Base Temperature (oC)', 'Temperature Ramp (s)',
                                    'index', 'Unnamed: 0',
                                    'Experiment Number', 'Boric Acid (mM)',
                                    'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                                    'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                    'average_shortest_path', 'average_clustering_coefficient',
                                    'average_degree', 'average_betweenness_centrality',
                                    'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                                    'graph_reciprocity', 's-metric', 'wiener_index']

            # Change the X and Y data, experiment name, scoring used
            x = final_data_set.drop(columns=columns_to_drop_list, errors='ignore')

            # split the original data set instances and the bolstered instances
            data_set_file_path_og = cwd + '/100_literature_ml_data_set.csv'
            data_set_og = pd.read_csv(data_set_file_path_og)
            data_set_og = data_set_og[~data_set_og['Thermal Profile'].isin(['Isothermal-without-initial-denaturation'])]
            # Remove all Experiments with NaN Outcome
            data_set_og = data_set_og[~data_set_og[predictor].isin([np.NaN])]

            # Remove all Experiments with Anomalous Mg values
            og_data_set = data_set_og[
                ~data_set_og['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
            og_data_set = og_data_set.drop(columns=columns_to_drop_list, errors='ignore')

            # ordinal encode target variable
            label_encoder = LabelEncoder()
            y = og_data_set[predictor]
            y = label_encoder.fit_transform(y)
            og_data_set['y'] = y

            bolster_data_set = x[x['Paper Number'] >= 101]

            # retrieve y values for the test sets
            og_data_set_y = og_data_set['y']
            bolster_data_set_y = bolster_data_set['y']

            def corr_drop(ds_given, drop_value):
                # find correlated features to drop
                corr_matrix = ds_given.corr().abs()
                print(ds_given.shape)
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
                were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
                return to_drop

            # remove correlated values (calculated as if from all instances)
            to_drop = corr_drop(x, 0.80)
            og_data_set = og_data_set.drop(og_data_set[to_drop], axis=1)
            bolster_data_set = bolster_data_set.drop(bolster_data_set[to_drop], axis=1)

            # remove y values from X sets
            og_data_set = og_data_set.drop(columns=["Paper Number", predictor, 'y'], errors='ignore')
            bolster_data_set = bolster_data_set.drop(columns=["Paper Number", predictor, 'y'], errors='ignore')

            # print the values
            print("this is the og data set:", len(og_data_set))
            print("this is the bolster data set:", len(bolster_data_set))

            # you would then create the train AND test sets from the OG data set.
            # you would then create a train only SPLIT of the bolster set (shuffle-split into n=3)
            # to do this you could just use the train-test split and combine the tr-val index values, add to the OG.

            y = final_data_set[predictor]

            experiment_name = experiment

            outer_cv = StratifiedKFold(n_splits=number_of_splits, shuffle=False)
            data_splits = list(outer_cv.split(og_data_set, og_data_set_y))
            data_splits_2 = list(outer_cv.split(bolster_data_set, bolster_data_set_y))

            inner_loop_rep = 0

            for i in range(number_of_splits):
                inner_loop_rep += 1

                # split the og data to create test set, and the train set for combining with bolster
                og_data_split = data_splits[i]
                og_train_idx = og_data_split[0]
                og_test_idx = og_data_split[1]
                og_X_train, og_y_train = og_data_set.iloc[og_train_idx], og_data_set_y.iloc[og_train_idx]
                X_test, y_test = og_data_set.iloc[og_test_idx], og_data_set_y.iloc[og_test_idx]

                # split the bolstered data to combined with the og train set
                bolster_data_split = data_splits_2[i]
                bolster_train_idx = bolster_data_split[0]
                bolster_test_idx = bolster_data_split[1]
                bolster_X_train, bolster_y_train = bolster_data_set.iloc[bolster_train_idx], \
                                                   bolster_data_set_y.iloc[bolster_train_idx]
                bolster_X_test, bolster_y_test = bolster_data_set.iloc[bolster_test_idx], \
                                                 bolster_data_set_y.iloc[bolster_test_idx]

                # combine the bolstered into one large set to combine
                concat_bolster = pd.concat([bolster_X_train, bolster_X_test])
                concat_bolster_y = pd.concat([bolster_y_train, bolster_y_test])

                # combine the large set with original train data
                X_train = pd.concat([og_X_train, concat_bolster])
                y_train = pd.concat([og_y_train, concat_bolster_y])

                print(len(X_train))
                print(len(y_train))

                print(len(X_test))
                print(len(y_test))

                # pickle folder location path
                inner_pickle_path = path_pickle + experiment_name + "saved_model_" + \
                                    str(rep) + "_fold_" + str(inner_loop_rep) + ".pkl"

                # unpickle the pickled file
                infile = open(inner_pickle_path, 'rb')
                best = pickle.load(infile)
                infile.close()

                # Explain Shap Importance -- DIAGNOSTIC --
                explain = shap.TreeExplainer(best.named_steps['estimator'])
                transformer = best.named_steps['preprocessor']

                # May need to use toarray to work depending on num of features
                try:
                    transformed_data = transformer.transform(X_test)
                    print(X_test.shape)
                    print(transformed_data.shape)
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
                except:
                    transformed_data = transformer.transform(X_test)
                    transformed_data = transformed_data.toarray()
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)

                feature_selection_rfe = True

                # IF NO FEATURE ELIMINATION USED -> transformed df only
                # df_test_shap = transformed_data

                if feature_selection_rfe is True:
                    rfe_named = best.named_steps['rfe']
                    mask = rfe_named.support_
                    x_column_numpy = np.array(x_transformed_columns_outer)[mask]
                    df_test_shap = df_test[x_column_numpy]

                    # Apply SHAP
                    shap_values = explain.shap_values(df_test_shap)

                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_column_numpy, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values[0], df_test_shap,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()

                else:
                    df_test_shap = transformed_data
                    shap_values = explain.shap_values(df_test_shap)
                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values[0], df_test_shap,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()

            if best_model_plot is True:
                # pickle folder location path
                outer_pickle_path = path_pickle + experiment_name + "saved_model_" + \
                                    str(rep) + "_all_data.pkl"

                # unpickle the pickled file
                infile = open(outer_pickle_path, 'rb')
                best_all = pickle.load(infile)
                infile.close()

                # Explain Shap Importance -- OUTER --
                explain = shap.TreeExplainer(best_all.named_steps['estimator'])
                transformer = best_all.named_steps['preprocessor']

                # May need to use to array to work depending on num of features
                try:
                    transformed_data = transformer.transform(x)
                    print(transformed_data.shape)
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
                except:
                    transformed_data = transformer.transform(x)
                    transformed_data = transformed_data.toarray()
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)

                if feature_selection_rfe is True:
                    rfe_named = best_all.named_steps['rfe']
                    mask = rfe_named.support_
                    x_column_numpy = np.array(x_transformed_columns_outer)[mask]
                    df_test_fe = df_test[x_column_numpy]
                    columns_used_names = df_test_fe.columns

                    # Feature Importance for whole data fitted set best model -- IMPORTANT --
                    feature_importance = best_all.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(columns_used_names, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)
                    # Explain Shap Importance -- IMPORTANT --
                    explain = shap.TreeExplainer(best_all.named_steps['estimator'])
                    shap_values = explain.shap_values(df_test_fe)

                    # plots whole data set
                    shap.summary_plot(shap_values[0], df_test_fe,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()

                else:
                    # Feature Importance for whole data fitted set best model -- IMPORTANT --
                    feature_importance = best_all.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)
                    # Explain Shap Importance -- IMPORTANT --
                    explain = shap.TreeExplainer(best_all.named_steps['estimator'])
                    shap_values = explain.shap_values(df_test)

                    # plots whole data set
                    shap.summary_plot(shap_values[0], df_test,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()


def plot_roc_auc_curve(path_to_file, original_data, change_title, original_predictor, save_path):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                # print(repetition_value[1])
                real_vs_actual_df = pd.read_csv(file)
                # import df of real and predicted values of folds
                real_list = real_vs_actual_df['reality']
                pred_list = real_vs_actual_df['prediction']
                ax = plt.subplot()
                display = metrics.RocCurveDisplay.from_predictions(real_list, pred_list, ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                title = (original_predictor + change_title + ' ROC Curve; Repetition: ' + str(repetition_value[1]))
                ax.set_title("\n".join(wrap(title, 42)))
                plt.savefig(save_path + "created_figs/ROCAUC_summary_plot_rep_" + str(repetition_value[1])
                            + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                plt.show()


def get_fold_scores_for_reps(path_to_file, path_to_score, original_data):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
            # if int(repetition_value[1]) > 0:  # shows all fold scores
                print(repetition_value[1])
                fold_score_files = sorted(glob.glob(path_to_score + experiment + "_rep_" + str(repetition_value[1]) +
                                                    "_fold_*_score.csv"), key=numerical_sort)
                if len(fold_score_files) < 1:
                    print(" ")
                # print(fold_score_files)
                for fold_score_file in fold_score_files:
                    columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
                    fold_score_df = pd.read_csv(fold_score_file, names=columns)
                    print(fold_score_df)


def get_reevaluated_fold_scores_for_reps(path_to_file, path_to_score, original_data):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                print(repetition_value[1])
                rep = str(repetition_value[1])
                # Re-Evaluate Scores
                all_reeval_files = sorted(glob.glob(path_metadata + experiment + "_actual_vs_pred_rep_" + rep +
                                          "_fold_*.csv"), key=numerical_sort)

                for actual_vs_pred_file in all_reeval_files:

                    # score lists
                    new_accuracy_score = []
                    new_balanced_accuracy_score = []
                    new_recall_score = []
                    new_precision_score = []
                    new_f1_score = []
                    new_rocauc_score = []
                    scores = []

                    fold_cv_df = pd.read_csv(actual_vs_pred_file)
                    fold_cv_df.drop(columns=["Unnamed: 0"], inplace=True)
                    pred = fold_cv_df["prediction"].values
                    y_test = fold_cv_df["reality"].values

                    # Create scoring metrics and append for folds
                    accuracy_result = accuracy_score(y_test, pred)
                    scores.append(accuracy_result)
                    new_accuracy_score.append(accuracy_result)

                    # Balanced Accuracy
                    balanced_accuracy_result = balanced_accuracy_score(y_test, pred)
                    scores.append(balanced_accuracy_result)
                    new_balanced_accuracy_score.append(balanced_accuracy_result)

                    # Precision Score
                    precision_result = precision_score(y_test, pred, average="weighted")
                    scores.append(precision_result)
                    new_precision_score.append(precision_result)

                    # Recall Score
                    recall_result = recall_score(y_test, pred, average="weighted")
                    scores.append(recall_result)
                    new_recall_score.append(recall_result)

                    # F1 Score
                    f1_result = f1_score(y_test, pred, average="weighted")
                    scores.append(f1_result)
                    new_f1_score.append(f1_result)

                    # ROC AUC Score
                    roc_auc_result = roc_auc_score(y_test, pred, average="weighted")
                    scores.append(roc_auc_result)
                    new_rocauc_score.append(roc_auc_result)

                    # Create any necessary averages of scores for nested models and print them
                    final_accuracy = np.average(new_accuracy_score)
                    final_balanced_accuracy = np.average(new_balanced_accuracy_score)
                    final_precision = np.average(new_precision_score)
                    final_recall = np.average(new_recall_score)
                    final_f1 = np.average(new_f1_score)
                    final_roc_auc = np.average(new_rocauc_score)
                    # print("final accuracy", final_accuracy)
                    # print("final balanced accuracy", final_balanced_accuracy)
                    # print("final precision", final_precision)
                    # print("final recall", final_recall)
                    # print("final F1", final_f1)
                    # print("final roc auc", final_roc_auc)
                    final_scores = [final_balanced_accuracy, final_precision, final_recall, final_f1, final_roc_auc]
                    acc_columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
                    fold_score_df = pd.DataFrame(columns=acc_columns)
                    fold_score_df.loc[0] = final_scores
                    fold_score_df.index.name = None
                    print(fold_score_df)


def get_params_for_reps(path_to_file, path_to_score, original_data, original_variable, save_path, path_to_pickle, high):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if high:
                if int(repetition_value[1]) == int(df_high_index):
                    print("using high repetition for params")
                    # if int(repetition_value[1]) > 0:  # shows all fold scores
                    print(repetition_value[1])
                    fold_score_files = sorted(
                        glob.glob(path_to_score + experiment + "_rep_" + str(repetition_value[1]) +
                                  "_fold_*_score.csv"), key=numerical_sort)
                    if len(fold_score_files) < 1:
                        print(" ")

                    # create each fold score table
                    fold_score_table_df = pd.DataFrame()
                    fold_score_file_number = 0
                    for fold_score_file in fold_score_files:
                        fold_score_file_number += 1
                        columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
                        fold_score_df = pd.read_csv(fold_score_file, names=columns)
                        fold_score_df.insert(loc=0, column='fold', value=fold_score_file_number)
                        fold_score_table_df = pd.concat([fold_score_table_df, fold_score_df], ignore_index=True)
                    print(fold_score_table_df)
                    fold_pickle_files = sorted(glob.glob(path_to_pickle + experiment + "saved_model_" +
                                                         str(repetition_value[1]) +
                                                         "_fold_*.pkl"), key=numerical_sort)
                    fold_number = 0
                    print(fold_pickle_files)
                    for i in range(number_of_splits):
                        fold_pickle_file = fold_pickle_files[i]
                        loaded_fold_model = pickle.load(open(fold_pickle_file, 'rb'))
                        estimator_params = loaded_fold_model.named_steps['estimator']
                        # print(estimator_params)
                        rfe_named = loaded_fold_model.named_steps['rfe']
                        # print(" ")
                        # print(rfe_named)
                        # print(" ")

                        # print all of the parameters possible:
                        for param, value in loaded_fold_model.get_params(deep=True).items():
                            if "depth" in param:
                                print(f"{param} -> {value}")
                            if "estimators" in param:
                                print(f"{param} -> {value}")
                            if "n_features_to_select" in param:
                                print(f"{param} -> {value}")

                        print(" ")
                        print("---next fold---")
            else:
                if int(repetition_value[1]) == int(df_low_index):
                    print("using worst repetition for params")
                    # if int(repetition_value[1]) > 0:  # shows all fold scores
                    print(repetition_value[1])
                    fold_score_files = sorted(glob.glob(path_to_score + experiment + "_rep_" + str(repetition_value[1]) +
                                                        "_fold_*_score.csv"), key=numerical_sort)
                    if len(fold_score_files) < 1:
                        print(" ")

                    # create each fold score table
                    fold_score_table_df = pd.DataFrame()
                    fold_score_file_number = 0
                    for fold_score_file in fold_score_files:
                        fold_score_file_number += 1
                        columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
                        fold_score_df = pd.read_csv(fold_score_file, names=columns)
                        fold_score_df.insert(loc=0, column='fold', value=fold_score_file_number)
                        fold_score_table_df = pd.concat([fold_score_table_df, fold_score_df], ignore_index=True)
                    print(fold_score_table_df)
                    fold_pickle_files = sorted(glob.glob(path_to_pickle + experiment + "saved_model_" +
                                                         str(repetition_value[1]) +
                                                         "_fold_*.pkl"), key=numerical_sort)
                    fold_number = 0
                    print(fold_pickle_files)
                    for i in range(number_of_splits):
                        fold_pickle_file = fold_pickle_files[i]
                        loaded_fold_model = pickle.load(open(fold_pickle_file, 'rb'))
                        estimator_params = loaded_fold_model.named_steps['estimator']
                        # print(estimator_params)
                        rfe_named = loaded_fold_model.named_steps['rfe']
                        # print(" ")
                        # print(rfe_named)
                        # print(" ")

                        # print all of the parameters possible:
                        for param, value in loaded_fold_model.get_params(deep=True).items():
                            if "depth" in param:
                                print(f"{param} -> {value}")
                            if "estimators" in param:
                                print(f"{param} -> {value}")
                            if "n_features_to_select" in param:
                                print(f"{param} -> {value}")

                        print(" ")
                        print("---next fold---")


def get_params_for_reps_no_fold_scores(path_to_file, path_to_score, original_data,
                                       original_variable, save_path, path_to_pickle, high):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        for file in all_files:
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if high:
                if int(repetition_value[1]) == int(df_high_index):
                    print("using high repetition for params")
                    # if int(repetition_value[1]) > 0:  # shows all fold scores
                    print(repetition_value[1])
                    fold_pickle_files = sorted(glob.glob(path_to_pickle + experiment + "saved_model_" +
                                                         str(repetition_value[1]) +
                                                         "_fold_*.pkl"), key=numerical_sort)
                    fold_number = 0
                    print(fold_pickle_files)
                    for i in range(number_of_splits):
                        fold_pickle_file = fold_pickle_files[i]
                        loaded_fold_model = pickle.load(open(fold_pickle_file, 'rb'))
                        estimator_params = loaded_fold_model.named_steps['estimator']
                        # print(estimator_params)
                        rfe_named = loaded_fold_model.named_steps['rfe']
                        # print(" ")
                        # print(rfe_named)
                        # print(" ")

                        # print all of the parameters possible:
                        for param, value in loaded_fold_model.get_params(deep=True).items():
                            if "depth" in param:
                                print(f"{param} -> {value}")
                            if "estimators" in param:
                                print(f"{param} -> {value}")
                            if "n_features_to_select" in param:
                                print(f"{param} -> {value}")

                        print(" ")
                        print("---next fold---")
            else:
                if int(repetition_value[1]) == int(df_low_index):
                    print("using worst repetition for params")
                    # if int(repetition_value[1]) > 0:  # shows all fold scores
                    print(repetition_value[1])
                    fold_score_files = sorted(glob.glob(path_to_score + experiment + "_rep_" + str(repetition_value[1]) +
                                                        "_fold_*_score.csv"), key=numerical_sort)
                    if len(fold_score_files) < 1:
                        print(" ")

                    # create each fold score table
                    fold_pickle_files = sorted(glob.glob(path_to_pickle + experiment + "saved_model_" +
                                                         str(repetition_value[1]) +
                                                         "_fold_*.pkl"), key=numerical_sort)
                    fold_number = 0
                    print(fold_pickle_files)
                    for i in range(number_of_splits):
                        fold_pickle_file = fold_pickle_files[i]
                        loaded_fold_model = pickle.load(open(fold_pickle_file, 'rb'))
                        estimator_params = loaded_fold_model.named_steps['estimator']
                        # print(estimator_params)
                        rfe_named = loaded_fold_model.named_steps['rfe']
                        # print(" ")
                        # print(rfe_named)
                        # print(" ")

                        # print all of the parameters possible:
                        for param, value in loaded_fold_model.get_params(deep=True).items():
                            if "depth" in param:
                                print(f"{param} -> {value}")
                            if "estimators" in param:
                                print(f"{param} -> {value}")
                            if "n_features_to_select" in param:
                                print(f"{param} -> {value}")

                        print(" ")
                        print("---next fold---")

# upgrade to downgrade to 1.0.2 to create roc aucs?
# plot_roc_auc_curve(path_all, original_data, change_title, "Thermal Profile (Binary) ", path)
# plot_confusion_matrix(path_all, original_data, change_title, "Thermal Profile (Binary) ", path)

# # downgrade to 0.24.2 to create shap plots
# create_revnano_classification_model_shap_plots_from_pickle(path_all, original_data, change_title, predictor_of_interest,
#                                                            number_of_splits, path)

# create_classification_model_shap_plots_from_pickle(path_all, original_data, change_title, predictor_of_interest,
#                                                    number_of_splits, path)
# create_bolstered_classification_model_shap_plots_from_pickle(path_all, original_data, change_title,
#                                                              predictor_of_interest, number_of_splits, path)
# get_fold_scores_for_reps(path_all, path_metadata)
# get_params_for_reps(path_all, path_metadata, original_data, predictor_of_interest, path, path_pickle, high=False)
# get_reevaluated_fold_scores_for_reps(path_all, path_metadata, original_data)
# get_params_for_reps_no_fold_scores(path_all, path_metadata, original_data,
#                                    predictor_of_interest, path, path_pickle, high=False)
