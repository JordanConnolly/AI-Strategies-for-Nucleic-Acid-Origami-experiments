import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from textwrap import wrap
from sklearn.model_selection import KFold
import shap
import random
import pickle
from pathlib import Path

# print(plt.style.available)
plt.style.use('ggplot')  # best so far
# plt.style.use('seaborn-whitegrid')

# set params for if RFE is used, if BEST is plotted
feature_selection_rfe = True
best_model_plot = True

# Pandas and Numpy Options
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

# change it to scikit-learn 0.23.2, not 24
'''
Create the paths to the appropriate stored files
'''

# Directories containing files
cwd = os.getcwd()

# original_data = "/correct_imputation_magnesium_v3_ml_data_set_no_0_25.csv"
# original_data = "/subset_1_all_magnesium_0_25_yield_removed_literature_ml_data_set.csv"
# original_data = "/dot_file_data_set_bolstered.csv"
# original_data = "/dot_file_data_set.csv"
# original_data = "/correct_imputation_afm_yield_ml_data_set_no_0_25.csv"
original_data = "/100_literature_ml_data_set.csv"
# original_data = "/subset_1_all_literature_high_cardinal_removed_ml_data_set.csv"

# predictor_of_interest = "Staple Molarity (nM)"
predictor_of_interest = "Magnesium (mM)"
number_of_splits = 3

cv = str(number_of_splits) + "CV"

# Title to change for Plot
change_title = "Experiment Set 1 Extra Trees RFE " + cv + " Stratified Baseline"
# experiment = "Extra_Trees_RFE_" + cv + "_Stratified_NoRevNano"  # no revnano

# change_title = "Experiment Set 1 Extra Trees RFE " + cv + " Stratified with RevNano Features"
experiment = "Extra_Trees_RFE_" + cv + "_Stratified_Baseline"  # revnano

quick_save = "E:/PhD Files/RQ3/graphing_machine_learning_results/graphing_machine_learning_results/quick_save_path/"

path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Regression Experiments/" + \
       "Cardinal_Features_Removed/" \
       "Experiment Set 1/" \
       "full_dataset_magnesium/" \
       "80_pearson_corr/" \
       + cv + "/"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/Regression Experiments/" + \
#        "Cardinal_Features_Removed/" \
#         "Experiment Set 3/" \
#         "bolstered_train_staple/" \
#         "80_pearson_corr/" \
#         + cv + "/"

# path = "E:/PhD Files/RQ3/final_machine_learning_experiments/RevNano Experiments/" \
#        "Cardinal_Features_Removed/" \
#        "Experiment Set 2 RevNano Base Removed/" \
#        "full_dataset_staple/" \
#        "80_pearson_corr/" \
#        "removed_features_5CV/"

# Experiment Set 1 RevNano Base Removed

# create figures directory
try:
    os.mkdir(path + "created_figs")
except:
    print("already created")

try:
    os.mkdir(path + "fold_interrogation")
except:
    print("already created")

# additional directory paths
path_final = path + "/extra_trees_results/model_final_scores/" + experiment + "_final_scores.csv"
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

columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]
df = pd.read_csv(path_final, names=columns)
print(df)
df.index += 1
df.sort_values(by=["r2"], inplace=True)
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


def plot_combined_predicted_vs_actual(path_to_file, original_data, change_title, original_variable, log, alpha,
                                      save_path):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")
    # if original file is required
    data_set_file_path = cwd + original_data
    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        original_data_frame = pd.read_csv(data_set_file_path).reset_index()
        y = original_data_frame[original_variable]
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

                # Save folds actual vs predicts -- DIAGNOSTIC --
                fig, ax = plt.subplots()
                ax.scatter(real_list, pred_list, alpha=alpha)
                if log is True:
                    ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    title = (change_title + " repetition " + repetition_value[1])
                    plt.title("\n".join(wrap(title + " Actual vs Predicted log-plot ", 29)))
                    ax.set_xlabel('Log10 Actual ' + str(original_variable))
                    ax.set_ylabel('Log10 Predicted ' + str(original_variable))
                else:
                    title = (change_title + " repetition " + repetition_value[1])
                    plt.title("\n".join(wrap(title + " Actual vs Predicted plot ", 29)))
                    ax.set_xlabel('Actual ' + str(original_variable))
                    ax.set_ylabel('Predicted ' + str(original_variable))
                plt.savefig(save_path + "created_figs/" + experiment +
                            "_Actual_vs_Predicted_plot_rep_" + repetition + '.png', bbox_inches="tight")
                plt.cla()
                # plt.show()

    elif "xlsx" in ext:
        original_data_frame = pd.read_excel(data_set_file_path).reset_index()
        # # Remove all Experiments with Anomalous Mg values
        # original_data_frame = original_data_frame[~original_data_frame['Paper Number'].isin(['8', '46', '48',
        #                                                                                      '59', '71', '85',
        #                                                                                      '96', '98'])]
        y = original_data_frame[original_variable]
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

                # Save folds actual vs predicts -- DIAGNOSTIC --
                fig, ax = plt.subplots()
                ax.scatter(real_list, pred_list, alpha=alpha)
                if log is True:
                    ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    title = (change_title + " repetition " + repetition_value[1])
                    plt.title("\n".join(wrap(title + " Actual vs Predicted log-plot ", 29)))
                else:
                    title = (change_title + " repetition " + repetition_value[1])
                    plt.title("\n".join(wrap(title + " Actual vs Predicted plot ", 29)))
                ax.set_xlabel('Actual ' + str(original_variable))
                ax.set_ylabel('Predicted ' + str(original_variable))
                plt.savefig(save_path + "created_figs/" + experiment +
                            "_Actual_vs_Predicted_plot_rep_" + repetition + '.png', bbox_inches="tight")
                plt.cla()
                # plt.show()
        return


def plot_residual_values(path_to_file, original_data, change_title, index_value_chosen, original_variable, log, alpha,
                         save_path):
    all_files = sorted(glob.glob(path_to_file + experiment + "_actual_vs_pred_rep_" + "*" +
                                 "best_all_folds.csv"), key=numerical_sort)
    if len(all_files) < 1:
        print(" ")
        print("FIX FILE PATH FOR PREDICTED VS ACTUAL")

    # if original file is required
    data_set_file_path = cwd + original_data
    # check ext
    split_name = original_data.split(".")
    ext = split_name[1]
    if "csv" in ext:
        counter = 0
        for file in all_files:
            counter += 1
            new_path, filename = os.path.split(file)
            rep_regex = r"rep_\d*"
            rep_num = re.search(rep_regex, filename)
            repetition = str(rep_num.group())
            repetition_value = repetition.split("_")
            if int(repetition_value[1]) == int(df_high_index) or int(repetition_value[1]) == int(df_low_index):
                total_actual_vs_pred_df = pd.read_csv(file)
                total_actual_vs_pred_df = total_actual_vs_pred_df.drop(columns="Unnamed: 0")

                if log is True:
                    total_actual_vs_pred_df['Residual'] = total_actual_vs_pred_df['prediction'].apply(np.log10)
                    total_actual_vs_pred_df['Actual'] = total_actual_vs_pred_df['reality'].apply(np.log10)

                    plt.figure()
                    # Total Actual vs Predicted RESIDUALS across all folds -- IMPORTANT --
                    sns.residplot(y='Residual',
                                  x='Actual',
                                  data=total_actual_vs_pred_df,
                                  scatter_kws={'alpha': alpha})

                    plt.title(
                        "\n".join(wrap(change_title + " repetition " + repetition_value[1] + " Residual Log-Plot", 29)))
                    plt.xlabel('Log10 Actual Values of ' + original_variable)
                    plt.ylabel('Log10 Residual Values of ' + original_variable)
                    plt.savefig(save_path + "created_figs/" + experiment +
                                "_Residual_Log_plot_rep_" + repetition + '.png', bbox_inches="tight")
                    plt.cla()
                    # plt.show()
                    # plt.pause(1)
                else:
                    total_actual_vs_pred_df['Residual'] = total_actual_vs_pred_df['prediction']
                    total_actual_vs_pred_df['Actual'] = total_actual_vs_pred_df['reality']

                    plt.figure()
                    # Total Actual vs Predicted RESIDUALS across all folds -- IMPORTANT --
                    sns.residplot(y='Residual',
                                  x='Actual',
                                  data=total_actual_vs_pred_df,
                                  scatter_kws={'alpha': alpha})

                    plt.title(
                        "\n".join(wrap(change_title + " repetition " + repetition_value[1] + " Residual Plot", 29)))
                    plt.xlabel('Actual Values of ' + original_variable)
                    plt.ylabel('Residual Values of ' + original_variable)
                    plt.savefig(save_path + "created_figs/" + experiment +
                                "_Residual_plot_rep_" + repetition + '.png', bbox_inches="tight")
                    plt.cla()
                    # plt.show()
                    # plt.pause(1)


# begin to plot shap values
'''
Create the plots for the best and worst result of repetition experiments
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


def simplified_concurrent_shuffler(split_list, seed_number_chosen, label):
    """This is a ChatGPT simplified code, where it is o(n) rather than o(n^2).
    This is because it uses one combined list instead of two separate lists."""
    # get the values of the specified label column in split_list
    value = (split_list.loc[:, label])
    # list to store start and end indices of sections where values are decreasing
    indexes = []
    # list of current feature_of_interest values
    current_list = [value[feature_of_interest] for feature_of_interest in range(len(value))]
    # list of next feature_of_interest values
    next_list = [value[feature_of_interest + 1] for feature_of_interest in range(len(value) - 1)]
    # add 0 to the end of each list to ensure they have the same length
    current_list.insert(len(current_list), 0)
    next_list.insert(len(next_list), 0)
    # combine the current and next feature_of_interest values into pairs
    combined_list = list(zip(current_list, next_list))
    # iterate over the pairs and store the indices where the values are decreasing
    for i, (current_feature_of_interest, next_feature_of_interest) in enumerate(combined_list):
        if current_feature_of_interest > next_feature_of_interest:
            # if the difference between the current and previous index is greater than 1,
            # shuffle the rows in that section
            if i - indexes[-1] > 1:
                indexes.append(i)
                split_list[indexes[-1]:i].sample(frac=1, random_state=seed_number_chosen)
            # if the difference is 1 or less, store the index without shuffling the rows
            else:
                indexes.append(i)
    # return the shuffled or unshuffled split_list
    return split_list


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


def simplified_stratified_dataset(data, cv_chosen, label, seed_number, index_required):
    """
    stratified approach is to sort the Y variable, proceed to ensure that the data
    is put forwards to be split in even distribution.
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
    # #### SET THIS TO FOLD COUNT #####
    cv_count = cv_chosen
    ###################################
    total_y_list = total_list[:cv_count]
    # removed CV loop
    if index_required:
        final_data_set = pd.DataFrame(total_y_list, columns=names)
        index = final_data_set['index']
    else:
        final_data_set = pd.DataFrame(total_y_list, columns=names)
        index = []
    return final_data_set, index


def create_bolstered_regression_model_shap_plots_from_pickle(path_to_file, original_data, change_title,
                                                             original_variable,
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
    # Remove all Experiments with Anomalous Mg values
    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

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

            seed_number = seed_numbers[int(repetition_value[1])]
            final_data_set, index = stratified_dataset(data_set, number_of_splits, original_variable, seed_number,
                                                       index_required=True)
            # #### remove Correlated Features ####
            y = final_data_set[original_variable]
            # drop correlated features
            # corr_matrix = final_data_set.corr().abs()
            # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            # drop_value = 0.80
            # to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
            # were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
            # final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

            # Change the X and Y data, experiment name, scoring used

            # change to this for MAGNESIUM
            # x = final_data_set.drop(columns=['Magnesium (mM)', 'index', 'Unnamed: 0',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')
            #
            print(final_data_set.columns)
            # # change to this for STAPLE
            columns_to_drop_list = ['Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
                                    'Experiment Number', 'Boric Acid (mM)',
                                    'Buffer Name', 'Scaffold Name',
                                    'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                                    'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                    'average_shortest_path', 'average_clustering_coefficient',
                                    'average_degree', 'average_betweenness_centrality',
                                    'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                                    'graph_reciprocity', 's-metric', 'wiener_index']

            # # Change the X and Y data, experiment name, scoring used
            # x = final_data_set.drop(columns=columns_to_drop_list, errors='ignore')

            # change to this for MAGNESIUM REV-NANO
            # x = final_data_set.drop(columns=['Magnesium (mM)', 'index', 'Unnamed: 0',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
            #                                  'avg_neighbour_total', 'graph_density', 'graph_transitivity',
            #                                  'average_shortest_path', 'average_clustering_coefficient',
            #                                  'average_degree', 'average_betweenness_centrality',
            #                                  'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
            #                                  'graph_reciprocity', 's-metric', 'wiener_index'], errors='ignore')

            predictor = original_variable

            # columns_to_drop_list = ['Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
            #                         'Experiment Number', 'Boric Acid (mM)',
            #                         'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
            #                         'avg_neighbour_total', 'graph_density', 'graph_transitivity',
            #                         'average_shortest_path', 'average_clustering_coefficient',
            #                         'average_degree', 'average_betweenness_centrality',
            #                         'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
            #                         'graph_reciprocity', 's-metric', 'wiener_index']
            #
            # columns_to_drop_list = ['Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
            #                         'Experiment Number', 'Boric Acid (mM)',
            #                         'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
            #                         'avg_neighbour_total', 'graph_density', 'graph_transitivity',
            #                         'average_shortest_path', 'average_clustering_coefficient',
            #                         'average_degree', 'average_betweenness_centrality',
            #                         'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
            #                         'graph_reciprocity', 's-metric', 'wiener_index']
            #

            # columns_to_drop_list = ['index', 'Unnamed: 0',
            #                         'Experiment Number', 'Boric Acid (mM)',
            #                         'Acetic acid (mM)', 'Acetate (mM)']

            # Change the X and Y data, experiment name, scoring used
            x = final_data_set.drop(columns=columns_to_drop_list, errors='ignore')

            # in case of bolstered:

            # split the original data set instances and the bolstered instances
            data_set_file_path_og = cwd + original_data
            data_set_og = pd.read_csv(data_set_file_path_og)
            # Remove all Experiments with NaN Outcome
            data_set_og = data_set_og[~data_set_og[predictor].isin([np.NaN])]
            data_set_og, index_2 = stratified_dataset(data_set_og, number_of_splits, predictor, seed_number,
                                                      index_required=True)

            # Remove all Experiments with Anomalous Mg values
            og_data_set = data_set_og[
                ~data_set_og['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
            og_data_set = og_data_set.drop(columns=columns_to_drop_list, errors='ignore')

            print("OG DATA SET:", og_data_set.shape)
            print("OG DATA COLUMNS:", og_data_set.columns)

            bolster_data_set = x[x['Paper Number'] >= 101]

            # retrieve y values for the test sets
            og_data_set_y = og_data_set[predictor]
            bolster_data_set_y = bolster_data_set[predictor]

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
            og_data_set = og_data_set.drop(columns=["Paper Number", predictor], errors='ignore')
            bolster_data_set = bolster_data_set.drop(columns=["Paper Number", predictor], errors='ignore')

            # print the values
            print("this is the og data set:", len(og_data_set))
            print("this is the bolster data set:", len(bolster_data_set))

            # you would then create the train AND test sets from the OG data set.
            # you would then create a train only SPLIT of the bolster set (shuffle-split into n=3)
            # to do this you could just use the train-test split and combine the tr-val index values, add to the OG.

            experiment_name = experiment
            scoring = "r2"

            # outer_cv = KFold(n_splits=number_of_splits, shuffle=False)
            # data_splits = list(outer_cv.split(x, y))
            outer_cv = KFold(n_splits=number_of_splits, shuffle=False)

            data_splits = list(outer_cv.split(og_data_set, og_data_set_y))
            data_splits_2 = list(outer_cv.split(bolster_data_set, bolster_data_set_y))
            inner_loop_rep = 0

            #  Creates data splits
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

                    # # Apply SHAP
                    shap_values = explain.shap_values(df_test_shap)
                    expected_value = explain.expected_value

                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_column_numpy, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values, df_test_shap,
                                      feature_names=x_column_numpy, max_display=5, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    plt.savefig(save_path + "shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

                else:
                    df_test_shap = transformed_data
                    shap_values = explain.shap_values(df_test_shap)
                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values, df_test_shap,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

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
                print(transformer._feature_names_in)

                # May need to use toarray to work depending on num of features
                try:
                    # print(x.columns)
                    # x.drop(columns="Staple Molarity (nM)", inplace=True)
                    transformed_data = transformer.transform(x)
                    # Gather column names
                    x_transformed_columns_outer = get_transformer_feature_names(transformer)
                    print(len(x_transformed_columns_outer))
                    df_test = pd.DataFrame(transformed_data, columns=x_transformed_columns_outer)
                except:
                    # x.drop(columns="Staple Molarity (nM)", inplace=True)
                    print("running exception")
                    # print(transformer._feature_names_in)
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
                    shap.summary_plot(shap_values, df_test_fe,
                                      feature_names=x_column_numpy, max_display=5, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    plt.savefig(save_path + "shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

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
                    shap.summary_plot(shap_values, df_test,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    plt.savefig(save_path + "shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()


def create_regression_model_shap_plots_from_pickle(path_to_file, original_data, change_title,
                                                   original_variable, number_of_splits, save_path):
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
    # Remove all Experiments with Anomalous Mg values
    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

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

            seed_number = seed_numbers[int(rep)]
            final_data_set, index = stratified_dataset(data_set, number_of_splits, original_variable, seed_number,
                                                       index_required=True)
            # #### remove Correlated Features ####
            y = final_data_set[original_variable]
            # drop correlated features
            corr_matrix = final_data_set.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_value = 0.80
            to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
            print(to_drop)
            were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
            final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

            # Change the X and Y data, experiment name, scoring used

            # change to this for MAGNESIUM
            x = final_data_set.drop(columns=['Magnesium (mM)', 'index', 'Unnamed: 0',
                                             'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                             'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')

            # # change to this for STAPLE
            # x = final_data_set.drop(columns=['Staple Molarity (nM)', 'Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
            #                                  'avg_neighbour_total', 'graph_density', 'graph_transitivity',
            #                                  'average_shortest_path', 'average_clustering_coefficient',
            #                                  'average_degree', 'average_betweenness_centrality',
            #                                  'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
            #                                  'graph_reciprocity', 's-metric', 'wiener_index'], errors='ignore')

            # change to this for STAPLE-NoRevNano
            # x = final_data_set.drop(columns=['Staple Molarity (nM)', 'Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
            #                                  'Buffer Name', 'Scaffold Name',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
            #                                  'avg_neighbour_total', 'graph_density', 'graph_transitivity',
            #                                  'average_shortest_path', 'average_clustering_coefficient',
            #                                  'average_degree', 'average_betweenness_centrality',
            #                                  'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
            #                                  'graph_reciprocity', 's-metric', 'wiener_index'], errors='ignore')

            # change to this for STAPLE-RevNano
            # x = final_data_set.drop(columns=['Staple Molarity (nM)', 'Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
            #                                  'Buffer Name', 'Scaffold Name',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')

            # # change to this for MAGNESIUM No REV-NANO
            # x = final_data_set.drop(columns=['Magnesium (mM)', 'index', 'Unnamed: 0',
            #                                  'Buffer Name', 'Scaffold Name',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
            #                                  'avg_neighbour_total', 'graph_density', 'graph_transitivity',
            #                                  'average_shortest_path', 'average_clustering_coefficient',
            #                                  'average_degree', 'average_betweenness_centrality',
            #                                  'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
            #                                  'graph_reciprocity', 's-metric', 'wiener_index'], errors='ignore')

            experiment_name = experiment

            outer_cv = KFold(n_splits=number_of_splits, shuffle=False)
            data_splits = list(outer_cv.split(x, y))

            inner_loop_rep = 0
            #  Creates data splits
            for tr_idx, val_idx in data_splits:
                inner_loop_rep += 1

                # Removes correct answers from New_X, creates train-test sets
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

                    # # Apply SHAP
                    shap_values = explain.shap_values(df_test_shap)
                    expected_value = explain.expected_value

                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_column_numpy, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # # analyse features shap values
                    # shap_value_analysis = pd.DataFrame(shap_values, columns=x_column_numpy)
                    # # apply logic to remove certain columns
                    # shap_value_new = shap_value_analysis.loc[:, (shap_value_analysis != 0).any(axis=0)]
                    # columns_remain = shap_value_new.columns
                    # df_test_shap = df_test_shap[columns_remain]
                    # shap_value_array = explain.shap_values(df_test_shap)
                    #
                    # print(columns_remain)
                    # print(df_test_shap.columns)

                    # plots
                    shap.summary_plot(shap_values, df_test_shap,
                                      feature_names=x_column_numpy, max_display=5, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    # plt.show()
                    # plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                    #             + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.savefig(save_path + "shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

                else:
                    df_test_shap = transformed_data
                    shap_values = explain.shap_values(df_test_shap)
                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values, df_test_shap,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

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
                    shap.summary_plot(shap_values, df_test_fe,
                                      feature_names=x_column_numpy, max_display=5, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    # plt.show()
                    # plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                    #             + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.savefig(save_path + "shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)

                    plt.close()
                    plt.cla()

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
                    shap.summary_plot(shap_values, df_test,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    # plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                    #             + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()


def create_revnano_regression_model_shap_plots_from_pickle(path_to_file, original_data, change_title,
                                                           original_variable, number_of_splits, save_path):
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
    # Remove all Experiments with Anomalous Mg values
    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

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
            seed_number = seed_numbers[int(repetition_value[1])]
            final_data_set, index = stratified_dataset(data_set, number_of_splits, original_variable, seed_number,
                                                       index_required=True)

            ###### remove Correlated Features ####
            y = final_data_set[original_variable]

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
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_value = 0.80
            to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
            were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
            final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

            # Change the X and Y data, experiment name, scoring used
            # x = final_data_set.drop(columns=['Staple Molarity (nM)', 'Scaffold to Staple Ratio', 'index', 'Unnamed: 0',
            #                                  'Buffer Name', 'Scaffold Name',
            #                                  'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
            #                                  'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')
            #

            x = final_data_set.drop(columns=['Magnesium (mM)', 'index', 'Unnamed: 0',
                                             'Buffer Name', 'Scaffold Name',
                                             'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                             'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')

            x = pd.merge(x, x_stored, left_index=True, right_index=True)
            print(x.shape)

            experiment_name = experiment

            outer_cv = KFold(n_splits=number_of_splits, shuffle=False)
            data_splits = list(outer_cv.split(x, y))

            inner_loop_rep = 0
            #  Creates data splits
            for tr_idx, val_idx in data_splits:
                inner_loop_rep += 1

                # Removes correct answers from New_X, creates train-test sets
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

                    # # Apply SHAP
                    shap_values = explain.shap_values(df_test_shap)
                    expected_value = explain.expected_value

                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_column_numpy, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values, df_test_shap,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

                else:
                    df_test_shap = transformed_data
                    shap_values = explain.shap_values(df_test_shap)
                    # Feature Importance for best fitted set for fold -- DIAGNOSTIC --
                    feature_importance = best.named_steps['estimator'].feature_importances_
                    feature_importance_list = list(zip(x_transformed_columns_outer, feature_importance))
                    importance_df = pd.DataFrame(feature_importance_list, columns=['feature_name', 'importance_value'])
                    importance_df.sort_values(by=['importance_value'], inplace=True, ascending=False)

                    # plots
                    shap.summary_plot(shap_values, df_test_shap,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " fold " + str(inner_loop_rep))
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()

                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_fold_" + str(inner_loop_rep) + ".png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

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
                    shap.summary_plot(shap_values, df_test_fe,
                                      feature_names=x_column_numpy, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()

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
                    shap.summary_plot(shap_values, df_test,
                                      feature_names=x_transformed_columns_outer, show=False)
                    title = (change_title + " shap summary plot repetition " + str(rep)
                             + " whole data set")
                    plt.title("\n".join(wrap(title, 44)))
                    plt.tight_layout()
                    plt.savefig(save_path + "created_figs/shap_summary_plot_rep_" + str(rep)
                                + "_whole_dataset.png", bbox_inches='tight', dpi=100)
                    plt.close()
                    plt.cla()


def get_fold_scores_for_reps(path_to_file, path_to_score):
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
                    columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]
                    fold_score_df = pd.read_csv(fold_score_file, names=columns)
                    print(fold_score_df)


def get_fold_scores_and_preds_for_reps(path_to_file, path_to_score, original_data, original_variable, save_path,
                                       plot_worst_bar, plot_worst_hist, residual_limit):
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
            if int(repetition_value[1]) == int(df_low_index):
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
                    columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]
                    fold_score_df = pd.read_csv(fold_score_file, names=columns)
                    fold_score_df.insert(loc=0, column='fold', value=fold_score_file_number)
                    fold_score_table_df = pd.concat([fold_score_table_df, fold_score_df], ignore_index=True)
                fold_score_table_df.drop(columns=["MedianAE"], inplace=True)
                fold_score_table_df.to_csv(save_path + "fold_interrogation/"
                                                       "fold_scores_of_worst_rep_"
                                           + str(repetition_value[1]) + ".csv")

                # gather the fold predictions and try find residuals / worst instances
                fold_pred_files = sorted(glob.glob(path_to_score + experiment + "_actual_vs_pred_rep_" +
                                                   str(repetition_value[1]) +
                                                   "_fold_*.csv"), key=numerical_sort)
                if len(fold_pred_files) < 1:
                    print(" ")

                # gather the fold parameters chosen by grid-search CV
                fold_param_files = sorted(glob.glob(path_to_score + experiment + "_best_inner_model_parameters_"
                                                    + str(repetition_value[1]) +
                                                    "_fold_*" + ".txt"), key=numerical_sort)
                if len(fold_param_files) < 1:
                    print(" ")

                fold_number = 0

                fold_score_df_concat = pd.DataFrame()

                for i in range(number_of_splits):
                    fold_pred_file = fold_pred_files[i]
                    print(fold_pred_file)
                    fold_score_df = pd.read_csv(fold_pred_file)
                    fold_score_df['absolute_residual'] = abs(fold_score_df['prediction'] - fold_score_df['reality'])
                    fold_score_df['residual'] = fold_score_df['reality'] - fold_score_df['prediction']
                    fold_score_df_worst_residuals_to_save = fold_score_df.loc[fold_score_df["absolute_residual"]
                                                                              >= residual_limit]
                    fold_score_df_worst_residuals_to_save.drop(columns=["Unnamed: 0"], inplace=True)
                    fold_score_df_worst_residuals_to_save.sort_values(by=["residual"], inplace=True, ascending=False)

                    # map back to data set
                    # Set Random Seed
                    seed_numbers = list(range(1, 1000))
                    random.Random(42).shuffle(seed_numbers)
                    seed_number = seed_numbers[int(repetition_value[1])]

                    # if original file is required
                    data_set_file_path = cwd + original_data

                    # import the data set
                    data_set = pd.read_csv(data_set_file_path)
                    # Remove all Experiments with Anomalous Mg values
                    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

                    # Remove all Experiments with NaN Outcome
                    data_set = data_set[~data_set[original_variable].isin([np.NaN])]
                    final_data_set, index = stratified_dataset(data_set, number_of_splits, original_variable,
                                                               seed_number, index_required=True)
                    y = final_data_set[[original_variable]]
                    x = final_data_set.drop(columns=[original_variable])

                    # calculate the outer_cv
                    outer_cv = KFold(n_splits=number_of_splits, shuffle=False)

                    data_splits = list(outer_cv.split(x, y))
                    inner_loop_rep = 0

                    #  Creates data splits
                    inner_loop_rep += 1
                    # split the og data to create test set, and the train set for combining with bolster
                    og_data_split = data_splits[fold_number]
                    # train_idx = og_data_split[0]
                    val_idx = og_data_split[1]

                    # Create data splits of original dataset and y values
                    X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

                    # print(X_test.head(5))
                    # print(y_test.head(5))
                    # Concatenate the y values and original data
                    fold_score_df["og_dataset_actual"] = y_test.values
                    fold_score_df["og_dataset_index"] = y_test.index
                    x_test_df = pd.DataFrame(X_test)

                    fold_score_df = pd.merge(fold_score_df, x_test_df, left_on="og_dataset_index", right_index=True)
                    fold_score_df.sort_values(by=["residual"], inplace=True, ascending=False)
                    fold_score_df = fold_score_df.loc[fold_score_df["absolute_residual"] >= residual_limit]
                    fold_score_df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True, errors='ignore')
                    fold_score_df_worst_residuals_to_save = fold_score_df[["Experiment Number", "prediction", "reality",
                                                                           "absolute_residual", "residual"]]
                    fold_score_df_worst_residuals_to_save.to_csv(save_path + "fold_interrogation/"
                                                                             "worst_residuals_of_worst_rep_"
                                                                 + str(repetition_value[1]) +
                                                                 "_fold_" + str(fold_number + 1) + ".csv")
                    fold_score_df.to_csv(save_path + "fold_interrogation/worst_residual_instances_of_worst_rep_"
                                         + str(repetition_value[1]) + "_fold_" + str(fold_number + 1) + ".csv")
                    # print(fold_score_df)
                    fold_number += 1

                    fold_score_df_concat = pd.concat([fold_score_df_concat, fold_score_df])

                if plot_worst_bar is True:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    """Compare the descriptive statistics of the data set using Pandas"""

                    plt.style.use('seaborn-whitegrid')

                    # if original file is required
                    data_set_file_path = cwd + original_data
                    # import the data set
                    data_set = pd.read_csv(data_set_file_path)
                    # Remove all Experiments with Anomalous Mg values
                    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
                    # Remove all Experiments with Experiment Number values in tough instance set
                    tough_instance_list = fold_score_df_concat[["Experiment Number"]].values
                    data_set = data_set[~data_set['Experiment Number'].isin([tough_instance_list])]
                    data_set.drop(columns=["Unnamed: 0", 'Experiment Number', 'Paper Number', 'Scaffold Name',
                                           'Characterised By', 'Scaffold to Staple Ratio',
                                           'Constructed By', 'Buffer Name', 'MgCl2 Used',
                                           'Magnesium Acetate Used', 'Magnesium (mM)',
                                           'Boric Acid (mM)', 'Acetate (mM)', 'Acetic acid (mM)'],
                                  errors='ignore', inplace=True)
                    data_set.drop(columns=[predictor_of_interest], errors='ignore', inplace=True)
                    predictor_list = data_set.columns
                    print(predictor_list)
                    print(fold_score_df_concat.columns)

                    for predictor in predictor_list:
                        print(predictor)

                        # plot histograms - not suitable for this data really unless viewing pure distribution
                        series_1 = data_set[[predictor]]
                        series_2 = fold_score_df_concat[[predictor]]

                        # BAR PLOT # # # #
                        # plot the bar plot instead of Magnesium (mM)
                        series_1_values_df = series_1.value_counts().rename_axis(
                            'Quantity of ' + predictor).reset_index(
                            name='Instances')
                        series_1_values_df.sort_values(by="Quantity of " + predictor, inplace=True)
                        series_1_values_df["Group"] = 'Other Instances from Dataset ' + predictor

                        series_2_values_df = series_2.value_counts().rename_axis(
                            'Quantity of ' + predictor).reset_index(
                            name='Instances')
                        series_2_values_df.sort_values(by="Quantity of " + predictor, inplace=True)
                        series_2_values_df["Group"] = 'Worst Residual Instances from Folds ' + predictor

                        appended_df = series_1_values_df.append(series_2_values_df)
                        appended_df.sort_values(by="Quantity of " + predictor, inplace=True)
                        # appended_df.plot(kind="bar", x="Quantity of Magnesium (mM)", y="Instances", label="Group")
                        ax = appended_df.pivot("Quantity of " + predictor, "Group", "Instances").plot(kind='barh')
                        tough_instance_title = predictor_of_interest + " model " + experiment + \
                                               " worst repetition tough_instances " + "quantity of " + predictor
                        plt.title("\n".join(wrap(tough_instance_title, 48)))
                        plt.xlabel("Instance Counts")
                        plt.ylabel(predictor)
                        plt.tight_layout()
                        plt.tick_params(axis='both', which='major', labelsize=8)
                        # for p in ax.patches:
                        #     ax.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10),
                        #                 textcoords='offset points')
                        plt.legend(fontsize=8)
                        plt.savefig(save_path + predictor_of_interest + "_model_" + experiment +
                                    "_worst_rep_tough_instances_" +
                                    predictor + "_comparison_bar.png",
                                    bbox_inches='tight')
                        plt.cla()
                        plt.close()

                if plot_worst_hist is True:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    """Compare the descriptive statistics of the data set using Pandas"""

                    plt.style.use('seaborn-whitegrid')

                    # if original file is required
                    data_set_file_path = cwd + original_data
                    # import the data set
                    data_set = pd.read_csv(data_set_file_path)
                    # Remove all Experiments with Anomalous Mg values
                    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]
                    # Remove all Experiments with Experiment Number values in tough instance set
                    tough_instance_list = fold_score_df_concat[["Experiment Number"]].values
                    data_set = data_set[~data_set['Experiment Number'].isin([tough_instance_list])]
                    data_set.drop(columns=["Unnamed: 0", 'Experiment Number', 'Paper Number', 'Scaffold Name',
                                           'Characterised By', 'Scaffold to Staple Ratio',
                                           'Constructed By', 'Buffer Name', 'MgCl2 Used',
                                           'Magnesium Acetate Used', 'Magnesium (mM)',
                                           'Boric Acid (mM)', 'Acetate (mM)', 'Acetic acid (mM)'],
                                  errors='ignore', inplace=True)
                    data_set.drop(columns=[predictor_of_interest], errors='ignore', inplace=True)
                    predictor_list = data_set.select_dtypes(include=['int64', 'float64']).columns
                    print(predictor_list)
                    print(fold_score_df_concat.columns)

                    for predictor in predictor_list:
                        print(predictor)

                        # plot histograms - not suitable for this data really unless viewing pure distribution
                        series_1_values_df = data_set[[predictor]]
                        series_2_values_df = fold_score_df_concat[[predictor]]

                        # HIST PLOT # # # #
                        plt.hist(series_1_values_df, bins=25, alpha=0.5,
                                 label='Other Instances from Dataset ' + predictor, color='b')
                        plt.hist(series_2_values_df, bins=25, alpha=0.5,
                                 label='Worst Residual Instances from Folds ' + predictor, color='r')
                        tough_instance_title = predictor_of_interest + " model " + experiment + \
                                               " worst repetition tough_instances " + "quantity of " + predictor
                        plt.title("\n".join(wrap(tough_instance_title, 48)))
                        plt.xlabel("Frequency")
                        plt.ylabel(predictor)
                        plt.tight_layout()
                        plt.tick_params(axis='both', which='major', labelsize=8)
                        # for p in ax.patches:
                        #     ax.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10),
                        #                 textcoords='offset points')
                        plt.legend(fontsize=8)
                        plt.savefig(save_path + predictor_of_interest + "_model_" + experiment +
                                    "_worst_rep_tough_instances_" +
                                    predictor + "_comparison_hist.png",
                                    bbox_inches='tight')
                        plt.cla()
                        plt.close()

            # apply to the best repetitions
            if int(repetition_value[1]) == int(df_high_index):
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
                    columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]
                    fold_score_df = pd.read_csv(fold_score_file, names=columns)
                    fold_score_df.insert(loc=0, column='fold', value=fold_score_file_number)
                    fold_score_table_df = pd.concat([fold_score_table_df, fold_score_df], ignore_index=True)
                fold_score_table_df.drop(columns=["MedianAE"], inplace=True)
                fold_score_table_df.to_csv(save_path + "fold_interrogation/"
                                                       "fold_scores_of_best_rep_"
                                           + str(repetition_value[1]) + ".csv")

                # gather the fold predictions and try find residuals /worst instances
                fold_pred_files = sorted(glob.glob(path_to_score + experiment + "_actual_vs_pred_rep_" +
                                                   str(repetition_value[1]) +
                                                   "_fold_*.csv"), key=numerical_sort)
                if len(fold_pred_files) < 1:
                    print(" ")

                # gather the fold parameters chosen by grid-search CV
                fold_param_files = sorted(glob.glob(path_to_score + experiment + "_best_inner_model_parameters_"
                                                    + str(repetition_value[1]) +
                                                    "_fold_*" + ".txt"), key=numerical_sort)
                if len(fold_param_files) < 1:
                    print(" ")

                fold_number = 0
                for i in range(number_of_splits):
                    fold_pred_file = fold_pred_files[i]
                    print(fold_pred_file)
                    fold_score_df = pd.read_csv(fold_pred_file)
                    fold_score_df['absolute_residual'] = abs(fold_score_df['prediction'] - fold_score_df['reality'])
                    fold_score_df['residual'] = fold_score_df['reality'] - fold_score_df['prediction']
                    fold_score_df_worst_residuals_to_save = fold_score_df.loc[fold_score_df["absolute_residual"]
                                                                              >= residual_limit]
                    fold_score_df_worst_residuals_to_save.drop(columns=["Unnamed: 0"], inplace=True)
                    fold_score_df_worst_residuals_to_save.sort_values(by=["residual"], inplace=True, ascending=False)

                    # map back to data set
                    # Set Random Seed
                    seed_numbers = list(range(1, 1000))
                    random.Random(42).shuffle(seed_numbers)
                    seed_number = seed_numbers[int(repetition_value[1])]

                    # if original file is required
                    data_set_file_path = cwd + original_data

                    # import the data set
                    data_set = pd.read_csv(data_set_file_path)
                    # Remove all Experiments with Anomalous Mg values
                    data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

                    # Remove all Experiments with NaN Outcome
                    data_set = data_set[~data_set[original_variable].isin([np.NaN])]
                    final_data_set, index = stratified_dataset(data_set, number_of_splits, original_variable,
                                                               seed_number, index_required=True)
                    y = final_data_set[[original_variable]]
                    x = final_data_set.drop(columns=[original_variable])

                    # calculate the outer_cv
                    outer_cv = KFold(n_splits=number_of_splits, shuffle=False)

                    data_splits = list(outer_cv.split(x, y))
                    inner_loop_rep = 0

                    #  Creates data splits
                    inner_loop_rep += 1
                    # split the og data to create test set, and the train set for combining with bolster
                    og_data_split = data_splits[fold_number]
                    # train_idx = og_data_split[0]
                    val_idx = og_data_split[1]

                    # Create data splits of original dataset and y values
                    X_test, y_test = x.iloc[val_idx], y.iloc[val_idx]

                    # print(X_test.head(5))
                    # print(y_test.head(5))
                    # Concatenate the y values and original data
                    fold_score_df["og_dataset_actual"] = y_test.values
                    fold_score_df["og_dataset_index"] = y_test.index
                    x_test_df = pd.DataFrame(X_test)

                    fold_score_df = pd.merge(fold_score_df, x_test_df, left_on="og_dataset_index", right_index=True)
                    fold_score_df.sort_values(by=["residual"], inplace=True, ascending=False)
                    fold_score_df = fold_score_df.loc[fold_score_df["absolute_residual"] >= residual_limit]
                    fold_score_df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True, errors='ignore')

                    fold_score_df_worst_residuals_to_save = fold_score_df[["Experiment Number", "prediction", "reality",
                                                                           "absolute_residual", "residual"]]
                    fold_score_df_worst_residuals_to_save.to_csv(save_path + "fold_interrogation/"
                                                                             "worst_residuals_of_best_rep_"
                                                                 + str(repetition_value[1]) +
                                                                 "_fold_" + str(fold_number + 1) + ".csv")

                    fold_score_df.to_csv(save_path + "fold_interrogation/worst_residual_instances_of_best_rep_"
                                         + str(repetition_value[1]) + "_fold_" + str(fold_number + 1) + ".csv")
                    # print(fold_score_df)
                    fold_number += 1


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
                        columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]
                        fold_score_df = pd.read_csv(fold_score_file, names=columns)
                        fold_score_df.insert(loc=0, column='fold', value=fold_score_file_number)
                        fold_score_table_df = pd.concat([fold_score_table_df, fold_score_df], ignore_index=True)
                    fold_score_table_df.drop(columns=["MedianAE"], inplace=True)
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
                        columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]
                        fold_score_df = pd.read_csv(fold_score_file, names=columns)
                        fold_score_df.insert(loc=0, column='fold', value=fold_score_file_number)
                        fold_score_table_df = pd.concat([fold_score_table_df, fold_score_df], ignore_index=True)
                    fold_score_table_df.drop(columns=["MedianAE"], inplace=True)
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

                        # print all the parameters possible:
                        for param, value in loaded_fold_model.get_params(deep=True).items():
                            if "depth" in param:
                                print(f"{param} -> {value}")
                            if "estimators" in param:
                                print(f"{param} -> {value}")
                            if "n_features_to_select" in param:
                                print(f"{param} -> {value}")

                        print(" ")
                        print("---next fold---")


# plot_combined_predicted_vs_actual(path_all, original_data, change_title, predictor_of_interest, log=True, alpha=0.4,
#                                   save_path=path)
# plot_residual_values(path_all, original_data, change_title, None, predictor_of_interest, log=True, alpha=0.2,
#                      save_path=path)
create_regression_model_shap_plots_from_pickle(path_all, original_data, change_title, predictor_of_interest,
                                               number_of_splits, quick_save)
# create_revnano_regression_model_shap_plots_from_pickle(path_all, original_data, change_title, predictor_of_interest,
#                                                        number_of_splits, path)
# create_bolstered_regression_model_shap_plots_from_pickle(path_all, original_data, change_title, predictor_of_interest,
#                                                          number_of_splits, quick_save)

# get_fold_scores_for_reps(path_all, path_metadata)  # don't use this one now

# get_fold_scores_and_preds_for_reps(path_all, path_metadata, original_data, predictor_of_interest, path,
#                                    False, True, 100)  # this allows tough instance analysis and plots

# get_params_for_reps(path_all, path_metadata, original_data, predictor_of_interest, path, path_pickle, high=True)

