import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

# Pandas and Numpy Options
cwd = os.getcwd()
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')


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


# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(1337).shuffle(seed_numbers)

# import the data set
data_set_file_path = cwd + '/subset_1_all_literature_high_cardinal_removed_ml_data_set.csv'
data_set = pd.read_csv(data_set_file_path)
# Remove all Experiments with Anomalous Mg values
data_set = data_set[~data_set['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

# which rep is performed
script = Path(__file__).stem

# get config file for genetic algorithm parameters
import json

hyper_parameters_dict = {}
with open('permutation_algorithm_config_used.json') as json_file:
    parsed = json.load(json_file)
    # print(json.dumps(parsed, indent=4, sort_keys=True))
    for i in parsed.get("ml_hyper_params"):
        hyper_parameters_dict = i
    json_file.close()

list_of_hyper_params = ["description", "dataset_filename", "splits", "experiment_name", "predictor"]

# get the hyper_parameters from loaded json config file
predictor = hyper_parameters_dict.get("predictor")
dataset_filename = hyper_parameters_dict.get("dataset_filename")
experiment_name = hyper_parameters_dict.get("experiment_name")
number_of_splits = hyper_parameters_dict.get("splits")


# Overall folder
name_of_root_folder = "./Experiment_Set_2_Magnesium_Permutations" + "/"
try:
    os.mkdir(name_of_root_folder)
    print("Created root folder")
except OSError as error:
    print(error)

for i in range(1, 31):

    # make folders in directory (os.mkdir)
    name_of_folder = name_of_root_folder + "Experiment_Set_2_Magnesium_Y_Permutation_" + str(i) + "/"
    try:
        os.mkdir(name_of_folder)
        print("Created Folder:" + str(i))
    except OSError as error:
        print(error)

    # add extra_trees_results folder
    name_of_results_folder = name_of_folder + "extra_trees_results/"
    try:
        os.mkdir(name_of_results_folder)
        print("Created Extra Trees Results:" + str(i))
    except OSError as error:
        print(error)

    # create folders to store meta data
    important = (name_of_results_folder + "/model_plots/")
    final_score_store = (name_of_results_folder + "/model_final_scores/")
    diagnostic = (name_of_results_folder + "/model_metadata/")
    pickle_store = (name_of_results_folder + "/model_pickle/")

    #  make directories for storing results
    os.makedirs(important)
    os.makedirs(final_score_store)
    os.makedirs(diagnostic)
    os.makedirs(pickle_store)

    rep = i
    number_of_splits = 3

    predictor = "Magnesium (mM)"
    seed_number = seed_numbers[rep]

    # ##### Apply Y-Permutation to the Data ####
    data_set_y = data_set[predictor]
    seed_numbers_for_y_perm = list(range(1, 100000))
    random.Random(1337).shuffle(seed_numbers_for_y_perm)  # Create Y-Permutation Random Seed
    y_perm_seed = seed_numbers_for_y_perm[i]
    y = data_set_y.sample(frac=1, random_state=y_perm_seed).reset_index(drop=True)

    # Re-add Y values back to the dataset after permutation
    data_set[predictor] = y.values

    # ##### Apply Stratification based on Predictor to the Data ####
    final_data_set, index = stratified_dataset(data_set, number_of_splits, predictor, seed_number, index_required=True)

    # take the stratified y, pre-pearson correlation removal of features
    y = final_data_set[predictor]

    # ##### remove Correlated Features ####
    corr_matrix = final_data_set.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_value = 0.80
    to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
    were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
    final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

    # Change the X and Y data, experiment name, scoring used
    x = final_data_set.drop(columns=[predictor, 'index', 'Unnamed: 0',
                                     'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                     'Acetic acid (mM)', 'Acetate (mM)'], errors='ignore')

    # Store final dataset
    permuted_final_dataset = pd.merge(y, x, left_index=True, right_index=True)
    permuted_final_dataset.to_csv(name_of_folder + "y_permutation_magnesium_experiment_" + str(i) + "_dataset.csv")
    print("Stored y-permutation dataset:" + str(i))

    # Store the config for dataset
    copy_of_dict = hyper_parameters_dict
    copy_of_dict.update(dataset_filename="/y_permutation_magnesium_experiment_" + str(i) + "_dataset.csv")
    copy_of_dict.update(permutation_id=str(i))

    store_copy_of_dict = {
        'ml_hyper_params':
            [copy_of_dict]
    }
    with open(name_of_folder + 'permutation_algorithm_config_used.json', "w") as json_file:
        json_file.write(json.dumps(store_copy_of_dict, indent=4, sort_keys=False))
        json_file.close()

    print("Updated and Created Unique Permutation Config File:" + str(i))

    # copy the machine learning python file to the folders
    import shutil
    shutil.copy("extra_trees_RFE_y_permuted_magnesium.py", name_of_folder)

    # create 30 repetition files of the original python file
    for j in range(1, 31):
        shutil.copy("extra_trees_RFE_y_permuted_magnesium.py",
                    name_of_folder + "extra_trees_RFE_y_permuted_magnesium_" + str(j) + ".py")

    # copy the run ML bash file
    shutil.copy("run_jobs_ML.sh", name_of_folder)
    shutil.copy("launcher.sh", name_of_folder)
    print("Copied Relevant Files:" + str(i))
    print("Finished")
