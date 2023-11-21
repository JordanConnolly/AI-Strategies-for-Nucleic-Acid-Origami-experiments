import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Pandas and Numpy Options
cwd = os.getcwd()
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

# Set Random Seed
seed_numbers = list(range(1, 1000))
random.Random(1337).shuffle(seed_numbers)

# import the data set
data_set_file_path = cwd + '/subset_1_all_literature_high_cardinal_removed_ml_data_set.csv'
data_set = pd.read_csv(data_set_file_path)

# remove 1 / 13 instances of thermal profile that are truly isothermal
data_set = data_set[~data_set['Thermal Profile'].isin(['Isothermal-without-initial-denaturation'])]

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
name_of_root_folder = "./Experiment_Set_2_thermal_Permutations" + "/"
try:
    os.mkdir(name_of_root_folder)
    print("Created root folder")
except OSError as error:
    print(error)

for i in range(1, 31):

    # make folders in directory (os.mkdir)
    name_of_folder = name_of_root_folder + "Experiment_Set_2_thermal_Y_Permutation_" + str(i) + "/"
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

    predictor = "Thermal Profile"
    seed_number = seed_numbers[rep]

    # Remove all Experiments with NaN Outcome
    final_data_set = data_set[~data_set[predictor].isin([np.NaN])]

    # ordinal encode target variable
    label_encoder = LabelEncoder()
    y = final_data_set[predictor]
    y = label_encoder.fit_transform(y)
    final_data_set['y'] = y

    # ##### Apply Y-Permutation to the Data ####
    data_set_y = final_data_set['y']
    seed_numbers_for_y_perm = list(range(1, 100000))
    random.Random(1337).shuffle(seed_numbers_for_y_perm)  # Create Y-Permutation Random Seed
    y_perm_seed = seed_numbers_for_y_perm[i]
    y = data_set_y.sample(frac=1, random_state=y_perm_seed).reset_index(drop=True)

    # Re-add Y values back to the dataset after permutation
    final_data_set['y'] = y.values

    # take the stratified y, pre-pearson correlation removal of features
    y = final_data_set['y']

    # ##### remove Correlated Features ####
    corr_matrix = final_data_set.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_value = 0.80
    to_drop = [column for column in upper.columns if any(upper[column] > drop_value)]
    were_dropped = [column for column in upper.columns if any(upper[column] < drop_value)]
    final_data_set = final_data_set.drop(final_data_set[to_drop], axis=1)

    # Change the X and Y data, experiment name, scoring used
    x = final_data_set.drop(columns=['y', 'Thermal Profile', 'Peak Temperature (oC)',
                                     'Base Temperature (oC)', 'Temperature Ramp (s)',
                                     'index', 'Unnamed: 0',
                                     'Paper Number', 'Experiment Number', 'Boric Acid (mM)',
                                     'Acetic acid (mM)', 'Acetate (mM)', 'nodes', 'edges',
                                     'avg_neighbour_total', 'graph_density', 'graph_transitivity',
                                     'average_shortest_path', 'average_clustering_coefficient',
                                     'average_degree', 'average_betweenness_centrality',
                                     'average_closeness_centrality', 'graph_assortivity', 'graph_diameter',
                                     'graph_reciprocity', 's-metric', 'wiener_index'], errors='ignore')

    # Store final dataset
    permuted_final_dataset = pd.merge(y, x, left_index=True, right_index=True)
    permuted_final_dataset.to_csv(name_of_folder + "y_permutation_thermal_experiment_" + str(i) + "_dataset.csv")
    print("Stored y-permutation dataset:" + str(i))

    # Store the config for dataset
    copy_of_dict = hyper_parameters_dict
    copy_of_dict.update(dataset_filename="/y_permutation_thermal_experiment_" + str(i) + "_dataset.csv")
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

    shutil.copy("extra_trees_RFE_y_permuted_thermal.py", name_of_folder)

    # create 30 repetition files of the original python file
    for j in range(1, 31):
        shutil.copy("extra_trees_RFE_y_permuted_thermal.py",
                    name_of_folder + "extra_trees_RFE_y_permuted_thermal_" + str(j) + ".py")

    # copy the run ML bash file
    shutil.copy("run_jobs_ML.sh", name_of_folder)
    shutil.copy("launcher.sh", name_of_folder)
    print("Copied Relevant Files:" + str(i))
    print("Finished")
