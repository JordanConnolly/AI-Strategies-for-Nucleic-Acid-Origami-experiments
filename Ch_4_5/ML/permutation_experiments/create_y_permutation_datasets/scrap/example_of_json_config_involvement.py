# get config file for genetic algorithm parameters
import json

hyper_parameters_dict = {}
with open('../permutation_algorithm_config_used.json') as json_file:
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

# store the hyper_parameters locally
storage_path = "../extra_trees_results/"
with open(storage_path + 'permutation_algorithm_config_used_' + str(i) + '.json', "w") as json_file:
    json_file.write(json.dumps(hyper_parameters_dict, indent=4, sort_keys=False))
    json_file.close()


# create the permutation config files
for i in range(1, 31):
    copy_of_dict = hyper_parameters_dict
    copy_of_dict.update(key2="/y_permuted_magnesium_experiment_" + str(i) + "_dataset.csv")

    # store the hyper_parameters locally
    storage_path = "../extra_trees_results/"
    with open(storage_path + 'permutation_algorithm_config_used.json', "w") as json_file:
        json_file.write(json.dumps(copy_of_dict, indent=4, sort_keys=False))
        json_file.close()

