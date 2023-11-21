# # used for computational resource comparison
#
import array
import os
import random
import json
import numpy
from math import sqrt
import time

# deap code
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures  # <-------------------- import futures module from scoop

# ben's heuristic code
import jordan_extended
import iohandler
import metrics
import dna
import contactmap.iohandler as cio

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import itertools as IT

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import multiprocessing
from pathlib import Path

# get config file for genetic algorithm parameters
import json


hyper_parameters_dict = {}
with open('genetic_algorithm_config_1torelli.json') as json_file:
    parsed = json.load(json_file)
    # print(json.dumps(parsed, indent=4, sort_keys=True))
    for i in parsed.get("ga_hyper_params"):
        hyper_parameters_dict = i
    json_file.close()


def ordinal_encoder(data_to_encode):
    my_array = np.array(list(data_to_encode))
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['A', 'C', 'G', 'U']))
    integer_encoded = label_encoder.transform(my_array)
    return integer_encoded


def ordinal_decoder(data_to_decode):
    array_to_decode = np.array(list(data_to_decode))
    decoded_as_int = array_to_decode.astype(int)
    decoded = decoded_as_int.astype(str)
    decoded[decoded == "0"] = "A"  # A
    decoded[decoded == "1"] = "C"  # C
    decoded[decoded == "2"] = "G"  # G
    decoded[decoded == "3"] = "U"  # T
    decoded = "".join(decoded)
    return decoded


# time the script
start = time.time()

# import functions from Ben's Scoring Criteria Work
score_scaffold = jordan_extended.score_scaffold  # custom evaluator
generate_random_scaffold = jordan_extended.generate_random_scaffold  # for initialisation (generator)
generate_staples_from_scaffold = jordan_extended.generate_staples_from_scaffold  # for initialisation (generator)
precompute_lookups = jordan_extended.precompute_lookups  # for initialisation (generator)

# contactmap for the origami we are using, placed in the _input/ directory
origami_contactmap_filename = hyper_parameters_dict.get("origami_contactmap_filename")


# name storage location and experiment name
origami_used = origami_contactmap_filename.split(".")[0]
descriptor_storage_path = hyper_parameters_dict.get("storage_path")
storage_path = "./output_GA/" + origami_used + descriptor_storage_path

metadata_path = storage_path + "metadata/"
results_path = storage_path + "results/"
plot_path = storage_path + "plots/"


# which rep is performed
script = Path(__file__).stem
rep = int(script.split("_")[::-1][0])

# which seed is chosen (starts from list[1])
random_seed_list = [8738, 5995, 9355, 9598, 2715, 5453, 6287, 5896, 5041, 6416, 3358, 5979, 1791, 6972, 6533, 1071,
                    8281, 5704, 6613, 6609, 5093, 293, 5787, 9740, 1943, 7552, 4037, 2806, 5141, 9906, 8738]

# set experiment name
experiment_name = origami_used + "_300000_random_walk_experiment_rep_" + str(rep)

# set random seed for experiment
set_seed = random_seed_list[rep]  # set random seed
random.seed(set_seed)  # set it for this script
numpy.random.seed(set_seed)
seed_list = [random.randrange(1, 1500001, 1) for _ in range(1500001)]  # get random seeds for random scaffold generation


# 1a. load the JSON settings
settings_default = iohandler.parse_json("settings_default_rna.json")
settings = iohandler.convert_json_data(settings_default, settings_default.copy())

# 1b. load and parse contact map file for the origami
# 		(this is needed: it says how the scaffold is connected to the staples)
st, sc, scaffold_circular = cio.read_origami_contactmap("_input/%s" % origami_contactmap_filename)

# 1c. make origami dict, containing all info
origami = {}

origami["contactmap"] = {}
origami["contactmap"]["st"] = st
origami["contactmap"]["sc"] = sc
origami["contactmap"]["scaffold_circular"] = ""
origami["contactmap"]["scaffold_type"] = "LINEAR"
if scaffold_circular : origami["contactmap"]["scaffold_type"] = "CIRCULAR"

origami["settings"] = settings
origami["precomputed"] = precompute_lookups(origami)




def create_origami_instances(seed_chosen):
    # generates origami instance
    # step a. generate a random scaffold sequence (or supply one of your own
    scaffold53 = generate_random_scaffold(origami, seed_chosen)
    # step a. extended: encode scaffold into numeric values
    scaffold53_encoded = ordinal_encoder(scaffold53)
    return scaffold53_encoded


def custom_evaluation(individual):
    # un-encode scaffold
    scaffold53 = ordinal_decoder(individual[0:])
    # step b. derive staples from individuals scaffold
    staple_set = generate_staples_from_scaffold(origami, scaffold53)
    scores = score_scaffold(origami, scaffold53, staple_set)
    print(scores)
    return scores[0], scores[1], scores[2], scores[3]


creator.create("FitnessMin_", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0,))
creator.create("Individual_", array.array, typecode='d', fitness=creator.FitnessMin_)
toolbox = base.Toolbox()
toolbox.register("map", futures.map)  # <--------------- overload the map function
toolbox.register("attr_item", create_origami_instances)

toolbox.register("evaluate", custom_evaluation)


def main(seed=set_seed):
    # use a random seed for reproducibility of work
    random.seed(seed)

    # set hyper-parameters for the RW
    NGEN = hyper_parameters_dict.get("NGEN")  # how many generations to produce
    MU = hyper_parameters_dict.get("MU")  # the size of the population (I.E: 100 scaffolds)
    TOTAL_RW_POP = 10000  # NGEN * MU

    # create the population
    import itertools as IT
    counter = IT.count(1)
    l = range(TOTAL_RW_POP)
    # every time we iterate in the list comprehension, increase the counter and use a new seed for scaffold generation
    pop = [creator.Individual_(toolbox.attr_item(seed_list[next(counter)])) for i, x in enumerate(l, start=1)]
    print(len(pop))
    print(pop[0])  # DNA sequence of one individual

    scores = []
    for individual in pop:
        with open(metadata_path + experiment_name + 'random_walk_scaffold.txt', 'a') as file:
            file.write(ordinal_decoder(individual) + '\n')
        file.close()

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        scores.append(ind.fitness.values)

        # store scores of sequence
        with open(metadata_path + experiment_name + 'random_walk_scaffold_score.txt', 'a') as file:
            file.write(str(ind.fitness.values) + '\n')
        file.close()
        
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(metadata_path + experiment_name + "random_walk_scaffold_scores" + '.csv')


if __name__ == "__main__":
    main()
